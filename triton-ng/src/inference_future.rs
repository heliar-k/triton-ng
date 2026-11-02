use crate::{InferenceResult, OutputTensor, TritonError};
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

pub(crate) type StreamItem = Result<InferenceResult, String>;
pub(crate) type StreamSender = mpsc::UnboundedSender<StreamItem>;

pub(crate) trait CancelInference: Send + Sync {
    fn cancel(&self);
}

/// Stream of responses to one BLS request, returned by
/// [`crate::Server::infer_stream`].
///
/// Each call to [`InferenceStream::next`] returns one response. A response or
/// submission error is returned as `Err`; FINAL ends the stream. Dropping the
/// stream before FINAL asks Triton to cancel the request on a best-effort
/// basis.
///
/// # Examples
///
/// ```ignore
/// let mut stream = ctx.server().infer_stream(request);
/// while let Some(result) = stream.next().await {
///     let result = result?;
///     consume(result)?;
/// }
/// ```
pub struct InferenceStream {
    receiver: mpsc::UnboundedReceiver<StreamItem>,
    request: Option<Arc<dyn CancelInference>>,
    cancel_on_drop: bool,
    finished: bool,
}

impl InferenceStream {
    pub(crate) fn channel(
        request: Option<Arc<dyn CancelInference>>,
        cancel_on_drop: bool,
    ) -> (StreamSender, Self) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            sender,
            Self {
                receiver,
                request,
                cancel_on_drop,
                finished: false,
            },
        )
    }

    pub(crate) fn failed(message: impl Into<String>) -> Self {
        let (sender, stream) = Self::channel(None, false);
        let _ = sender.send(Err(message.into()));
        stream
    }

    /// Waits for the next response, or `None` after FINAL.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// while let Some(result) = stream.next().await {
    ///     let result = result?;
    ///     println!("FINAL={}", result.is_final);
    /// }
    /// ```
    pub async fn next(&mut self) -> Option<Result<InferenceResult, TritonError>> {
        poll_fn(|cx| {
            self.poll_next(cx)
                .map(|item| item.map(|result| result.map_err(TritonError::from_message)))
        })
        .await
    }

    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Option<StreamItem>> {
        if self.finished {
            return Poll::Ready(None);
        }
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for InferenceStream {
    fn drop(&mut self) {
        if self.cancel_on_drop
            && !self.finished
            && let Some(request) = &self.request
        {
            request.cancel();
        }
    }
}

/// Inference running in Triton, resolving to one aggregated result.
///
/// Outputs of all responses are collected in order and the first error is
/// preserved. The request is submitted before this future is returned, so it
/// runs whether the future is polled or not. Dropping the future does not
/// cancel the request.
pub struct InferenceFuture {
    stream: InferenceStream,
    outputs: Vec<OutputTensor>,
    error: Option<String>,
    finished: bool,
}

impl InferenceFuture {
    pub(crate) fn new(stream: InferenceStream) -> Self {
        Self {
            stream,
            outputs: Vec::new(),
            error: None,
            finished: false,
        }
    }
}

impl InferenceFuture {
    fn poll_result(&mut self, cx: &mut Context<'_>) -> Poll<Result<InferenceResult, String>> {
        if self.finished {
            return Poll::Ready(Err("inference result was already taken".to_string()));
        }

        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(Ok(mut result))) => {
                    self.outputs.append(&mut result.outputs);
                }
                Poll::Ready(Some(Err(error))) => {
                    self.error.get_or_insert(error);
                }
                Poll::Ready(None) => {
                    self.finished = true;
                    return Poll::Ready(match self.error.take() {
                        Some(error) => Err(error),
                        None => Ok(InferenceResult {
                            outputs: std::mem::take(&mut self.outputs),
                            error: None,
                            is_final: true,
                        }),
                    });
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl Future for InferenceFuture {
    type Output = Result<InferenceResult, TritonError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut()
            .poll_result(cx)
            .map_err(TritonError::from_message)
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceFuture, InferenceStream};
    use crate::inference_result::OutputData;
    use crate::{DataType, InferenceResult, OutputTensor};
    use std::future::poll_fn;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn result(is_final: bool) -> InferenceResult {
        InferenceResult {
            outputs: vec![],
            error: None,
            is_final,
        }
    }

    fn result_with_output(is_final: bool, name: &str) -> InferenceResult {
        InferenceResult {
            outputs: vec![OutputTensor {
                name: name.to_string(),
                shape: vec![0],
                datatype: DataType::Fp32,
                data: OutputData::Host(Vec::new()),
            }],
            error: None,
            is_final,
        }
    }

    async fn next(stream: &mut InferenceStream) -> Option<Result<InferenceResult, String>> {
        poll_fn(|cx| stream.poll_next(cx)).await
    }

    async fn collect(future: &mut InferenceFuture) -> Result<InferenceResult, String> {
        poll_fn(|cx| future.poll_result(cx)).await
    }

    fn poll_future(future: &mut InferenceFuture, waker: &Waker) -> Poll<Result<usize, String>> {
        future
            .poll_result(&mut Context::from_waker(waker))
            .map(|result| result.map(|result| result.outputs.len()))
    }

    #[tokio::test]
    async fn failed_stream_yields_error_then_end() {
        let mut stream = InferenceStream::failed("submit failed");
        assert!(next(&mut stream).await.unwrap().is_err());
        assert!(next(&mut stream).await.is_none());
    }

    #[tokio::test]
    async fn stream_yields_each_response_and_preserves_final_flag() {
        let (sender, mut stream) = InferenceStream::channel(None, true);
        sender.send(Ok(result(false))).unwrap();
        sender.send(Ok(result(true))).unwrap();
        drop(sender);

        assert!(!next(&mut stream).await.unwrap().unwrap().is_final);
        assert!(next(&mut stream).await.unwrap().unwrap().is_final);
        assert!(next(&mut stream).await.is_none());
        assert!(next(&mut stream).await.is_none());
    }

    #[test]
    fn stream_wakes_pending_future() {
        let counter = Arc::new(CountingWaker(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&counter));
        let (sender, stream) = InferenceStream::channel(None, false);
        let mut future = InferenceFuture::new(stream);

        assert!(poll_future(&mut future, &waker).is_pending());
        sender.send(Ok(result(true))).unwrap();
        drop(sender);

        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        assert!(poll_future(&mut future, &waker).is_ready());
    }

    #[tokio::test]
    async fn completion_can_arrive_from_another_thread() {
        let (sender, stream) = InferenceStream::channel(None, false);
        let mut future = InferenceFuture::new(stream);
        std::thread::spawn(move || {
            sender.send(Ok(result(true))).unwrap();
            drop(sender);
        })
        .join()
        .unwrap();

        assert!(collect(&mut future).await.unwrap().is_final);
    }

    #[tokio::test]
    async fn failed_submission_resolves_to_error() {
        let mut future = InferenceFuture::new(InferenceStream::failed("submit failed"));
        assert_eq!(collect(&mut future).await.err().unwrap(), "submit failed");
    }

    #[tokio::test]
    async fn aggregate_preserves_first_error() {
        let (sender, stream) = InferenceStream::channel(None, false);
        sender.send(Err("first".to_string())).unwrap();
        sender.send(Err("second".to_string())).unwrap();
        drop(sender);

        let mut future = InferenceFuture::new(stream);
        assert_eq!(collect(&mut future).await.err().unwrap(), "first");
    }

    #[tokio::test]
    async fn aggregate_collects_outputs_and_is_final() {
        let (sender, stream) = InferenceStream::channel(None, false);
        sender.send(Ok(result_with_output(false, "first"))).unwrap();
        sender.send(Ok(result_with_output(true, "second"))).unwrap();
        drop(sender);

        let mut future = InferenceFuture::new(stream);
        let result = collect(&mut future).await.unwrap();
        assert!(result.is_final);
        assert_eq!(result.outputs.len(), 2);
        assert_eq!(result.outputs[0].name, "first");
        assert_eq!(result.outputs[1].name, "second");
    }

    #[test]
    fn polling_after_completion_is_an_error() {
        let mut future = InferenceFuture::new(InferenceStream::failed("first"));
        assert!(poll_future(&mut future, Waker::noop()).is_ready());
        assert_eq!(
            poll_future(&mut future, Waker::noop()),
            Poll::Ready(Err("inference result was already taken".to_string()))
        );
    }
}
