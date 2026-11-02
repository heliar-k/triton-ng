use crate::inference_future::{CancelInference, InferenceStream, StreamItem, StreamSender};
use crate::inference_request::InputBuffers;
use crate::inference_response::InferenceResponse;
use crate::message::Message;
use crate::response_allocator::{OutputPlan, ResponseAllocator};
use crate::utils::cstring_from_str;
use crate::{InferenceFuture, InferenceRequest, InferenceResult, TritonError};
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use triton_ng_macros::triton_call;

#[derive(Default)]
struct RequestState {
    final_seen: bool,
    released: bool,
    deleted: bool,
}

impl RequestState {
    fn complete(&mut self) -> bool {
        self.final_seen = true;
        self.mark_deleted_when_ready()
    }

    fn release(&mut self) -> bool {
        self.released = true;
        self.mark_deleted_when_ready()
    }

    fn can_cancel(&self) -> bool {
        !self.final_seen && !self.deleted
    }

    fn mark_deleted_when_ready(&mut self) -> bool {
        if self.final_seen && self.released && !self.deleted {
            self.deleted = true;
            return true;
        }
        false
    }
}

/// Synchronizes cancellation with the response and release callbacks.
pub(crate) struct RequestControl {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    state: Mutex<RequestState>,
}

// SAFETY: the raw request is accessed only while holding `state`. It is
// deleted once FINAL and RELEASE_ALL have both arrived, and cancellation is
// excluded after deletion.
unsafe impl Send for RequestControl {}
unsafe impl Sync for RequestControl {}

impl RequestControl {
    fn new(ptr: *mut triton_ng_sys::TRITONSERVER_InferenceRequest) -> Self {
        Self {
            ptr,
            state: Mutex::new(RequestState::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, RequestState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn complete(&self) {
        if self.lock().complete() {
            self.delete();
        }
    }

    fn release(&self) {
        if self.lock().release() {
            self.delete();
        }
    }

    fn delete(&self) {
        let error = unsafe { triton_ng_sys::TRITONSERVER_InferenceRequestDelete(self.ptr) };
        if !error.is_null() {
            let error = unsafe { TritonError::new(error) };
            crate::log::error(&format!("failed to delete inference request: {error}"));
        }
    }
}

impl CancelInference for RequestControl {
    fn cancel(&self) {
        let state = self.lock();
        if !state.can_cancel() {
            return;
        }
        // Cancel only sets the request's cancellation state in Triton. Holding
        // the lock prevents FINAL and release from deleting the request during
        // this call.
        let error = unsafe { triton_ng_sys::TRITONSERVER_InferenceRequestCancel(self.ptr) };
        if !error.is_null() {
            let error = unsafe { TritonError::new(error) };
            crate::log::error(&format!("failed to cancel inference request: {error}"));
        }
        drop(state);
    }
}

struct ReleaseContext {
    _buffers: InputBuffers,
    control: Arc<RequestControl>,
}

/// What the response callback of one BLS request needs, handed to Triton as
/// `response_userp` and freed with the final response.
pub(crate) struct InferenceContext {
    sender: StreamSender,
    allocator: ResponseAllocator,
    outputs: OutputPlan,
    control: Arc<RequestControl>,
}

/// The Triton server the backend runs in, used for BLS: inference of other
/// models from a backend. Batches get it from [`crate::Context::server`].
///
/// Borrows the `TRITONSERVER_Server`: Triton owns it, so there is nothing to
/// drop. The server stays valid while any model is loaded, and a model stays
/// loaded while any of its [`crate::Request`]s is alive. `Send + Sync + Clone`.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     let state = ctx.state::<State>()?;
///     let mut inference = InferenceRequest::new(ctx.server(), &state.encoder, -1)?;
///     let result = ctx.server().infer_async(inference).await?;
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct Server {
    ptr: *mut triton_ng_sys::TRITONSERVER_Server,
}

// SAFETY: the pointer is the TRITONSERVER_Server, which the frontends call
// from many threads at once. TRITONSERVER_ServerDelete waits for in-flight
// requests and for every model to unload before freeing it, and a model is
// not unloaded while any of its requests is alive, so the pointer outlives
// every `Request` the backend holds. Only an expired exit timeout on shutdown
// breaks this, when the process is exiting.
unsafe impl Send for Server {}
unsafe impl Sync for Server {}

impl Server {
    pub(crate) fn from_ptr(
        ptr: *mut triton_ng_sys::TRITONSERVER_Server,
    ) -> Result<Self, TritonError> {
        if ptr.is_null() {
            return Err(TritonError::from_message(
                "TRITONBACKEND_ModelServer returned null",
            ));
        }
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONSERVER_Server {
        self.ptr
    }

    pub fn api_version(&self) -> Result<(u32, u32), TritonError> {
        let mut major: u32 = 0;
        let mut minor: u32 = 0;

        triton_call!(triton_ng_sys::TRITONSERVER_ApiVersion(
            &mut major, &mut minor
        ))?;

        Ok((major, minor))
    }

    pub fn is_ready(&self) -> Result<bool, TritonError> {
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerIsReady(
            self.ptr, &mut ready
        ))?;

        Ok(ready)
    }

    pub fn is_model_ready(&self, model_name: &str, version: i64) -> Result<bool, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let mut ready = false;

        triton_call!(triton_ng_sys::TRITONSERVER_ServerModelIsReady(
            self.ptr,
            model_name_cstr.as_ptr(),
            version,
            &mut ready
        ))?;

        Ok(ready)
    }

    pub fn model_metadata(&self, model_name: &str, version: i64) -> Result<String, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let metadata = Message::from_ptr(triton_call!(
            triton_ng_sys::TRITONSERVER_ServerModelMetadata(
                self.ptr,
                model_name_cstr.as_ptr(),
                version,
                &mut _,
            )
        )?);

        metadata.to_json()
    }

    /// Submits inference and returns a future that aggregates every response
    /// into one final result. Submission errors are reported by the future.
    /// Dropping the future does not cancel the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let result = server.infer_async(inference).await?;
    /// let embedding = result.output("EMBEDDING")?.as_fp32_vec()?;
    /// ```
    pub fn infer_async(&self, request: InferenceRequest) -> InferenceFuture {
        InferenceFuture::new(self.submit(request, false))
    }

    /// Submits inference and returns each response through an
    /// [`InferenceStream`]. Requests with preallocated output buffers are
    /// rejected before submission because one buffer cannot receive multiple
    /// responses. Dropping the stream before FINAL cancels the request on a
    /// best-effort basis.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut stream = server.infer_stream(inference);
    /// while let Some(result) = stream.next().await {
    ///     let result = result?;
    ///     println!("{} output(s), FINAL={}", result.outputs.len(), result.is_final);
    /// }
    /// ```
    pub fn infer_stream(&self, request: InferenceRequest) -> InferenceStream {
        if let Err(error) = request.ensure_streaming_compatible() {
            return InferenceStream::failed(error.to_string());
        }
        self.submit(request, true)
    }

    fn submit(&self, request: InferenceRequest, cancel_on_drop: bool) -> InferenceStream {
        let allocator = match ResponseAllocator::new() {
            Ok(allocator) => allocator,
            Err(error) => return InferenceStream::failed(error.to_string()),
        };
        let (raw, buffers, outputs) = request.into_raw_parts();
        let control = Arc::new(RequestControl::new(raw));
        let cancellation: Arc<dyn CancelInference> = control.clone();
        let (sender, stream) = InferenceStream::channel(Some(cancellation), cancel_on_drop);

        if let Err(error) =
            self.submit_raw(raw, buffers, outputs, allocator, &control, sender.clone())
        {
            let _ = sender.send(Err(error.to_string()));
        }
        stream
    }

    fn submit_raw(
        &self,
        raw: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
        buffers: InputBuffers,
        outputs: crate::inference_request::OutputBuffers,
        allocator: ResponseAllocator,
        control: &Arc<RequestControl>,
        sender: StreamSender,
    ) -> Result<(), TritonError> {
        let release_context = Box::new(ReleaseContext {
            _buffers: buffers,
            control: Arc::clone(control),
        });
        let release_ptr = Box::into_raw(release_context);

        if let Err(error) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                raw,
                Some(inference_request_release),
                release_ptr.cast(),
            )
        ) {
            unsafe { cleanup_unsubmitted(raw, release_ptr) };
            return Err(error);
        }

        let context = Box::new(InferenceContext {
            sender,
            allocator,
            outputs: OutputPlan::new(outputs),
            control: Arc::clone(control),
        });
        let context_ptr = Box::into_raw(context);
        let context_ref = unsafe { &*context_ptr };

        if let Err(error) = triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                raw,
                context_ref.allocator.as_ptr(),
                ptr::from_ref(&context_ref.outputs).cast_mut().cast(),
                Some(inference_response_complete),
                context_ptr.cast(),
            )
        ) {
            unsafe {
                drop(Box::from_raw(context_ptr));
                cleanup_unsubmitted(raw, release_ptr);
            }
            return Err(error);
        }

        if let Err(error) = triton_call!(triton_ng_sys::TRITONSERVER_ServerInferAsync(
            self.ptr,
            raw,
            ptr::null_mut(),
        )) {
            // Submission failure guarantees neither callback ran.
            unsafe {
                drop(Box::from_raw(context_ptr));
                cleanup_unsubmitted(raw, release_ptr);
            }
            return Err(error);
        }

        Ok(())
    }
}

/// Cleans up a request that Triton never started. FINAL is marked so the
/// release path can delete the request immediately.
unsafe fn cleanup_unsubmitted(
    request: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    release_ptr: *mut ReleaseContext,
) {
    let control = unsafe { &(*release_ptr).control };
    control.complete();
    unsafe { inference_request_release(request, RELEASE_ALL, release_ptr.cast()) };
}

/// Called by Triton after the backend has fully released the inference request.
/// Input buffers are freed here. The small request object is deleted once the
/// FINAL response has also arrived, so a dropped stream can still cancel after
/// an early downstream request release.
unsafe extern "C" fn inference_request_release(
    _request: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    flags: u32,
    userp: *mut c_void,
) {
    if flags & RELEASE_ALL == 0 || userp.is_null() {
        return;
    }
    let context = unsafe { Box::from_raw(userp as *mut ReleaseContext) };
    context.control.release();
    drop(context);
}

const RELEASE_ALL: u32 =
    triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL;

fn read_response(
    response_ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
    is_final: bool,
) -> StreamItem {
    let response = InferenceResponse::from_ptr(response_ptr).map_err(|error| error.to_string())?;
    if let Some(error) = response.error() {
        return Err(error.to_string());
    }
    let outputs = response.outputs().map_err(|error| error.to_string())?;
    Ok(InferenceResult {
        outputs,
        error: None,
        is_final,
    })
}

/// Called by Triton for each response, and for an empty FINAL that only marks
/// end-of-stream. The callback context is reclaimed only with FINAL.
unsafe extern "C" fn inference_response_complete(
    response_ptr: *mut triton_ng_sys::TRITONSERVER_InferenceResponse,
    flags: u32,
    userp: *mut c_void,
) {
    const FINAL: u32 =
        triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;
    let is_final = flags & FINAL != 0;

    if userp.is_null() {
        if !response_ptr.is_null() {
            drop(InferenceResponse::from_ptr(response_ptr));
        }
        return;
    }

    let context = unsafe { &mut *(userp as *mut InferenceContext) };
    if !response_ptr.is_null() {
        let result = read_response(response_ptr, is_final);
        // A closed receiver discards the result. `read_response` has already
        // deleted the C response and retained only owned Rust output data.
        let _ = context.sender.send(result);
    }

    if is_final {
        context.control.complete();
        unsafe { drop(Box::from_raw(userp as *mut InferenceContext)) };
    }
}

#[cfg(test)]
mod tests {
    use super::RequestState;

    #[test]
    fn request_is_deleted_after_release_then_final() {
        let mut state = RequestState::default();
        assert!(!state.release());
        assert!(state.can_cancel());
        assert!(state.complete());
        assert!(!state.can_cancel());
        assert!(!state.complete());
        assert!(!state.release());
    }

    #[test]
    fn request_is_deleted_after_final_then_release() {
        let mut state = RequestState::default();
        assert!(!state.complete());
        assert!(!state.can_cancel());
        assert!(state.release());
        assert!(!state.release());
    }

    #[test]
    fn unfinished_request_is_not_deleted_or_cancelled_after_final() {
        let mut state = RequestState::default();
        assert!(state.can_cancel());
        assert!(!state.complete());
        assert!(!state.can_cancel());
        assert!(!state.complete());
    }
}
