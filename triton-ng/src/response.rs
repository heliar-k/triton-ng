use crate::Output;
use crate::error::TritonError;
use crate::request::Request;
use crate::timing::{PendingResponseTimestamps, ResponseTimestamps, monotonic_nanos};
use crate::types::{DataType, Parameter};
use crate::utils::cstring_from_str;
use std::ptr;
use triton_ng_macros::triton_call;

const FINAL: u32 =
    triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL;

#[derive(Clone, Copy, Debug, PartialEq)]
enum DropAction {
    SendErrorAndFinal,
    ReleaseRequest,
    DeleteOnly,
}

#[derive(Default)]
struct FactoryState {
    has_response: bool,
    final_sent: bool,
    released: bool,
}

impl FactoryState {
    fn can_create_response(&self) -> Result<(), String> {
        if self.final_sent {
            Err("the response stream already sent FINAL".to_string())
        } else {
            Ok(())
        }
    }

    fn can_release(&self) -> Result<(), String> {
        if self.final_sent {
            Err("the request cannot be released after FINAL".to_string())
        } else if self.released {
            Err("the request was already released".to_string())
        } else {
            Ok(())
        }
    }

    fn can_send_final(&self) -> Result<(), String> {
        if self.final_sent {
            Err("the response stream already sent FINAL".to_string())
        } else {
            Ok(())
        }
    }

    fn mark_response_sent(&mut self) {
        self.has_response = true;
    }

    fn mark_released(&mut self) -> Result<(), String> {
        self.can_release()?;
        self.released = true;
        Ok(())
    }

    fn mark_final(&mut self) -> Result<(), String> {
        self.can_send_final()?;
        self.final_sent = true;
        Ok(())
    }

    fn drop_action(&self) -> DropAction {
        match (self.final_sent, self.released) {
            (false, false) => DropAction::SendErrorAndFinal,
            (true, false) => DropAction::ReleaseRequest,
            (_, true) => DropAction::DeleteOnly,
        }
    }
}

/// Factory of a stream of responses to one [`Request`].
///
/// The factory owns the request. Responses created by [`ResponseFactory::response`]
/// borrow the factory mutably, so outputs of one response are completed before
/// the next response is created. [`ResponseFactory::release_request`] releases
/// the request and its inputs early; the factory can still create responses
/// and query cancellation afterwards. `Send`, not `Sync`.
///
/// Dropping a factory before [`ResponseFactory::finish`] sends an error
/// response and FINAL when the request is still owned. After an early request
/// release, dropping the factory only deletes the C factory, matching
/// `TRITONBACKEND_ResponseFactoryDelete`.
///
/// # Examples
///
/// ```ignore
/// let mut factory = ResponseFactory::new(request)?;
/// for tokens in chunks {
///     let mut response = factory.response()?;
///     response
///         .create_output("TOKENS", DataType::Int64, &[1, tokens.len() as i64])?
///         .write_i64_vec(&tokens)?;
///     response.send()?;
/// }
/// factory.finish()?;
/// ```
pub struct ResponseFactory {
    ptr: *mut triton_ng_sys::TRITONBACKEND_ResponseFactory,
    request: Request,
    state: FactoryState,
}

// SAFETY: a factory can be moved to another thread. All response construction
// requires `&mut self`, and the type is not `Sync`, so one request is never
// constructed from several threads at once.
unsafe impl Send for ResponseFactory {}

impl ResponseFactory {
    /// Creates a response factory and transfers request ownership to it.
    /// On failure the request sends an error response while being dropped.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut factory = ResponseFactory::new(request)?;
    /// factory.finish()?;
    /// ```
    pub fn new(mut request: Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryNew(
            &mut _,
            request.as_ptr()
        ))?;
        request.manage_with_factory();
        Ok(Self {
            ptr,
            request,
            state: FactoryState::default(),
        })
    }

    /// Creates the next non-final response, borrowing this factory until that
    /// response is sent or dropped.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut response = factory.response()?;
    /// response.create_output("VALUE", DataType::Int64, &[1])?.write_i64_vec(&[1])?;
    /// response.send()?;
    /// ```
    pub fn response(&mut self) -> Result<Response<'_>, TritonError> {
        self.state
            .can_create_response()
            .map_err(TritonError::from_message)?;
        let timing = PendingResponseTimestamps::new();
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNewFromFactory(
            &mut _,
            self.ptr
        ))?;
        Ok(Response {
            ptr,
            owner: ResponseOwner::Factory(self),
            pending_outputs: 0,
            sent: false,
            timing,
        })
    }

    /// Releases the request and its inputs before the response stream ends.
    /// The factory remains usable for responses and cancellation queries.
    /// Repeated release, or release after FINAL, fails. Request views and
    /// inputs must not be used after this call.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut factory = ResponseFactory::new(request)?;
    /// factory.release_request()?;
    /// factory.finish()?;
    /// ```
    pub fn release_request(&mut self) -> Result<(), TritonError> {
        self.state
            .can_release()
            .map_err(TritonError::from_message)?;
        self.request.release()?;
        self.state
            .mark_released()
            .map_err(TritonError::from_message)?;
        Ok(())
    }

    /// Returns `true` if the client cancelled the request. This remains valid
    /// after [`ResponseFactory::release_request`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if factory.is_cancelled()? {
    ///     factory.finish()?;
    ///     return Ok(());
    /// }
    /// ```
    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        let mut cancelled = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryIsCancelled(
            self.ptr,
            &mut cancelled
        ))?;
        Ok(cancelled)
    }

    /// Sends FINAL without a response, releases the request when needed, and
    /// consumes the factory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// factory.finish()?;
    /// ```
    pub fn finish(mut self) -> Result<(), TritonError> {
        self.send_final()?;
        self.release_after_final()?;
        Ok(())
    }

    fn send_final(&mut self) -> Result<(), TritonError> {
        self.state
            .can_send_final()
            .map_err(TritonError::from_message)?;
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactorySendFlags(
            self.ptr, FINAL
        ))?;
        self.state.mark_final().map_err(TritonError::from_message)
    }

    fn release_after_final(&mut self) -> Result<(), TritonError> {
        if !self.state.released {
            self.request.release()?;
            self.state.released = true;
        }
        Ok(())
    }

    fn send_error_internal(&mut self, message: &str) -> Result<(), TritonError> {
        self.state
            .can_create_response()
            .map_err(TritonError::from_message)?;
        let error = TritonError::internal(message);
        if error.as_ptr().is_null() {
            return Err(TritonError::from_message("failed to create error object"));
        }
        let timing = PendingResponseTimestamps::new();
        let response = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNewFromFactory(
            &mut _,
            self.ptr
        ))?;
        let timing = timing.finish(monotonic_nanos());
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            0,
            error.as_ptr()
        ))?;
        self.state.mark_response_sent();
        self.request.timestamps.record_response(timing);
        Ok(())
    }
}

impl Drop for ResponseFactory {
    fn drop(&mut self) {
        match self.state.drop_action() {
            DropAction::SendErrorAndFinal => {
                if let Err(e) =
                    self.send_error_internal("backend did not finish the response stream")
                {
                    crate::log::error(&format!("failed to send error response: {e}"));
                }
                if let Err(e) = self.send_final() {
                    crate::log::error(&format!("failed to send FINAL: {e}"));
                }
            }
            DropAction::ReleaseRequest => {
                if let Err(e) = self.release_after_final() {
                    crate::log::error(&format!("failed to release request: {e}"));
                }
            }
            DropAction::DeleteOnly => {}
        }

        if let Err(e) = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryDelete(self.ptr)) {
            crate::log::error(&format!("failed to delete response factory: {e}"));
        }
    }
}

enum ResponseOwner<'a> {
    Owned(Request),
    Factory(&'a mut ResponseFactory),
}

impl ResponseOwner<'_> {
    fn request(&self) -> &Request {
        match self {
            Self::Owned(request) => request,
            Self::Factory(factory) => &factory.request,
        }
    }

    fn send_flags(&self) -> u32 {
        match self {
            Self::Owned(_) => FINAL,
            Self::Factory(_) => 0,
        }
    }

    fn record_response(&mut self, timing: ResponseTimestamps) {
        match self {
            Self::Owned(request) => {
                request.mark_responded();
                request.timestamps.record_response(timing);
            }
            Self::Factory(factory) => {
                factory.state.mark_response_sent();
                factory.request.timestamps.record_response(timing);
            }
        }
    }
}

/// A response of a [`Request`], either owned directly or borrowed from a
/// [`ResponseFactory`].
///
/// [`Response::new`] returns `Response<'static>` and sends one FINAL response.
/// [`ResponseFactory::response`] returns `Response<'f>` and sends a non-final
/// response. An unsent response is deleted on Drop and replaced by an error
/// response. `Send`, not `Sync`.
///
/// # Examples
///
/// ```ignore
/// let scores = request.get_input("SCORES")?.as_fp32_vec()?;
/// let mut response = Response::new(request)?;
/// response
///     .create_output("SCORES", DataType::Fp32, &[1, scores.len() as i64])?
///     .write_fp32_vec(&scores)?;
/// response.send()?;
/// ```
pub struct Response<'a> {
    /// Null once the response was handed over to `TRITONBACKEND_ResponseSend`,
    /// which takes ownership whatever it returns.
    ptr: *mut triton_ng_sys::TRITONBACKEND_Response,
    owner: ResponseOwner<'a>,
    /// Outputs created but not written yet. Triton frontends read every
    /// output of a sent response, so one without a buffer crashes the server.
    pending_outputs: usize,
    sent: bool,
    timing: PendingResponseTimestamps,
}

// SAFETY: see `Request` and `ResponseFactory`. Outputs are created through
// `&mut self`, so they are never added from several threads at once.
unsafe impl Send for Response<'_> {}

impl Response<'static> {
    /// Creates the sole final response of `request`. On error the request is
    /// dropped and gets an error response.
    pub fn new(request: Request) -> Result<Self, TritonError> {
        let timing = PendingResponseTimestamps::new();
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(
            &mut _,
            request.as_ptr()
        ))?;
        Ok(Self {
            ptr,
            owner: ResponseOwner::Owned(request),
            pending_outputs: 0,
            sent: false,
            timing,
        })
    }
}

impl Response<'_> {
    /// The request this response answers, e.g. to read its inputs. Reading a
    /// request released early through its factory returns an error.
    pub fn request(&self) -> &Request {
        self.owner.request()
    }

    /// Creates an output tensor. It must be written exactly once before
    /// [`Response::send`].
    pub fn create_output(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<Output<'_>, TritonError> {
        if datatype == DataType::Invalid {
            return Err(TritonError::from_message(format!(
                "output '{name}' has invalid datatype"
            )));
        }
        let elements = element_count(shape)
            .map_err(|e| TritonError::from_message(format!("output '{name}': {e}")))?;

        let name_cstr = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseOutput(
            self.ptr,
            &mut _,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u32,
        ))?;

        self.pending_outputs += 1;

        Ok(Output::new(
            ptr,
            name.to_string(),
            datatype,
            elements,
            &mut self.pending_outputs,
            &mut self.timing.compute_output_start,
        ))
    }

    /// Adds the parameter `name` to the response. Clients read it from
    /// `parameters` of the gRPC `ModelInferResponse`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut response = Response::new(request)?;
    /// response.set_parameter("tokens", 42)?;
    /// response.set_parameter("language", "ru")?;
    /// response.send()?;
    /// ```
    pub fn set_parameter(
        &mut self,
        name: &str,
        value: impl Into<Parameter>,
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;
        match value.into() {
            Parameter::Bool(value) => {
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetBoolParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::Int(value) => {
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetIntParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::Double(value) => {
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetDoubleParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value
                ))
            }
            Parameter::String(value) => {
                let value_cstr = cstring_from_str(&value)?;
                triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSetStringParameter(
                    self.ptr,
                    name_cstr.as_ptr(),
                    value_cstr.as_ptr()
                ))
            }
        }
    }

    /// Sends this response. An owned response carries FINAL; a response
    /// borrowed from a factory is non-final.
    pub fn send(mut self) -> Result<(), TritonError> {
        if self.pending_outputs > 0 {
            return Err(TritonError::from_message(format!(
                "{} output(s) of the response were created but not written",
                self.pending_outputs
            )));
        }

        let response = std::mem::replace(&mut self.ptr, ptr::null_mut());
        let timing = self.timing.finish(monotonic_nanos());
        let flags = self.owner.send_flags();
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            flags,
            ptr::null_mut(),
        ))?;

        self.sent = true;
        self.owner.record_response(timing);
        Ok(())
    }
}

impl Drop for Response<'_> {
    fn drop(&mut self) {
        if self.sent {
            return;
        }
        if !self.ptr.is_null() {
            unsafe { triton_ng_sys::TRITONBACKEND_ResponseDelete(self.ptr) };
        }

        let error = "backend did not send the response";
        match &mut self.owner {
            ResponseOwner::Owned(request) => {
                if let Err(e) = request.respond_with(TritonError::internal(error)) {
                    crate::log::error(&format!("failed to send error response: {e}"));
                }
            }
            ResponseOwner::Factory(factory) => {
                if let Err(e) = factory.send_error_internal(error) {
                    crate::log::error(&format!("failed to send error response: {e}"));
                }
            }
        }
    }
}

/// Number of elements of a tensor with `shape`.
pub(crate) fn element_count(shape: &[i64]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim)
            .map_err(|_| format!("invalid dimension {dim} in shape {shape:?}"))?;
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("shape {shape:?} is too large"))
    })
}

#[cfg(test)]
mod tests {
    use super::{DropAction, FactoryState, element_count};

    #[test]
    fn factory_allows_responses_until_final_even_after_release() {
        let mut state = FactoryState::default();
        state.can_create_response().unwrap();
        state.mark_released().unwrap();
        state.can_create_response().unwrap();
        state.mark_final().unwrap();
        assert!(state.can_create_response().is_err());
    }

    #[test]
    fn factory_releases_only_once_and_before_final() {
        let mut state = FactoryState::default();
        state.mark_released().unwrap();
        assert!(state.mark_released().is_err());

        let mut state = FactoryState::default();
        state.mark_final().unwrap();
        assert!(state.mark_released().is_err());
        assert!(state.mark_final().is_err());
    }

    #[test]
    fn factory_drop_action_uses_final_and_release_state() {
        for has_response in [false, true] {
            for final_sent in [false, true] {
                for released in [false, true] {
                    let state = FactoryState {
                        has_response,
                        final_sent,
                        released,
                    };
                    let expected = match (final_sent, released) {
                        (false, false) => DropAction::SendErrorAndFinal,
                        (true, false) => DropAction::ReleaseRequest,
                        (_, true) => DropAction::DeleteOnly,
                    };
                    assert_eq!(state.drop_action(), expected);
                }
            }
        }
    }

    #[test]
    fn factory_records_sent_responses() {
        let mut state = FactoryState::default();
        assert!(!state.has_response);
        state.mark_response_sent();
        assert!(state.has_response);
    }

    #[test]
    fn element_count_multiplies_dimensions() {
        assert_eq!(element_count(&[2, 3, 4]), Ok(24));
        assert_eq!(element_count(&[1, 0, 2]), Ok(0));
        assert_eq!(element_count(&[]), Ok(1));
    }

    #[test]
    fn element_count_rejects_negative_dimensions() {
        assert!(element_count(&[1, -1]).is_err());
    }

    #[test]
    fn element_count_rejects_overflow() {
        assert!(element_count(&[i64::MAX, i64::MAX]).is_err());
    }
}
