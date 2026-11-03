use crate::Input;
use crate::error::{Error, TritonError};
use crate::model_instance::InstanceHandle;
use crate::timing::{PendingResponseTimestamps, RequestTimestamps, monotonic_nanos};
use crate::types::{Parameter, RequestFlags};
use crate::utils::{cstr_to_string, cstring_from_str};
use libc::{c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use triton_ng_macros::triton_call;

/// Inference request passed to [`crate::Backend::model_instance_execute`].
///
/// The backend owns the request and can answer it from any thread of the
/// instance runtime. [`crate::Response::new`] produces one final response;
/// [`crate::ResponseFactory`] produces a stream. A request dropped without
/// response handling gets an error response. The request is released to
/// Triton when it is dropped, or earlier with
/// [`crate::ResponseFactory::release_request`].
///
/// Owns the `TRITONBACKEND_Request`. `Send`, not `Sync`: Triton allows a
/// request to be used from one thread at a time.
///
/// # Examples
///
/// ```ignore
/// async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///     for request in requests {
///         let text = request.get_input("TEXT")?.as_string()?;
///         let mut response = Response::new(request)?;
///         response
///             .create_output("TEXT", DataType::Bytes, &[1])?
///             .write_string(&text.to_uppercase())?;
///         response.send()?;
///     }
///     Ok(())
/// }
/// ```
pub struct Request {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    responded: bool,
    managed_by_factory: bool,
    statistics_reported: bool,
    pub(crate) instance: Option<InstanceHandle>,
    pub(crate) timestamps: RequestTimestamps,
}

// SAFETY: once ModelInstanceExecute succeeds, Triton hands the request over to
// the backend, which may respond to and release it from any thread, as long as
// one request is not used from several threads at once. `Request` is not
// `Sync`, so it never is.
unsafe impl Send for Request {}

impl Request {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self {
            ptr,
            responded: false,
            managed_by_factory: false,
            statistics_reported: false,
            instance: None,
            timestamps: RequestTimestamps::new(),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut triton_ng_sys::TRITONBACKEND_Request {
        self.ptr
    }

    pub(crate) fn set_instance(&mut self, instance: InstanceHandle) {
        debug_assert!(!instance.as_ptr().is_null());
        self.instance = Some(instance);
    }

    pub(crate) fn start_compute(&mut self) {
        self.timestamps.start_compute(monotonic_nanos());
    }

    pub(crate) fn mark_responded(&mut self) {
        self.responded = true;
    }

    pub(crate) fn manage_with_factory(&mut self) {
        self.managed_by_factory = true;
    }

    /// Borrows this request for reading without transferring ownership.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = request.view();
    /// let input = view.get_input("INPUT")?;
    /// ```
    pub fn view(&self) -> RequestView<'_> {
        // SAFETY: the returned view cannot outlive this request.
        unsafe { RequestView::from_ptr(self.ptr) }
    }

    /// Lists the input tensor names.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let names = request.input_names()?;
    /// ```
    pub fn input_names(&self) -> Result<Vec<String>, TritonError> {
        self.view().input_names()
    }

    /// Lists the requested output tensor names.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let names = request.output_names()?;
    /// ```
    pub fn output_names(&self) -> Result<Vec<String>, TritonError> {
        self.view().output_names()
    }

    /// Gets an input tensor borrowed from this request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let input = request.get_input("INPUT")?;
    /// ```
    pub fn get_input(&self, name: &str) -> Result<Input<'_>, TritonError> {
        self.view().get_input(name)
    }

    /// Returns a request parameter, see [`RequestView::parameter`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let verbose = request.parameter("verbose")? == Some(Parameter::Bool(true));
    /// ```
    pub fn parameter(&self, key: &str) -> Result<Option<Parameter>, Error> {
        self.view().parameter(key)
    }

    /// Returns the client-supplied request ID, or an empty string when no ID
    /// was supplied. Fails when the request was released or Triton rejects
    /// the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let id = request.id()?;
    /// ```
    pub fn id(&self) -> Result<String, TritonError> {
        self.view().id()
    }

    /// Returns the numeric correlation ID. Zero means no correlation ID.
    /// Fails when the request has a string correlation ID, was released, or
    /// Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let sequence = request.correlation_id()?;
    /// ```
    pub fn correlation_id(&self) -> Result<u64, TritonError> {
        self.view().correlation_id()
    }

    /// Returns the string correlation ID. Fails when the request has a numeric
    /// correlation ID, including the default value of zero. Also fails when
    /// the request was released or Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let sequence = request.correlation_id_string()?;
    /// ```
    pub fn correlation_id_string(&self) -> Result<String, TritonError> {
        self.view().correlation_id_string()
    }

    /// Returns the sequence-start and sequence-end flags. Fails when the
    /// request was released or Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if request.flags()?.sequence_start {
    ///     // Initialize sequence state.
    /// }
    /// ```
    pub fn flags(&self) -> Result<RequestFlags, TritonError> {
        self.view().flags()
    }

    /// Returns the request timeout in microseconds. Zero means no timeout.
    /// Fails when the request was released or Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let timeout_us = request.timeout()?;
    /// ```
    pub fn timeout(&self) -> Result<u64, TritonError> {
        self.view().timeout()
    }

    /// Returns `true` if the client cancelled the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if request.is_cancelled()? {
    ///     request.respond_cancelled()?;
    ///     return Ok(());
    /// }
    /// ```
    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        self.view().is_cancelled()
    }

    /// Sends a final response carrying `message` as an internal error.
    ///
    /// Use it to fail a single request of a batch without failing the others.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Err(e) = validate(&request) {
    ///     request.respond_error(&e.to_string())?;
    /// }
    /// ```
    pub fn respond_error(mut self, message: &str) -> Result<(), TritonError> {
        self.respond_with(TritonError::internal(message))
    }

    /// Sends a final response with the CANCELLED status, see
    /// [`Request::is_cancelled`].
    pub fn respond_cancelled(mut self) -> Result<(), TritonError> {
        self.respond_with(TritonError::cancelled("request was cancelled"))
    }

    pub(crate) fn respond_with(&mut self, error: TritonError) -> Result<(), TritonError> {
        if self.responded {
            return Err(TritonError::from_message(
                "the request already has a final response",
            ));
        }
        if self.ptr.is_null() {
            return Err(TritonError::from_message("the request was released"));
        }
        if error.as_ptr().is_null() {
            return Err(TritonError::from_message("failed to create error object"));
        }

        let timing = PendingResponseTimestamps::new();
        let response = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseNew(&mut _, self.ptr))?;
        let timing = timing.finish(monotonic_nanos());
        crate::statistics::report_response(self, ptr::null_mut(), timing, Some(&error))?;

        // ResponseSend takes ownership of the response whatever it returns.
        triton_call!(triton_ng_sys::TRITONBACKEND_ResponseSend(
            response,
            triton_ng_sys::tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            error.as_ptr(),
        ))?;

        self.responded = true;
        self.timestamps.record_response(timing, false);
        Ok(())
    }

    pub(crate) fn release(&mut self, empty_success: bool) -> Result<(), TritonError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        self.report_statistics(empty_success)?;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestRelease(
            self.ptr,
            triton_ng_sys::tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL,
        ))?;
        self.ptr = ptr::null_mut();
        Ok(())
    }

    fn report_statistics(&mut self, empty_success: bool) -> Result<(), TritonError> {
        if self.statistics_reported {
            return Ok(());
        }
        let Some(instance) = &self.instance else {
            return Ok(());
        };
        let [exec_start, compute_start, compute_end, exec_end] =
            self.timestamps.statistics(monotonic_nanos());
        triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceReportStatistics(
            instance.as_ptr(),
            self.ptr,
            self.timestamps.success(empty_success),
            exec_start,
            compute_start,
            compute_end,
            exec_end,
        ))?;
        self.statistics_reported = true;
        Ok(())
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if !self.responded
            && !self.managed_by_factory
            && let Err(e) = self.respond_with(TritonError::internal(
                "backend did not produce a response for the request",
            ))
        {
            crate::log::error(&format!("failed to send error response: {e}"));
        }
        if let Err(e) = self.release(false) {
            crate::log::error(&format!("failed to release request: {e}"));
        }
    }
}

/// Trace borrowed from a [`Request`].
///
/// The request owns the trace. This view never deletes it and cannot outlive
/// the request. Neither `Send` nor `Sync`.
///
/// # Examples
///
/// ```ignore
/// if let Some(trace) = request.view().trace()? {
///     trace.report_activity("tokens.generated")?;
/// }
/// ```
pub struct RequestTrace<'a> {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceTrace,
    borrow: PhantomData<&'a Request>,
}

impl RequestTrace<'_> {
    /// Records a CUSTOM_ACTIVITY with the current monotonic timestamp.
    /// Activity names may contain any characters except NUL.
    ///
    /// # Errors
    ///
    /// Fails when `name` contains NUL or Triton rejects the activity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// trace.report_activity("decode.start")?;
    /// ```
    pub fn report_activity(&self, name: &str) -> Result<(), TritonError> {
        let name = cstring_from_str(name)?;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceTraceReportActivity(
            self.ptr,
            monotonic_nanos(),
            name.as_ptr(),
        ))
    }
}

/// Read-only borrow of a [`Request`].
///
/// The view never releases the request. Its lifetime prevents it from
/// outliving the request owner. Neither `Send` nor `Sync`.
///
/// # Examples
///
/// ```ignore
/// fn inspect(request: RequestView<'_>) -> Result<(), Error> {
///     for name in request.input_names()? {
///         let input = request.get_input(&name)?;
///         println!("{name}: {} bytes", input.properties()?.byte_size);
///     }
///     Ok(())
/// }
/// ```
pub struct RequestView<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Request,
    borrow: PhantomData<&'a Request>,
}

impl<'a> RequestView<'a> {
    pub(crate) unsafe fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Request) -> Self {
        Self {
            ptr,
            borrow: PhantomData,
        }
    }

    fn ptr(&self) -> Result<*mut triton_ng_sys::TRITONBACKEND_Request, TritonError> {
        if self.ptr.is_null() {
            Err(TritonError::from_message("the request was released"))
        } else {
            Ok(self.ptr)
        }
    }

    /// Lists the input tensor names.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let names = request.view().input_names()?;
    /// ```
    pub fn input_names(&self) -> Result<Vec<String>, TritonError> {
        let ptr = self.ptr()?;
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestInputCount(
            ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = ptr::null();
                triton_call!(triton_ng_sys::TRITONBACKEND_RequestInputName(
                    ptr, i, &mut name,
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    /// Lists the requested output tensor names.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let names = request.view().output_names()?;
    /// ```
    pub fn output_names(&self) -> Result<Vec<String>, TritonError> {
        let ptr = self.ptr()?;
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestOutputCount(
            ptr, &mut count
        ))?;
        (0..count)
            .map(|i| {
                let mut name: *const c_char = ptr::null();
                triton_call!(triton_ng_sys::TRITONBACKEND_RequestOutputName(
                    ptr, i, &mut name,
                ))?;
                Ok(unsafe { cstr_to_string(name) })
            })
            .collect()
    }

    /// Gets an input tensor borrowed from the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let input = request.view().get_input("INPUT")?;
    /// ```
    pub fn get_input(&self, name: &str) -> Result<Input<'a>, TritonError> {
        let ptr = self.ptr()?;
        let name = cstring_from_str(name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_RequestInput(
            ptr,
            name.as_ptr(),
            &mut _
        ))?;
        Ok(Input::from_ptr(ptr))
    }

    /// The value of the request parameter `key`, `None` when the request has
    /// no such parameter. Fails for a BYTES parameter, which is unsupported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let verbose = request.view().parameter("verbose")? == Some(Parameter::Bool(true));
    /// ```
    pub fn parameter(&self, key: &str) -> Result<Option<Parameter>, Error> {
        let ptr = self.ptr()?;
        let mut count = 0u32;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestParameterCount(
            ptr, &mut count
        ))?;

        for index in 0..count {
            let mut name: *const c_char = ptr::null();
            let mut kind: triton_ng_sys::TRITONSERVER_ParameterType = 0;
            let mut value: *const c_void = ptr::null();
            triton_call!(triton_ng_sys::TRITONBACKEND_RequestParameter(
                ptr, index, &mut name, &mut kind, &mut value
            ))?;

            if unsafe { cstr_to_string(name) } == key {
                // SAFETY: Triton returns `value` matching `kind`, owned by the request.
                return unsafe { Parameter::from_sys(kind, value) }
                    .map(Some)
                    .map_err(|e| format!("request parameter '{key}': {e}").into());
            }
        }

        Ok(None)
    }

    /// Returns the client-supplied request ID, or an empty string when no ID
    /// was supplied. The returned Rust string owns a copy. Fails when the
    /// request was released or Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let id = request.view().id()?;
    /// ```
    pub fn id(&self) -> Result<String, TritonError> {
        let ptr = self.ptr()?;
        let mut id = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestId(ptr, &mut id))?;
        Ok(unsafe { cstr_to_string(id) })
    }

    /// Returns the numeric correlation ID. Zero means no correlation ID.
    /// Fails when the request has a string correlation ID, was released, or
    /// Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let sequence = request.view().correlation_id()?;
    /// ```
    pub fn correlation_id(&self) -> Result<u64, TritonError> {
        let ptr = self.ptr()?;
        let mut id = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestCorrelationId(
            ptr, &mut id
        ))?;
        Ok(id)
    }

    /// Returns an owned copy of the string correlation ID. Fails when the
    /// request has a numeric correlation ID, including the default value of
    /// zero. Also fails when the request was released or Triton rejects the
    /// read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let sequence = request.view().correlation_id_string()?;
    /// ```
    pub fn correlation_id_string(&self) -> Result<String, TritonError> {
        let ptr = self.ptr()?;
        let mut id = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestCorrelationIdString(
            ptr, &mut id
        ))?;
        Ok(unsafe { cstr_to_string(id) })
    }

    /// Returns the sequence-start and sequence-end flags. Fails when the
    /// request was released or Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let flags = request.view().flags()?;
    /// ```
    pub fn flags(&self) -> Result<RequestFlags, TritonError> {
        let ptr = self.ptr()?;
        let mut flags = 0;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestFlags(ptr, &mut flags))?;
        Ok(RequestFlags::from_sys(flags))
    }

    /// Returns the request timeout in microseconds. Zero means no timeout.
    /// Fails when the request was released or Triton rejects the read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let timeout_us = request.view().timeout()?;
    /// ```
    pub fn timeout(&self) -> Result<u64, TritonError> {
        let ptr = self.ptr()?;
        let mut timeout = 0;
        triton_call!(
            triton_ng_sys::TRITONBACKEND_InferenceRequestTimeoutMicroseconds(ptr, &mut timeout)
        )?;
        Ok(timeout)
    }

    /// Returns the request trace, or `None` when this request is not sampled
    /// or the server was built without tracing support.
    ///
    /// # Errors
    ///
    /// Fails when the request was released or Triton returns an error other
    /// than UNSUPPORTED.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(trace) = request.view().trace()? {
    ///     trace.report_activity("input.ready")?;
    /// }
    /// ```
    pub fn trace(&self) -> Result<Option<RequestTrace<'a>>, TritonError> {
        let request = self.ptr()?;
        let mut trace = ptr::null_mut();
        let error = unsafe { triton_ng_sys::TRITONBACKEND_RequestTrace(request, &mut trace) };
        if !error.is_null() {
            let error = unsafe { TritonError::new(error) };
            if error.is_unsupported() {
                return Ok(None);
            }
            return Err(error);
        }
        Ok((!trace.is_null()).then_some(RequestTrace {
            ptr: trace,
            borrow: PhantomData,
        }))
    }

    /// Returns `true` if the client cancelled the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if request.view().is_cancelled()? {
    ///     return Ok(());
    /// }
    /// ```
    pub fn is_cancelled(&self) -> Result<bool, TritonError> {
        let ptr = self.ptr()?;
        let mut cancelled = false;
        triton_call!(triton_ng_sys::TRITONBACKEND_RequestIsCancelled(
            ptr,
            &mut cancelled
        ))?;
        Ok(cancelled)
    }
}
