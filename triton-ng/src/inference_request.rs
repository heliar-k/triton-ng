use crate::TritonError;
use crate::server::Server;
use crate::types::{DataType, Parameter, RequestFlags};
use crate::utils::cstring_from_str;
use crate::{InputBuffer, OutputBuffer};
use std::collections::HashMap;
use triton_ng_macros::triton_call;

/// Buffers referenced by a submitted request, grouped by input name.
pub(crate) type InputBuffers = HashMap<String, Vec<Box<dyn Send>>>;

/// Buffers that requested outputs are written into, by output name. `None`
/// once the buffer is handed to Triton.
pub(crate) type OutputBuffers = HashMap<String, Option<OutputBuffer>>;

fn validate_streaming_outputs(outputs: &OutputBuffers) -> Result<(), String> {
    if outputs.values().any(Option::is_some) {
        Err("preallocated output buffers are not supported by infer_stream".to_string())
    } else {
        Ok(())
    }
}

/// A BLS request to another model, submitted with [`crate::Server::infer_async`].
///
/// Owns the `TRITONSERVER_InferenceRequest` and its input buffers. Dropped
/// before it is submitted, it deletes the request; once submitted, Triton
/// owns it and the release callback deletes it. `Send`, not `Sync`.
///
/// # Examples
///
/// ```ignore
/// let mut inference = InferenceRequest::new(ctx.server(), "encoder", -1)?;
/// inference.add_input_with_data("AUDIO", DataType::Fp32, &[1, samples], bytes)?;
/// inference.add_requested_output("EMBEDDING")?;
/// let result = ctx.server().infer_async(inference).await?;
/// ```
pub struct InferenceRequest {
    ptr: *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
    /// Input buffers referenced by the request. Triton does not copy input
    /// data, so the buffers must outlive the request: they are handed over to
    /// the release callback together with the request (see `Server::infer_async`).
    buffers: InputBuffers,
    outputs: OutputBuffers,
}

// SAFETY: Triton lets a request be built and submitted from any thread, it is
// only not safe to use one request from several threads at once, and
// `InferenceRequest` is not `Sync`. The buffers are `Send`.
unsafe impl Send for InferenceRequest {}

impl InferenceRequest {
    pub fn new(server: &Server, model_name: &str, model_version: i64) -> Result<Self, TritonError> {
        let model_name_cstr = cstring_from_str(model_name)?;
        let ptr = triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestNew(
            &mut _,
            server.as_ptr(),
            model_name_cstr.as_ptr(),
            model_version,
        ))?;
        Ok(Self {
            ptr,
            buffers: HashMap::new(),
            outputs: HashMap::new(),
        })
    }

    /// Adds a bool, integer, floating-point or string parameter to the BLS
    /// request. Triton appends parameters: setting the same key twice produces
    /// two entries, and this API does not remove an earlier value.
    ///
    /// # Errors
    ///
    /// Fails when the key or string value contains NUL, or Triton rejects the
    /// parameter.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_parameter("streaming", true)?;
    /// inference.set_parameter("max_tokens", 128)?;
    /// inference.set_parameter("temperature", 0.7)?;
    /// inference.set_parameter("request_kind", "generate")?;
    /// ```
    pub fn set_parameter(
        &mut self,
        key: &str,
        value: impl Into<Parameter>,
    ) -> Result<(), TritonError> {
        let key = cstring_from_str(key)?;
        match value.into() {
            Parameter::Bool(value) => {
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetBoolParameter(
                        self.ptr,
                        key.as_ptr(),
                        value
                    )
                )
            }
            Parameter::Int(value) => {
                triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestSetIntParameter(
                    self.ptr,
                    key.as_ptr(),
                    value
                ))
            }
            Parameter::Double(value) => {
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetDoubleParameter(
                        self.ptr,
                        key.as_ptr(),
                        value
                    )
                )
            }
            Parameter::String(value) => {
                let value = cstring_from_str(&value)?;
                triton_call!(
                    triton_ng_sys::TRITONSERVER_InferenceRequestSetStringParameter(
                        self.ptr,
                        key.as_ptr(),
                        value.as_ptr()
                    )
                )
            }
        }
    }

    /// Sets the numeric correlation ID. Zero clears the correlation ID.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the update.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_correlation_id(42)?;
    /// ```
    pub fn set_correlation_id(&mut self, correlation_id: u64) -> Result<(), TritonError> {
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetCorrelationId(self.ptr, correlation_id)
        )
    }

    /// Sets the string correlation ID, replacing any numeric ID. Triton
    /// rejects strings longer than 128 bytes.
    ///
    /// # Errors
    ///
    /// Fails when the string contains NUL or exceeds 128 bytes, or Triton
    /// rejects the update.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_correlation_id_string("conversation-42")?;
    /// ```
    pub fn set_correlation_id_string(&mut self, correlation_id: &str) -> Result<(), TritonError> {
        let correlation_id = cstring_from_str(correlation_id)?;
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetCorrelationIdString(
                self.ptr,
                correlation_id.as_ptr()
            )
        )
    }

    /// Replaces the sequence-start and sequence-end flags.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the update.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_flags(RequestFlags {
    ///     sequence_start: true,
    ///     sequence_end: false,
    /// })?;
    /// ```
    pub fn set_flags(&mut self, flags: RequestFlags) -> Result<(), TritonError> {
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestSetFlags(
            self.ptr,
            flags.to_sys()
        ))
    }

    /// Sets the request timeout in microseconds. Zero means no timeout.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the update.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_timeout(500_000)?;
    /// ```
    pub fn set_timeout(&mut self, timeout_us: u64) -> Result<(), TritonError> {
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
                self.ptr, timeout_us
            )
        )
    }

    /// Sets the scheduling priority. Triton replaces zero or a value above
    /// the model's maximum priority with the model's default priority.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the update.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.set_priority(1)?;
    /// ```
    pub fn set_priority(&mut self, priority: u64) -> Result<(), TritonError> {
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestSetPriorityUInt64(self.ptr, priority)
        )
    }

    pub(crate) fn ensure_streaming_compatible(&self) -> Result<(), TritonError> {
        validate_streaming_outputs(&self.outputs).map_err(TritonError::from_message)
    }

    /// Splits the request into the raw pointer, the input buffers it
    /// references and the buffers of its outputs, without running `Drop`.
    pub(crate) fn into_raw_parts(
        self,
    ) -> (
        *mut triton_ng_sys::TRITONSERVER_InferenceRequest,
        InputBuffers,
        OutputBuffers,
    ) {
        let mut this = std::mem::ManuallyDrop::new(self);
        (
            this.ptr,
            std::mem::take(&mut this.buffers),
            std::mem::take(&mut this.outputs),
        )
    }

    pub fn add_input(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestAddInput(
            self.ptr,
            name_cstr.as_ptr(),
            datatype.to_sys(),
            shape.as_ptr(),
            shape.len() as u64,
        ))
    }

    /// Appends a copy of `data` to the input `name`.
    pub fn append_input_data(&mut self, name: &str, data: &[u8]) -> Result<(), TritonError> {
        self.append_input_data_owned(name, data.to_vec())
    }

    /// Appends `data` to the input `name`, the request takes ownership of the buffer.
    pub fn append_input_data_owned(
        &mut self,
        name: &str,
        data: Vec<u8>,
    ) -> Result<(), TritonError> {
        self.append_input_buffer(name, data)
    }

    /// Appends a copy of `data` for model instances using `host_policy_name`.
    /// Also add a buffer with [`InferenceRequest::append_input_data`]: Triton
    /// uses that ordinary buffer on devices that have no buffer for their
    /// host policy.
    ///
    /// # Errors
    ///
    /// Fails when either name contains NUL, the input does not exist, or
    /// Triton rejects the buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.append_input_data("INPUT", &fallback)?;
    /// inference.append_input_data_with_host_policy("INPUT", &numa_data, "numa-0")?;
    /// ```
    pub fn append_input_data_with_host_policy(
        &mut self,
        name: &str,
        data: &[u8],
        host_policy_name: &str,
    ) -> Result<(), TritonError> {
        self.append_input_buffer_with_policy(name, data.to_vec(), Some(host_policy_name))
    }

    /// Appends `buffer` to the input `name` without copying it; Triton reads
    /// it from the memory it is in, see [`InputBuffer`]. The request owns the
    /// buffer until Triton releases the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.add_input("hi", DataType::Fp32, &[1, batch, 320])?;
    /// inference.append_input_buffer("hi", Arc::clone(&state))?;
    /// ```
    pub fn append_input_buffer(
        &mut self,
        name: &str,
        buffer: impl InputBuffer,
    ) -> Result<(), TritonError> {
        self.append_input_buffer_with_policy(name, buffer, None)
    }

    fn append_input_buffer_with_policy(
        &mut self,
        name: &str,
        buffer: impl InputBuffer,
        host_policy_name: Option<&str>,
    ) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;
        let host_policy = host_policy_name.map(cstring_from_str).transpose()?;
        let raw = buffer.raw();

        // The memory of every input buffer is on the heap or in Triton's
        // pools, so the pointer stays valid after the buffer is moved into
        // `self.buffers`.
        if let Some(host_policy) = host_policy {
            triton_call!(
                triton_ng_sys::TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(
                    self.ptr,
                    name_cstr.as_ptr(),
                    raw.ptr.cast_const(),
                    raw.len,
                    raw.memory_type,
                    raw.memory_type_id,
                    host_policy.as_ptr(),
                )
            )?;
        } else {
            triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestAppendInputData(
                self.ptr,
                name_cstr.as_ptr(),
                raw.ptr.cast_const(),
                raw.len,
                raw.memory_type,
                raw.memory_type_id,
            ))?;
        }

        self.buffers
            .entry(name.to_string())
            .or_default()
            .push(Box::new(buffer));
        Ok(())
    }

    /// Removes an input and drops only the Rust buffers associated with that
    /// input. Fails when the name contains NUL, the input does not exist, or
    /// Triton rejects the removal. Rust buffers are dropped only after the C
    /// removal succeeds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.remove_input("UNUSED")?;
    /// ```
    pub fn remove_input(&mut self, name: &str) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;
        triton_call!(triton_ng_sys::TRITONSERVER_InferenceRequestRemoveInput(
            self.ptr,
            name_cstr.as_ptr()
        ))?;
        self.buffers.remove(name);
        Ok(())
    }

    /// Adds an input tensor together with its data in one call.
    pub fn add_input_with_data(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
        data: Vec<u8>,
    ) -> Result<(), TritonError> {
        self.add_input_with_buffer(name, datatype, shape, data)
    }

    /// Adds an input tensor whose data is `buffer`, see
    /// [`InferenceRequest::append_input_buffer`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tokens = ctx.memory().pinned(batch * 8)?;
    /// tokens.copy_from_slice(&token_bytes);
    /// inference.add_input_with_buffer("x", DataType::Int64, &[batch, 1], tokens)?;
    /// ```
    pub fn add_input_with_buffer(
        &mut self,
        name: &str,
        datatype: DataType,
        shape: &[i64],
        buffer: impl InputBuffer,
    ) -> Result<(), TritonError> {
        self.add_input(name, datatype, shape)?;
        self.append_input_buffer(name, buffer)
    }

    /// Requests the output `name`, copied to CPU memory that the result owns.
    pub fn add_requested_output(&mut self, name: &str) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;

        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestAddRequestedOutput(
                self.ptr,
                name_cstr.as_ptr(),
            )
        )
    }

    /// Removes a requested output and any preallocated buffer registered for
    /// it. Removing an output that was not requested is allowed. Fails when
    /// the name contains NUL or Triton rejects the removal.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// inference.remove_requested_output("UNUSED")?;
    /// ```
    pub fn remove_requested_output(&mut self, name: &str) -> Result<(), TritonError> {
        let name_cstr = cstring_from_str(name)?;
        triton_call!(
            triton_ng_sys::TRITONSERVER_InferenceRequestRemoveRequestedOutput(
                self.ptr,
                name_cstr.as_ptr()
            )
        )?;
        self.outputs.remove(name);
        Ok(())
    }

    /// Requests the output `name` written into `buffer`. The result hands the
    /// buffer back with [`crate::InferenceResult::take_gpu`] (or `take_pinned`,
    /// `take_ram`), so the memory can be cached and reused.
    ///
    /// The inference fails when the output size differs from the length of
    /// the buffer, and when the model returns the output more than once
    /// (decoupled models): the buffer receives one output. Fails right away
    /// when `name` is requested twice.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let state = ctx.memory().gpu(ctx.device_id(), batch * 320 * 4)?;
    /// inference.add_requested_output_into("ho", state)?;
    /// let mut result = ctx.server().infer_async(inference).await?;
    /// let state: GpuBuffer = result.take_gpu("ho")?;
    /// ```
    pub fn add_requested_output_into(
        &mut self,
        name: &str,
        buffer: impl Into<OutputBuffer>,
    ) -> Result<(), TritonError> {
        if self.outputs.contains_key(name) {
            return Err(TritonError::from_message(format!(
                "output '{name}' already has a buffer"
            )));
        }
        self.add_requested_output(name)?;
        self.outputs.insert(name.to_string(), Some(buffer.into()));
        Ok(())
    }
}

impl Drop for InferenceRequest {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                triton_ng_sys::TRITONSERVER_InferenceRequestDelete(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OutputBuffers, validate_streaming_outputs};
    use crate::MemoryManager;

    #[test]
    fn streaming_rejects_preallocated_output_buffers() {
        let mut outputs = OutputBuffers::new();
        assert!(validate_streaming_outputs(&outputs).is_ok());
        outputs.insert("used".to_string(), None);
        assert!(validate_streaming_outputs(&outputs).is_ok());
        outputs.insert(
            "preallocated".to_string(),
            Some(MemoryManager::heap().ram(8).unwrap().into()),
        );
        assert!(validate_streaming_outputs(&outputs).is_err());
    }
}
