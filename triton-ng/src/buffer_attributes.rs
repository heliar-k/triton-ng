//! Descriptions of memory occupied by models and model instances.

use crate::error::TritonError;
use crate::types::MemoryType;
use triton_ng_macros::triton_call;

/// Memory type and byte size of one model or instance allocation.
///
/// Owns the `TRITONSERVER_BufferAttributes` and deletes it on drop. Pass a
/// complete current set to [`crate::Model::report_memory_usage`] or
/// [`crate::ModelInstance::report_memory_usage`]. Neither `Send` nor `Sync`.
///
/// # Examples
///
/// ```ignore
/// let usage = [BufferAttributes::new(MemoryType::Cpu, 0, 4096)?];
/// model.report_memory_usage(&usage)?;
/// ```
pub struct BufferAttributes {
    pub(crate) ptr: *mut triton_ng_sys::TRITONSERVER_BufferAttributes,
}

impl BufferAttributes {
    /// Describes `byte_size` bytes on `memory_type` and `memory_type_id`.
    /// Every field is set explicitly because the C object does not initialize
    /// its byte size.
    ///
    /// # Errors
    ///
    /// Fails when `byte_size` does not fit the platform's `size_t`, or when
    /// Triton cannot create or populate the attributes object.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let weights = BufferAttributes::new(MemoryType::Gpu, 0, 64 << 20)?;
    /// ```
    pub fn new(
        memory_type: MemoryType,
        memory_type_id: i64,
        byte_size: u64,
    ) -> Result<Self, TritonError> {
        let attributes = Self {
            ptr: triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesNew(&mut _))?,
        };
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetMemoryType(
            attributes.ptr,
            memory_type.to_sys(),
        ))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetMemoryTypeId(
            attributes.ptr,
            memory_type_id,
        ))?;
        let byte_size = usize::try_from(byte_size)
            .map_err(|_| TritonError::from_message("memory usage byte size is too large"))?;
        triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesSetByteSize(
            attributes.ptr,
            byte_size,
        ))?;
        Ok(attributes)
    }
}

impl Drop for BufferAttributes {
    fn drop(&mut self) {
        if let Err(error) =
            triton_call!(triton_ng_sys::TRITONSERVER_BufferAttributesDelete(self.ptr))
        {
            crate::log::error(&format!("failed to delete buffer attributes: {error}"));
        }
    }
}
