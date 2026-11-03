use crate::error::{Error, TritonError};
use crate::memory_manager::MemoryManager;
use crate::message::Message;
use crate::model::SharedState;
use crate::types::InstanceKind;
use crate::utils::cstr_to_string;
use libc::{c_char, c_void};
use std::any::Any;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use triton_ng_macros::triton_call;

/// The backend, given to [`crate::Backend::initialize`] and
/// [`crate::Backend::finalize`].
///
/// Borrows the `TRITONBACKEND_Backend`: Triton owns it and keeps it alive
/// for the hook, so there is nothing to drop. Neither `Send` nor `Sync`.
///
/// # Examples
///
/// ```ignore
/// fn initialize(backend: &BackendHandle) -> Result<(), Error> {
///     log::info(&format!("loading backend {}", backend.name()?));
///     Ok(())
/// }
/// ```
pub struct BackendHandle {
    ptr: *mut triton_ng_sys::TRITONBACKEND_Backend,
}

impl BackendHandle {
    pub(crate) fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_Backend) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, TritonError> {
        let mut name: *const c_char = std::ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendName(
            self.ptr, &mut name
        ))?;
        Ok(unsafe { cstr_to_string(name) })
    }

    /// Returns the backend configuration as JSON. Command-line settings from
    /// `--backend-config` are under the `cmdline` key. The message is borrowed
    /// from Triton and only the serialized JSON is retained.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the query or cannot serialize the message.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config: serde_json::Value = serde_json::from_str(&backend.config()?)?;
    /// let setting = config["cmdline"]["setting"].as_str();
    /// ```
    pub fn config(&self) -> Result<String, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_BackendConfig(self.ptr, &mut _))?;
        Message::borrowed_json(ptr)
    }

    /// Returns the directory containing the backend shared library and its
    /// accompanying files. Fails when Triton rejects the query or does not
    /// provide the backend artifacts through the local file system.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let vocabulary = std::fs::read(backend.artifacts()?.join("vocab.txt"))?;
    /// ```
    pub fn artifacts(&self) -> Result<PathBuf, TritonError> {
        let mut artifact =
            triton_ng_sys::TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM;
        let mut location = ptr::null();
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendArtifacts(
            self.ptr,
            &mut artifact,
            &mut location,
        ))?;
        if artifact
            != triton_ng_sys::TRITONBACKEND_artifacttype_enum_TRITONBACKEND_ARTIFACT_FILESYSTEM
        {
            return Err(TritonError::from_message(
                "the backend artifacts are not on the local file system",
            ));
        }
        Ok(PathBuf::from(unsafe { cstr_to_string(location) }))
    }

    /// Borrows Triton's process-wide memory manager. It can allocate buffers
    /// during backend initialization, before any model is loaded.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the query or returns no memory manager.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let scratch = backend.memory_manager()?.ram(4096)?;
    /// ```
    pub fn memory_manager(&self) -> Result<MemoryManager, TritonError> {
        MemoryManager::of_backend(self.ptr)
    }

    /// Stores one state value shared by every model that uses this backend.
    /// Call it once from [`crate::Backend::initialize`]. A second call fails
    /// because replacing the value would invalidate references returned by
    /// [`BackendHandle::state`]. The value is dropped after
    /// [`crate::Backend::finalize`] returns. Triton state-access errors are
    /// also returned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// struct Shared {
    ///     setting: String,
    /// }
    ///
    /// fn initialize(backend: &BackendHandle) -> Result<(), Error> {
    ///     backend.set_state(Shared { setting: "value".to_string() })?;
    ///     Ok(())
    /// }
    /// ```
    pub fn set_state<T: Any + Send + Sync>(&self, state: T) -> Result<(), TritonError> {
        if !self.raw_state()?.is_null() {
            return Err(TritonError::from_message("backend state is already set"));
        }
        let boxed: Box<SharedState> = Box::new(Arc::new(state));
        let state_ptr = Box::into_raw(boxed) as *mut c_void;
        if let Err(error) = triton_call!(triton_ng_sys::TRITONBACKEND_BackendSetState(
            self.ptr, state_ptr
        )) {
            unsafe { drop(Box::from_raw(state_ptr as *mut SharedState)) };
            return Err(error);
        }
        Ok(())
    }

    /// Returns the state stored with [`BackendHandle::set_state`]. Fails when
    /// no state is stored, its type is not `T`, or Triton rejects the state
    /// read. The reference cannot outlive this backend handle.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let shared = backend.state::<Shared>()?;
    /// println!("{}", shared.setting);
    /// ```
    pub fn state<T: Any>(&self) -> Result<&T, Error> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Err("backend state is not initialized".into());
        }
        // SAFETY: set_state stores a Box<SharedState> at this pointer and the
        // box remains until clear_state after the finalize hook.
        let state = unsafe { &*(state_ptr as *const SharedState) };
        state
            .downcast_ref::<T>()
            .ok_or_else(|| "backend state has unexpected type".into())
    }

    pub(crate) fn clear_state(&self) -> Result<(), TritonError> {
        let state_ptr = self.raw_state()?;
        if state_ptr.is_null() {
            return Ok(());
        }
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendSetState(
            self.ptr,
            ptr::null_mut()
        ))?;
        unsafe { drop(Box::from_raw(state_ptr as *mut SharedState)) };
        Ok(())
    }

    fn raw_state(&self) -> Result<*mut c_void, TritonError> {
        let mut state_ptr = ptr::null_mut();
        triton_call!(triton_ng_sys::TRITONBACKEND_BackendState(
            self.ptr,
            &mut state_ptr
        ))?;
        Ok(state_ptr)
    }
}

/// Mutable backend attributes borrowed from one
/// `TRITONBACKEND_GetBackendAttribute` call. Triton owns the C object and
/// reuses it only for the duration of the callback; this value cannot be
/// stored or sent to another thread.
///
/// # Examples
///
/// ```ignore
/// fn attributes(
///     backend: &BackendHandle,
///     attributes: &mut BackendAttributes,
/// ) -> Result<(), Error> {
///     attributes.add_preferred_instance_group(InstanceKind::Cpu, 1, &[])?;
///     Ok(())
/// }
/// ```
pub struct BackendAttributes<'a> {
    ptr: *mut triton_ng_sys::TRITONBACKEND_BackendAttribute,
    borrow: PhantomData<&'a mut triton_ng_sys::TRITONBACKEND_BackendAttribute>,
}

impl BackendAttributes<'_> {
    pub(crate) unsafe fn from_ptr(ptr: *mut triton_ng_sys::TRITONBACKEND_BackendAttribute) -> Self {
        Self {
            ptr,
            borrow: PhantomData,
        }
    }

    /// Adds a preferred instance group. Calls are ordered from most to least
    /// preferred. Triton uses the first available group only when the model
    /// config omits `instance_group`. A count of zero and an empty device list
    /// leave those choices to Triton.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the instance group.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// attributes.add_preferred_instance_group(InstanceKind::Gpu, 2, &[0, 1])?;
    /// attributes.add_preferred_instance_group(InstanceKind::Cpu, 1, &[])?;
    /// ```
    pub fn add_preferred_instance_group(
        &mut self,
        kind: InstanceKind,
        count: u64,
        device_ids: &[u64],
    ) -> Result<(), TritonError> {
        let device_ptr = if device_ids.is_empty() {
            ptr::null()
        } else {
            device_ids.as_ptr()
        };
        triton_call!(
            triton_ng_sys::TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
                self.ptr,
                kind.to_sys(),
                count,
                device_ptr,
                device_ids.len() as u64,
            )
        )
    }

    /// Declares whether Triton may initialize multiple model instances of
    /// this backend concurrently. Every attributes callback must set this
    /// value again when enabling it.
    ///
    /// # Errors
    ///
    /// Fails when Triton rejects the attribute update.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// attributes.set_parallel_model_instance_loading(true)?;
    /// ```
    pub fn set_parallel_model_instance_loading(
        &mut self,
        enabled: bool,
    ) -> Result<(), TritonError> {
        triton_call!(
            triton_ng_sys::TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
                self.ptr, enabled,
            )
        )
    }
}
