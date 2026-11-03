use crate::error::TritonError;
use crate::request::Request;
use crate::timing::ResponseTimestamps;
use std::ptr;
use triton_ng_macros::triton_call;

pub(crate) fn report_response(
    request: &Request,
    factory: *mut triton_ng_sys::TRITONBACKEND_ResponseFactory,
    timing: ResponseTimestamps,
    error: Option<&TritonError>,
) -> Result<(), TritonError> {
    let Some(instance) = &request.instance else {
        return Ok(());
    };
    let owned_factory = if factory.is_null() {
        Some(ResponseFactoryHandle::new(request)?)
    } else {
        None
    };
    let factory = owned_factory
        .as_ref()
        .map_or(factory, |factory| factory.ptr);

    let Some(statistics) = ResponseStatistics::new()? else {
        return Ok(());
    };
    triton_call!(
        triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetModelInstance(
            statistics.ptr,
            instance.as_ptr(),
        )
    )?;
    triton_call!(
        triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseFactory(
            statistics.ptr,
            factory,
        )
    )?;
    triton_call!(
        triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseStart(
            statistics.ptr,
            timing.response_start,
        )
    )?;
    if let Some(compute_output_start) = timing.compute_output_start {
        triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetComputeOutputStart(
                statistics.ptr,
                compute_output_start,
            )
        )?;
    }
    triton_call!(
        triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetResponseEnd(
            statistics.ptr,
            timing.response_end,
        )
    )?;
    if let Some(error) = error {
        triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsSetError(
                statistics.ptr,
                error.as_ptr(),
            )
        )?;
    }
    triton_call!(triton_ng_sys::TRITONBACKEND_ModelInstanceReportResponseStatistics(statistics.ptr))
}

struct ResponseFactoryHandle {
    ptr: *mut triton_ng_sys::TRITONBACKEND_ResponseFactory,
}

impl ResponseFactoryHandle {
    fn new(request: &Request) -> Result<Self, TritonError> {
        let ptr = triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryNew(
            &mut _,
            request.as_ptr()
        ))?;
        Ok(Self { ptr })
    }
}

impl Drop for ResponseFactoryHandle {
    fn drop(&mut self) {
        if let Err(error) =
            triton_call!(triton_ng_sys::TRITONBACKEND_ResponseFactoryDelete(self.ptr))
        {
            crate::log::error(&format!(
                "failed to delete statistics response factory: {error}"
            ));
        }
    }
}

struct ResponseStatistics {
    ptr: *mut triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatistics,
}

impl ResponseStatistics {
    fn new() -> Result<Option<Self>, TritonError> {
        let mut ptr = ptr::null_mut();
        let error =
            unsafe { triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsNew(&mut ptr) };
        if !error.is_null() {
            return Err(unsafe { TritonError::new(error) });
        }
        // 编译时关闭 TRITON_ENABLE_STATS 的服务器会返回成功，但不创建对象。
        Ok((!ptr.is_null()).then_some(Self { ptr }))
    }
}

impl Drop for ResponseStatistics {
    fn drop(&mut self) {
        if let Err(error) = triton_call!(
            triton_ng_sys::TRITONBACKEND_ModelInstanceResponseStatisticsDelete(self.ptr)
        ) {
            crate::log::error(&format!("failed to delete response statistics: {error}"));
        }
    }
}
