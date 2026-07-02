// A loaded CUDA module (the result of `cuModuleLoadData` JIT-compiling a PTX image) and the
// functions looked up within it. See `context.rs` for the cross-resource drop-ordering note
// that applies equally here.

use std::ffi::c_void;

use crate::driver::{check, CudaDriver};
use crate::error::CudaError;
use crate::ffi::{CUfunction, CUmodule, CUstream};

pub struct CudaModule<'a> {
    driver: &'a CudaDriver,
    module: CUmodule,
}

impl<'a> CudaModule<'a> {
    pub(crate) fn new(driver: &'a CudaDriver, module: CUmodule) -> Self {
        CudaModule { driver, module }
    }

    /// Looks up an entry point by its `.visible .entry` name via `cuModuleGetFunction`.
    pub fn get_function(&self, name: &str) -> Result<CudaFunction<'a>, CudaError> {
        let cname = std::ffi::CString::new(name).map_err(|_| CudaError::DriverCallFailed {
            call: "cuModuleGetFunction",
            code: -1,
            message: "function name contains an interior NUL byte".to_string(),
        })?;

        let mut func = CUfunction(std::ptr::null_mut());
        // SAFETY: matches `cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char
        // *name)`; `self.module` came from a successful `cuModuleLoadData` and has not been
        // unloaded (it is only unloaded by this struct's own `Drop`, which cannot run before
        // this call returns since it needs `&self`); `cname` is NUL-terminated and kept alive
        // across the call.
        let rc = unsafe {
            (self.driver.fns().cu_module_get_function)(&mut func, self.module, cname.as_ptr())
        };
        check(self.driver.fns(), "cuModuleGetFunction", rc)?;

        Ok(CudaFunction {
            driver: self.driver,
            func,
        })
    }
}

impl<'a> Drop for CudaModule<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.module` was produced by a successful `cuModuleLoadData` and unloaded
        // at most once. See `context.rs`'s module-level note on cross-resource drop ordering:
        // any `CudaFunction` still referencing this module that outlives it will see its next
        // driver call fail with a handle-validation error, not a crash.
        unsafe {
            let _ = (self.driver.fns().cu_module_unload)(self.module);
        }
    }
}

/// A function within a loaded module. Carries no `Drop` of its own — `cuModuleGetFunction`
/// hands back a reference into its module, not a handle the driver expects to be
/// individually released.
pub struct CudaFunction<'a> {
    driver: &'a CudaDriver,
    func: CUfunction,
}

impl<'a> CudaFunction<'a> {
    /// Launches this function via `cuLaunchKernel` and blocks until it completes
    /// (`cuCtxSynchronize` runs immediately after the launch). A fire-and-forget async launch
    /// is not exposed: this crate's only consumers so far are correctness tests and the
    /// diff-vs-oracle harness, both of which need the result before they can check anything,
    /// so folding the sync into `launch` keeps every call site from having to remember it.
    ///
    /// `params` follows `cuLaunchKernel`'s own convention: one pointer per kernel argument,
    /// each pointing at that argument's value (e.g. a `&mut CUdeviceptr` cast to
    /// `*mut c_void` for a device-pointer argument).
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        let params_ptr = if params.is_empty() {
            std::ptr::null_mut()
        } else {
            params.as_mut_ptr()
        };

        // SAFETY: matches `cuLaunchKernel(CUfunction f, unsigned int gridDimX/Y/Z, unsigned
        // int blockDimX/Y/Z, unsigned int sharedMemBytes, CUstream hStream, void
        // **kernelParams, void **extra)`. `self.func` came from a successful
        // `cuModuleGetFunction` on a still-loaded module (borrowed for `'a`, so the module
        // outlives this call per the type's own lifetime). `params_ptr` is either null (valid
        // per the API when the kernel takes no arguments) or a pointer to a live, in-bounds
        // `&mut [*mut c_void]` whose element count the caller is responsible for matching to
        // the kernel's real parameter list — `cuLaunchKernel` has no way to check this itself,
        // same as any C variadic-by-convention argument-passing API. `extra` is null, meaning
        // "unused" per the driver's own documented convention. The default stream (`CUstream`
        // null) is used; no other in-flight work on this context is assumed.
        let rc = unsafe {
            (self.driver.fns().cu_launch_kernel)(
                self.func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                CUstream::NULL,
                params_ptr,
                std::ptr::null_mut(),
            )
        };
        check(self.driver.fns(), "cuLaunchKernel", rc)?;

        // SAFETY: `cuCtxSynchronize` takes no arguments; it blocks the calling host thread
        // until every queued operation on the current context (including the launch above)
        // completes.
        let rc = unsafe { (self.driver.fns().cu_ctx_synchronize)() };
        check(self.driver.fns(), "cuCtxSynchronize", rc)
    }
}
