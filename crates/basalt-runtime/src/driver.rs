// The loaded CUDA driver: a `dlopen` handle plus every Driver API entry point this crate
// needs, resolved once via `dlsym` into a plain function-pointer table. Every other type in
// this crate (`CudaContext`, `CudaModule`, `CudaFunction`, `DeviceBuffer`) borrows a
// `CudaDriver` for its whole lifetime, so none of them can outlive the library they call
// through.

use std::ffi::c_char;

use crate::context::CudaContext;
use crate::dl::Library;
use crate::error::CudaError;
use crate::ffi::{
    CUcontext, CUdevice, FnCuCtxCreate, FnCuCtxDestroy, FnCuCtxSynchronize, FnCuDeviceGet,
    FnCuDeviceGetCount, FnCuGetErrorString, FnCuInit, FnCuLaunchKernel, FnCuMemAlloc, FnCuMemFree,
    FnCuMemcpyDtoH, FnCuMemcpyHtoD, FnCuModuleGetFunction, FnCuModuleLoadData, FnCuModuleUnload,
    CUDA_SUCCESS,
};

pub(crate) struct FnTable {
    pub cu_device_get_count: FnCuDeviceGetCount,
    pub cu_device_get: FnCuDeviceGet,
    pub cu_ctx_create: FnCuCtxCreate,
    pub cu_ctx_destroy: FnCuCtxDestroy,
    pub cu_ctx_synchronize: FnCuCtxSynchronize,
    pub cu_module_load_data: FnCuModuleLoadData,
    pub cu_module_unload: FnCuModuleUnload,
    pub cu_module_get_function: FnCuModuleGetFunction,
    pub cu_launch_kernel: FnCuLaunchKernel,
    pub cu_mem_alloc: FnCuMemAlloc,
    pub cu_mem_free: FnCuMemFree,
    pub cu_memcpy_htod: FnCuMemcpyHtoD,
    pub cu_memcpy_dtoh: FnCuMemcpyDtoH,
    pub cu_get_error_string: FnCuGetErrorString,
}

/// Resolves `symbol` under each candidate in `names` (first hit wins) and reinterprets the
/// address as `F`. This is how the driver's `_v2`/`_v3`-suffixed entry points are handled:
/// CUDA has historically frozen an entry point's C-level name at a versioned symbol
/// (`cuCtxCreate_v2`, `cuMemAlloc_v2`, …) when the ABI changed, rather than renaming the
/// "logical" API — modern drivers may only export the versioned symbol, so every caller here
/// tries the versioned name first and falls back to the bare name for older/unusual builds.
///
/// SAFETY CONTRACT: the caller must instantiate `F` as the exact `extern "C" fn` type
/// matching the real C signature documented for every name in `names` — argument types,
/// order, and return type. All call sites in this module pass the `Fn*` type aliases from
/// `ffi.rs`, which are transcribed from the Driver API's published, stable ABI.
fn resolve<F>(lib: &Library, names: &[&'static str]) -> Result<F, CudaError> {
    for name in names {
        if let Some(ptr) = lib.symbol(name) {
            // SAFETY: `ptr` is a non-null address `dlsym` resolved for `name`, one of the
            // driver's real exported symbols; per this function's documented contract, `F`
            // is a function-pointer type whose size equals a pointer's (true for every
            // `extern "C" fn` type) and whose signature the caller has matched to `name`'s
            // real C prototype. `transmute_copy` reads `size_of::<F>()` bytes from `&ptr`,
            // which is exactly `size_of::<*mut c_void>()` for any function pointer.
            let f: F = unsafe { std::mem::transmute_copy(&ptr) };
            return Ok(f);
        }
    }
    Err(CudaError::SymbolNotFound(names[0]))
}

/// Turns a non-success `CUresult` into a `CudaError::DriverCallFailed`, using
/// `cuGetErrorString` to attach a human-readable message.
pub(crate) fn check(fns: &FnTable, call: &'static str, code: i32) -> Result<(), CudaError> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    let mut msg_ptr: *const c_char = std::ptr::null();
    // SAFETY: `cu_get_error_string` was resolved against `cuGetErrorString`, whose documented
    // signature is `CUresult cuGetErrorString(CUresult error, const char **pStr)`: it writes
    // a pointer to a static, driver-owned string into `*pStr` and returns `CUDA_SUCCESS` on a
    // recognized code. `msg_ptr` is a valid, aligned, writable location for the call.
    let rc = unsafe { (fns.cu_get_error_string)(code, &mut msg_ptr) };
    let message = if rc == CUDA_SUCCESS && !msg_ptr.is_null() {
        // SAFETY: `msg_ptr` was just confirmed non-null; per `cuGetErrorString`'s contract it
        // points at a NUL-terminated, driver-owned static string (no ownership transfer, no
        // lifetime concern beyond this call).
        unsafe { std::ffi::CStr::from_ptr(msg_ptr) }
            .to_string_lossy()
            .into_owned()
    } else {
        format!("unknown CUDA error {code}")
    };
    Err(CudaError::DriverCallFailed {
        call,
        code,
        message,
    })
}

/// A loaded CUDA driver: an open `libcuda.so*` handle plus its resolved entry points.
/// `cuInit` has already run by the time `load()` returns successfully.
pub struct CudaDriver {
    _lib: Library,
    fns: FnTable,
}

impl CudaDriver {
    /// `dlopen`s the CUDA driver (`libcuda.so.1`, the real installed SONAME, falling back to
    /// `libcuda.so`, which typically only exists via a `-dev`/SDK symlink), resolves every
    /// entry point this crate uses, then calls `cuInit(0)` — the Driver API requires `cuInit`
    /// to have succeeded before any other driver call in the process.
    pub fn load() -> Result<CudaDriver, CudaError> {
        let lib = Library::open_first(&["libcuda.so.1", "libcuda.so"])
            .map_err(CudaError::DriverNotFound)?;

        let cu_init: FnCuInit = resolve(&lib, &["cuInit"])?;
        let fns = FnTable {
            cu_device_get_count: resolve(&lib, &["cuDeviceGetCount"])?,
            cu_device_get: resolve(&lib, &["cuDeviceGet"])?,
            cu_ctx_create: resolve(&lib, &["cuCtxCreate_v2", "cuCtxCreate"])?,
            cu_ctx_destroy: resolve(&lib, &["cuCtxDestroy_v2", "cuCtxDestroy"])?,
            cu_ctx_synchronize: resolve(&lib, &["cuCtxSynchronize"])?,
            cu_module_load_data: resolve(&lib, &["cuModuleLoadData"])?,
            cu_module_unload: resolve(&lib, &["cuModuleUnload"])?,
            cu_module_get_function: resolve(&lib, &["cuModuleGetFunction"])?,
            cu_launch_kernel: resolve(&lib, &["cuLaunchKernel"])?,
            cu_mem_alloc: resolve(&lib, &["cuMemAlloc_v2", "cuMemAlloc"])?,
            cu_mem_free: resolve(&lib, &["cuMemFree_v2", "cuMemFree"])?,
            cu_memcpy_htod: resolve(&lib, &["cuMemcpyHtoD_v2", "cuMemcpyHtoD"])?,
            cu_memcpy_dtoh: resolve(&lib, &["cuMemcpyDtoH_v2", "cuMemcpyDtoH"])?,
            cu_get_error_string: resolve(&lib, &["cuGetErrorString", "cuGetErrorName"])?,
        };

        // SAFETY: `cu_init` was resolved against `cuInit`, documented as
        // `CUresult cuInit(unsigned int Flags)`; the Driver API defines no flag values other
        // than 0 today, so 0 is the only correct argument.
        let rc = unsafe { cu_init(0) };
        check(&fns, "cuInit", rc)?;

        Ok(CudaDriver { _lib: lib, fns })
    }

    /// Number of CUDA-capable devices visible to the driver.
    pub fn device_count(&self) -> Result<i32, CudaError> {
        let mut count: i32 = 0;
        // SAFETY: matches `cuDeviceGetCount(int *count)`; `count` is a valid, writable `i32`
        // on the stack for the duration of the call.
        let rc = unsafe { (self.fns.cu_device_get_count)(&mut count) };
        check(&self.fns, "cuDeviceGetCount", rc)?;
        Ok(count)
    }

    /// Resolves device ordinal `device_index` and creates a primary context on it.
    pub fn create_context(&self, device_index: i32) -> Result<CudaContext<'_>, CudaError> {
        let mut device: CUdevice = 0;
        // SAFETY: matches `cuDeviceGet(CUdevice *device, int ordinal)`.
        let rc = unsafe { (self.fns.cu_device_get)(&mut device, device_index) };
        check(&self.fns, "cuDeviceGet", rc)?;

        let mut ctx = CUcontext(std::ptr::null_mut());
        // SAFETY: matches `cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)`;
        // `flags = 0` requests the driver's default context behavior.
        let rc = unsafe { (self.fns.cu_ctx_create)(&mut ctx, 0, device) };
        check(&self.fns, "cuCtxCreate", rc)?;

        Ok(CudaContext::new(self, ctx))
    }

    pub(crate) fn fns(&self) -> &FnTable {
        &self.fns
    }
}
