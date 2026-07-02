// A CUDA context and the device memory allocated within it.
//
// Lifetime discipline: `CudaContext<'a>`, `CudaModule<'a>`, `CudaFunction<'a>`, and
// `DeviceBuffer<'a>` all borrow the *driver* for the same `'a`, not each other — matching
// `cuCtxCreate`'s own C-level design, where a context, its modules, and its allocations are
// siblings under one driver, not a nested-ownership tree Rust's borrow checker can see through.
// That means the type system alone does not stop a caller from dropping a `CudaContext` while
// a `CudaModule`/`DeviceBuffer` derived from it is still alive; the discipline that keeps this
// sound is:
//   1. Straight-line Rust already gets this right by default: locals drop in reverse
//      declaration order, so `let ctx = ...; let m = ctx.load_module(...)?;` frees `m` before
//      `ctx` at scope exit with no extra effort.
//   2. If a caller *does* destroy a context early (explicit `drop(ctx)`), every handle the
//      driver hands out (`CUmodule`, `CUdeviceptr`, `CUfunction`) is an opaque ID the driver
//      validates against its own live-context table on every call — not a raw pointer Basalt
//      dereferences. Using a stale handle after its context is destroyed comes back as a
//      driver error code (`check` below turns it into a `CudaError`), not memory corruption.
// A stronger design (tying `CudaModule`'s lifetime to a borrow of `CudaContext` itself) was
// considered and rejected here: it would force `load_module`/`alloc` to take `&'a self`
// instead of `&self`, which forbids calling either more than once through the same binding —
// too restrictive for what is otherwise a thin, repeatedly-callable wrapper.

use std::ffi::c_void;

use crate::driver::{check, CudaDriver};
use crate::error::CudaError;
use crate::ffi::{CUcontext, CUdeviceptr, CUmodule};
use crate::module::CudaModule;

pub struct CudaContext<'a> {
    driver: &'a CudaDriver,
    ctx: CUcontext,
}

impl<'a> CudaContext<'a> {
    pub(crate) fn new(driver: &'a CudaDriver, ctx: CUcontext) -> Self {
        CudaContext { driver, ctx }
    }

    /// JIT-compiles and loads a PTX (or cubin) image via `cuModuleLoadData`. `ptx_text` must
    /// not contain an interior NUL byte; the driver's own image parser expects a
    /// NUL-terminated buffer, same as any C string.
    pub fn load_module(&self, ptx_text: &str) -> Result<CudaModule<'a>, CudaError> {
        let image = std::ffi::CString::new(ptx_text).map_err(|_| CudaError::DriverCallFailed {
            call: "cuModuleLoadData",
            code: -1,
            message: "PTX text contains an interior NUL byte".to_string(),
        })?;

        let mut module = CUmodule(std::ptr::null_mut());
        // SAFETY: matches `cuModuleLoadData(CUmodule *module, const void *image)`; `image`
        // is a NUL-terminated `CString` buffer kept alive across the call, which is exactly
        // the format `cuModuleLoadData` expects for a PTX text image (the driver's JIT reads
        // until the terminating NUL).
        let rc = unsafe {
            (self.driver.fns().cu_module_load_data)(&mut module, image.as_ptr().cast::<c_void>())
        };
        check(self.driver.fns(), "cuModuleLoadData", rc)?;

        Ok(CudaModule::new(self.driver, module))
    }

    /// Allocates `bytes` of device memory via `cuMemAlloc`.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer<'a>, CudaError> {
        let mut dptr: CUdeviceptr = 0;
        // SAFETY: matches `cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)`; `dptr` is a
        // valid, writable location for the duration of the call.
        let rc = unsafe { (self.driver.fns().cu_mem_alloc)(&mut dptr, bytes) };
        check(self.driver.fns(), "cuMemAlloc", rc)?;

        Ok(DeviceBuffer {
            driver: self.driver,
            dptr,
            len: bytes,
        })
    }
}

impl<'a> Drop for CudaContext<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was produced by a successful `cuCtxCreate` in
        // `CudaDriver::create_context` and this `Drop` runs at most once per context (Rust
        // drops a value exactly once). The return code is intentionally discarded — `Drop`
        // cannot propagate a `Result`, and per the module-level note above, a failure here
        // (e.g. because a derived handle was already destroyed some other way) cannot corrupt
        // memory, only leak or no-op.
        unsafe {
            let _ = (self.driver.fns().cu_ctx_destroy)(self.ctx);
        }
    }
}

/// A device memory allocation. `copy_from_host`/`copy_to_host` bounds-check against the byte
/// length `cuMemAlloc` was asked for.
pub struct DeviceBuffer<'a> {
    driver: &'a CudaDriver,
    dptr: CUdeviceptr,
    len: usize,
}

impl<'a> DeviceBuffer<'a> {
    /// Copies `src` to the device, starting at this buffer's base address. `src` must fit
    /// within the buffer's allocated length.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<(), CudaError> {
        if src.len() > self.len {
            return Err(CudaError::DriverCallFailed {
                call: "cuMemcpyHtoD",
                code: -1,
                message: format!(
                    "host slice of {} bytes exceeds device buffer of {} bytes",
                    src.len(),
                    self.len
                ),
            });
        }
        // SAFETY: matches `cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t
        // ByteCount)`; `src.as_ptr()` is valid for `src.len()` bytes (guaranteed by the
        // slice), and `src.len() <= self.len`, the size `self.dptr` was allocated with, so
        // the driver's device-side write stays in bounds too.
        let rc = unsafe {
            (self.driver.fns().cu_memcpy_htod)(self.dptr, src.as_ptr().cast::<c_void>(), src.len())
        };
        check(self.driver.fns(), "cuMemcpyHtoD", rc)
    }

    /// Copies from the device, starting at this buffer's base address, into `dst`. `dst` must
    /// fit within the buffer's allocated length.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        if dst.len() > self.len {
            return Err(CudaError::DriverCallFailed {
                call: "cuMemcpyDtoH",
                code: -1,
                message: format!(
                    "host slice of {} bytes exceeds device buffer of {} bytes",
                    dst.len(),
                    self.len
                ),
            });
        }
        // SAFETY: matches `cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t ByteCount)`;
        // `dst.as_mut_ptr()` is valid for `dst.len()` writable bytes (guaranteed by the
        // slice), and `dst.len() <= self.len`, so the read from `self.dptr` stays within the
        // range `cuMemAlloc` returned.
        let rc = unsafe {
            (self.driver.fns().cu_memcpy_dtoh)(
                dst.as_mut_ptr().cast::<c_void>(),
                self.dptr,
                dst.len(),
            )
        };
        check(self.driver.fns(), "cuMemcpyDtoH", rc)
    }

    /// The raw `CUdeviceptr` value, for building a `cuLaunchKernel` parameter array.
    pub fn device_ptr(&self) -> u64 {
        self.dptr
    }
}

impl<'a> Drop for DeviceBuffer<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.dptr` was produced by a successful `cuMemAlloc` above and freed at
        // most once. See the module-level note on cross-resource drop ordering.
        unsafe {
            let _ = (self.driver.fns().cu_mem_free)(self.dptr);
        }
    }
}
