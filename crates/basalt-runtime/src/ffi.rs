// Raw CUDA Driver API shapes: opaque handle types and function-pointer signatures matching
// the driver's stable, published C ABI (as documented by NVIDIA's Driver API reference).
// Nothing here is generated from or checked against real CUDA headers — there are none in
// this build — so every signature is transcribed by hand from the documented ABI and must
// stay in lockstep with it.

use std::ffi::{c_char, c_void};

/// `CUresult`: a C enum, always a 4-byte signed int at the ABI level. `0` is `CUDA_SUCCESS`.
pub type CUresult = i32;
pub const CUDA_SUCCESS: CUresult = 0;

/// `CUdevice`: an ordinal, not a pointer.
pub type CUdevice = i32;

/// `CUdeviceptr`: a genuine integer device address (not a host pointer), per the real API.
pub type CUdeviceptr = u64;

/// Opaque driver-owned handles. Basalt only ever passes these back into the driver; it never
/// dereferences the pointer itself, so a `#[repr(transparent)]` pointer-sized newtype is a
/// faithful and safe-to-move representation.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CUcontext(pub *mut c_void);

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CUmodule(pub *mut c_void);

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CUfunction(pub *mut c_void);

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CUstream(pub *mut c_void);

impl CUstream {
    /// The default (null) stream, used for every launch/copy in this crate — no
    /// multi-stream support yet.
    pub const NULL: CUstream = CUstream(std::ptr::null_mut());
}

pub type FnCuInit = unsafe extern "C" fn(flags: u32) -> CUresult;
pub type FnCuDeviceGetCount = unsafe extern "C" fn(count: *mut i32) -> CUresult;
pub type FnCuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult;
pub type FnCuCtxCreate =
    unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
pub type FnCuCtxDestroy = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
pub type FnCuCtxSynchronize = unsafe extern "C" fn() -> CUresult;

pub type FnCuModuleLoadData =
    unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult;
pub type FnCuModuleUnload = unsafe extern "C" fn(module: CUmodule) -> CUresult;
pub type FnCuModuleGetFunction =
    unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult;

pub type FnCuLaunchKernel = unsafe extern "C" fn(
    f: CUfunction,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult;

pub type FnCuMemAlloc = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
pub type FnCuMemFree = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;
pub type FnCuMemcpyHtoD = unsafe extern "C" fn(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult;
pub type FnCuMemcpyDtoH = unsafe extern "C" fn(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult;

pub type FnCuGetErrorString =
    unsafe extern "C" fn(error: CUresult, p_str: *mut *const c_char) -> CUresult;
