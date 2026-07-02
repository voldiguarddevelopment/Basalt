// HSA / CUDA driver loaders (dlopen), launch ABI.
//
// This crate has zero compile-time dependency on any GPU vendor's SDK — no CUDA headers, no
// `cuda-sys`-style crate, nothing linked at build time. Every driver entry point is resolved
// at runtime via `dlopen`/`dlsym` against the vendor's own shared library, so a Basalt binary
// built without any GPU present still links and runs; a GPU-touching feature simply reports
// "driver not found" instead of failing to compile.
//
// The CUDA half of that (this module tree) wraps the CUDA *Driver* API — `libcuda.so`, the
// library every user-space CUDA program ultimately talks to — not the higher-level CUDA
// *Runtime* API (`libcudart.so`) that `nvcc`-compiled `.cu` files normally call through. The
// Driver API is lower-level and vendor-stable: it lets any tool load a PTX or cubin image and
// run it without ever invoking NVCC. `cuModuleLoadData` in particular hands raw PTX *text* to
// the JIT compiler built into the driver itself, which is what makes it possible to take
// `basalt-ptx`'s output straight to real hardware with no offline NVIDIA toolchain in the
// loop at all.
//
// HSA (the ROCm-side loader for AMDGPU) is not implemented yet; see this crate's own
// module list as it grows.

mod context;
mod dl;
mod driver;
mod error;
mod ffi;
mod module;

pub use context::{CudaContext, DeviceBuffer};
pub use driver::CudaDriver;
pub use error::CudaError;
pub use module::{CudaFunction, CudaModule};
