// The HSA (Heterogeneous System Architecture) Runtime loader: the ROCm-side counterpart to
// `../driver.rs`'s CUDA Driver API loader, wrapping `libhsa-runtime64.so` the same way — `dlopen`
// at runtime via `../dl.rs`'s shared helpers, zero compile-time ROCm dependency, a
// `dlsym`-resolved function-pointer table per loaded library.
//
// The two loaders otherwise look nothing alike, because the two APIs don't: CUDA's Driver API
// hides dispatch behind a single `cuLaunchKernel` call. HSA has no such call — dispatching a
// kernel means constructing an AQL (Architected Queuing Language) kernel-dispatch packet by
// hand (`ffi::HsaKernelDispatchPacket`, a fixed 64-byte layout the HSA specification documents
// field-for-field) and writing it directly into the target queue's ring buffer before ringing
// its doorbell signal. `queue.rs` is where that happens.
//
// Scope note on fault handling: `hsa_queue_create` takes a callback the runtime invokes on a
// queue-level asynchronous error (bad dispatch, memory fault, …). This module wires that
// callback to a real Rust `extern "C" fn` (`queue::queue_error_callback`) that turns HSA's own
// status code into this crate's `HsaError`, stashed where `HsaQueue::last_fault` can retrieve
// it after the fact. That is the whole of this task's fault-handling scope: turning a real HSA
// fault into a structured error this crate already knows how to report. The fuller diagnostic
// system described elsewhere in this project — IBM-style completion codes, correlating a fault
// address against tracked allocations, a full dispatch snapshot — is separate, later work;
// nothing here builds toward those codes, only the plumbing they would eventually sit on top of.

mod error;
mod executable;
mod ffi;
mod queue;
mod runtime;

pub use error::HsaError;
pub use executable::{HsaExecutable, HsaKernel};
pub use ffi::{
    build_kernel_dispatch_header, HsaAgent, HsaKernelDispatchPacket, HsaRegion,
    HSA_PACKET_TYPE_KERNEL_DISPATCH,
};
pub use queue::HsaQueue;
pub use runtime::{HsaAgentInfo, HsaBuffer, HsaDeviceType, HsaRuntime};
