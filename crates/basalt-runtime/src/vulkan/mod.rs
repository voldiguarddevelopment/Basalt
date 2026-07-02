// A real, hand-rolled Vulkan compute runtime — `dlopen("libvulkan.so.1")` plus hand-transcribed
// FFI, matching this crate's own established convention (`../dl.rs`, `../driver.rs`'s CUDA
// Driver API loader, `../hsa/`'s HSA Core Runtime loader): zero compile-time dependency on any
// Vulkan SDK, no `ash`/`vulkano`/`erupt` crate. A Basalt binary built without a Vulkan loader
// present still links and runs; this backend simply reports "loader not found" instead of
// failing to compile.
//
// This module exists to answer one question, empirically, for `basalt-spirv`'s benefit: can
// `basalt-spirv`'s emitted SPIR-V — deliberately `Kernel`-execution-model, `Physical64`-
// addressing, OpenCL-memory-model (see `../../basalt-spirv/src/emit.rs`'s own header for why) —
// be dispatched through a real Vulkan compute pipeline? Vulkan's compute pipeline API is built
// around the `GLCompute` execution model under `Logical` addressing, not `Kernel`/`Physical64`,
// so this was never expected to "just work"; the point of this module was to find out exactly
// where it breaks, against a real driver, rather than guess.
//
// # The finding (verified against real llvmpipe/Mesa on this project's test machine)
//
// Two real Vulkan API calls are involved, and they behave differently:
//
//   1. `vkCreateShaderModule` **succeeds** for a `Kernel`-execution-model module, including
//      `basalt-spirv`'s own real emitted bytes for `tests/kernels/vector_add.cu`. The Mesa
//      loader/driver combination validated here parses the SPIR-V binary's structure at
//      module-creation time — magic number, bound, well-formed instruction stream — without
//      checking the entry point's execution model against any particular pipeline stage. That
//      check is deferred to wherever it's actually meaningful.
//
//   2. `vkCreateComputePipelines` **fails** for that same shader module, returning
//      `VK_ERROR_UNKNOWN` (-13), while succeeding (returning `VK_SUCCESS`) for a semantically
//      identical, hand-written `GLCompute`-execution-model module. This is not a llvmpipe quirk
//      to work around: Vulkan's compute pipeline creation is specified to require the
//      `GLCompute` execution model unconditionally — no capability or extension relaxes it — so
//      a `Kernel`-model module is rejected by design, not by accident of this one driver.
//
// `pipeline.rs`'s `VulkanComputePipeline::create` is where both calls happen; see its own
// comments at each call site for the exact detail. `tests/vulkan_gpu_proof.rs` exercises both
// halves of this finding directly against `basalt-spirv`'s real output.
//
// # What this means for P9-T2's scope
//
// Bridging `Kernel`/`Physical64` to `GLCompute`/`Logical` inside `basalt-spirv` itself is *not*
// the small, additive, second-emit-path change it might look like at a glance. It requires two
// things BIR and `basalt-spirv` have no data for today, and inventing either would mean guessing
// rather than lowering (the same standard `emit.rs`'s own header already holds this backend to):
//
//   - A resource-binding ABI: which BIR pointer parameter becomes which descriptor set/binding,
//     and how scalar parameters pack into a push-constant block. This module's own test fixture
//     invents exactly such an ABI (see `pipeline.rs`'s header) — one storage-buffer binding per
//     pointer argument in order, one push-constant range for everything else — but that ABI is
//     defined and consumed only by this crate's own test, not by `basalt-spirv`, and was never
//     validated against anything beyond `vector_add`'s own shape.
//   - A `Logical`-addressing pointer story: `Logical` addressing has no pointer-to-integer
//     conversion and no "pointer plus a byte offset" at all — every access must go through
//     `OpAccessChain`'s *element indices* into a statically-typed composite. BIR represents
//     pointer arithmetic as ordinary integer `Bin::Add`/`Sub` on an opaque address (see
//     `emit.rs`'s "Pointer representation" section); recovering "this integer arithmetic is
//     actually an element-index computation with a known stride" for arbitrary BIR programs
//     (not just `vector_add`'s single, uniformly-strided access pattern) is a real static-
//     analysis problem, not a mechanical translation.
//
// Both of those are exactly the class of gap `emit.rs` already flags as the reason `basalt-spirv`
// targets `Kernel` in the first place. This module therefore does not attempt to add a
// `GLCompute` emission path to `basalt-spirv`; it scopes down to what is actually provable today:
// a real, working Vulkan compute runtime (this module), a real demonstration of exactly where
// `basalt-spirv`'s current output can and cannot go in that runtime (the finding above), and a
// full, real, hardware-dispatched proof of the runtime's own correctness using a hand-written
// `GLCompute` stand-in shader (see `tests/vulkan_gpu_proof.rs`'s own header for why that's the
// honest way to prove the *runtime* works without overstating what `basalt-spirv` can do today).
// A follow-up task to give `basalt-spirv` a real `GLCompute` emission path — with its own
// resource-binding ABI and pointer-arithmetic-to-access-chain lowering — is real, valuable,
// separate work, not a same-task extension of this one.
//
// # Layout
//
// Mirrors `../hsa/`'s file split: `ffi.rs` (raw types), `error.rs` (`VulkanError`),
// `instance.rs` (the loaded instance, physical-device enumeration — the `VulkanInstance`
// counterpart to `HsaRuntime`), `device.rs` (logical device, queue, host-visible buffer
// allocation), `pipeline.rs` (shader module / descriptor-set-layout / pipeline-layout /
// compute-pipeline construction — the `VulkanComputePipeline` counterpart to `HsaExecutable`),
// `dispatch.rs` (descriptor set binding, command buffer recording, submit, fence wait — the
// counterpart to `HsaQueue`'s manual AQL dispatch), `buffer.rs` (`VulkanBuffer`, the counterpart
// to `HsaBuffer`/`DeviceBuffer`).

mod buffer;
mod device;
mod dispatch;
mod error;
mod ffi;
mod instance;
mod pipeline;

pub use buffer::VulkanBuffer;
pub use device::VulkanDevice;
pub use error::VulkanError;
pub use instance::{VulkanDeviceType, VulkanInstance, VulkanPhysicalDeviceInfo};
pub use pipeline::VulkanComputePipeline;
