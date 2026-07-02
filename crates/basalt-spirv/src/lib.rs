// Hand-rolled SPIR-V (`Kernel` execution model, `Physical64`/`OpenCL`) emitter: this project's
// second GPU target
// and, like `basalt-ptx`, a virtual/SSA-native one — SPIR-V has no register allocator of its
// own to speak of, so there is no `basalt-x86/src/regalloc.rs`-style physical assignment here
// either. See `emit.rs`'s header for the full design (module layout, type/constant
// deduplication, builtin-variable mapping, control-flow lowering, and the refusal surface).

mod emit;

pub use emit::Spirv;
