// Hand-rolled x86-64: the oracle emitter plus a regalloc-based emitter for the CPU
// performance path. Both are `Backend` impls in this one crate (backend isolation is about
// targets, not about "how many code paths one ISA gets").
//
// `enc` is the shared low-level instruction encoder (no external assembler/encoder crate —
// see its own header for why); `oracle` is the stack-everything `Backend` impl built on it;
// `regalloc` is the SSA/linear-scan-based `Backend` impl, sharing `enc` but never `oracle`'s
// own code (see that module's header for the register budget and design).

mod enc;
mod oracle;
mod regalloc;

pub use oracle::X86Oracle;
pub use regalloc::X86Regalloc;
