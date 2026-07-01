// Hand-rolled x86-64: the oracle emitter (this task) plus, later, a regalloc-based emitter for
// the CPU performance path. Both are `Backend` impls in this one crate (backend isolation is
// about targets, not about "how many code paths one ISA gets").
//
// `enc` is the shared low-level instruction encoder (no external assembler/encoder crate —
// see its own header for why); `oracle` is the stack-everything `Backend` impl built on it.

mod enc;
mod oracle;

pub use oracle::X86Oracle;
