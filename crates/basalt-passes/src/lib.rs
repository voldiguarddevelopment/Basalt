// Target-independent, BIR-to-BIR mid-end passes.
//
// `ssa::construct_ssa` promotes the synthetic memory-slot pattern `basalt-sema`'s lowering
// emits for locals/params/`__shared__`/`__constant__` variables (see that crate's `lower.rs`
// header) into real SSA form: `load`/`store` traffic through a slot's synthesized address is
// eliminated wherever it is safe to do so, replaced with direct value references and real
// `phi` instructions at control-flow merge points. `global`-space memory (an actual pointer
// dereference into device memory, not a synthesized local's home) is never touched — it stays
// exactly as written, regardless of how "SSA" the rest of a function becomes.

mod ssa;

pub use ssa::construct_ssa;
