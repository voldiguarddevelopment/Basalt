// Target-independent, BIR-to-BIR mid-end passes.
//
// `ssa::construct_ssa` promotes the synthetic memory-slot pattern `basalt-sema`'s lowering
// emits for locals/params/`__shared__`/`__constant__` variables (see that crate's `lower.rs`
// header) into real SSA form: `load`/`store` traffic through a slot's synthesized address is
// eliminated wherever it is safe to do so, replaced with direct value references and real
// `phi` instructions at control-flow merge points. `global`-space memory (an actual pointer
// dereference into device memory, not a synthesized local's home) is never touched — it stays
// exactly as written, regardless of how "SSA" the rest of a function becomes.
//
// `regalloc::allocate` consumes SSA-form BIR (typically `construct_ssa`'s output) and assigns
// every value a fixed register or spill-slot location via linear scan; see that module's
// header for the algorithm and its documented simplifications.
//
// `dom::Dominators`/`dom::detect_loops` compute a function's dominator relation and natural
// loops from its own control flow (source-level `for`/`while`/`do-while`, not the sequential
// per-thread loop a CPU backend synthesizes at emission time, which BIR has no visibility
// into). `constfold::constant_fold` and `dce::eliminate_dead_code` are straightforward BIR to
// BIR cleanups; `licm::licm` hoists loop-invariant pure computation out of those same natural
// loops. `divergence::analyze_divergence` classifies every value as uniform or divergent
// across a warp/block; it has no consumer yet in this tree — it exists for a later
// divergence-aware GPU register allocator to use. See each module's header for exactly what
// it covers and what it deliberately doesn't.

mod constfold;
mod dce;
mod divergence;
mod dom;
mod licm;
mod regalloc;
mod ssa;

pub use constfold::constant_fold;
pub use dce::eliminate_dead_code;
pub use divergence::{analyze_divergence, Divergence, DivergenceInfo};
pub use dom::{detect_loops, Dominators, NaturalLoop};
pub use licm::licm;
pub use regalloc::{allocate, Allocation, Location, RegClass, ValueId};
pub use ssa::construct_ssa;
