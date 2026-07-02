// The standard mid-end pipeline: every backend's `emit()` is handed the output of `optimize`,
// not the raw output of `basalt-sema`'s lowering pass, so these transforms apply uniformly
// regardless of which backend a module ends up on — see `basalt-cli`'s call sites for where
// this is wired in.
//
// # Chosen order and why
//
// `construct_ssa` runs first. `basalt-sema`'s lowering pass hands every local/param variable a
// synthetic memory slot (see that crate's `lower.rs` header) addressed through plain
// `load`/`store` traffic; both `constant_fold` and `licm` are explicit about treating `load` as
// an opaque, unanalyzable operand (see each module's own header) rather than reasoning about
// memory. Without promoting that slot traffic to real SSA values first, folding a chain like
// `int a = 2 + 3; int b = a * 4;` stops dead at the read of `a` — the multiply's left operand is
// a `load`, not a constant, even though `a`'s value is known at compile time. Running
// `construct_ssa` first turns that read into a direct reference to the instruction that computed
// it, which is exactly what lets the rest of the pipeline see through it.
//
// `constant_fold` then `eliminate_dead_code` follow immediately: folding first maximizes what
// DCE has to work with (a folded chain often leaves every intermediate `Bin`/`Cast` unread once
// its final constant has propagated forward), and `constant_fold`'s own single-forward-pass
// design (see that module's header) means there is nothing to gain from interleaving the two
// further or iterating either to a fixed point on its own.
//
// `licm` runs after that first DCE so it is hoisting out of a loop body that has already been
// stripped of dead work, then a final `eliminate_dead_code` follows it. `licm` itself never
// changes the static instruction count (see that module's header) — it only relocates
// instructions — so this second DCE pass is not expected to find anything new; it is cheap
// insurance against a hoist interacting with something unexpected, not a load-bearing step.
//
// A `Backend` that does its own `construct_ssa` internally (`basalt-x86`'s regalloc backend, for
// instance) simply finds no more promotable slots left when it runs — every local `optimize`
// could safely promote is already gone, and the ones it couldn't (a slot that fails the safety
// checks in `ssa.rs`) are exactly the ones no later `construct_ssa` call could promote either.
// Running it twice is therefore redundant but harmless, not incorrect.
pub fn optimize(module: &basalt_bir::Module) -> basalt_bir::Module {
    let module = crate::construct_ssa(module);
    let module = crate::constant_fold(&module);
    let module = crate::eliminate_dead_code(&module);
    let module = crate::licm(&module);
    crate::eliminate_dead_code(&module)
}
