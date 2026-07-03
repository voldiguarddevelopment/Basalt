// Type checking over the `basalt-frontend-c` AST. Builds a symbol table across nested scopes
// (global -> namespace -> function -> block), resolves every identifier and type reference,
// and assigns a type to every expression, collecting problems as `basalt_diag::Diag`s instead
// of stopping at the first one. Lowering to BIR is later work layered on top of this
// (ARCHITECTURE.md §6-7); this half of the crate covers the CUDA-C side only.
//
// Public API: `check(&TranslationUnit) -> Vec<Diag>`. See `checker.rs` for the E-codes used,
// the scoping/redefinition rules, and the documented simplifications this pass makes.
//
// `lower(&TranslationUnit) -> (basalt_bir::Module, Vec<Diag>)` lowers a checked translation
// unit to BIR; see `lower.rs`'s module header for the lowering design (stack-slot locals,
// address-space rules, and the BIR gaps this pass had to work around).
//
// Triton tile-shape inference (ARCHITECTURE.md §6, PLAN.md P10-T2) is the other half:
// `check_triton(&basalt_frontend_triton::ast::Module) -> (Vec<KernelShapes>, Vec<Diag>)`. It
// deliberately does not share `ty.rs`'s `Ty` (CUDA-C-specific, built on
// `basalt_frontend_c::ast::ScalarKind`) — a Triton kernel's tiles are a different type system
// entirely. See `triton_ty.rs` for the tile-shape type and `triton_check.rs` for the pass
// itself. This is a sema barrier: it hands P10-T3 a shape-annotated representation to lower
// `tl.dot`/masked `tl.load`/`tl.store` from, but does no BIR lowering of its own.

mod checker;
mod lower;
mod scope;
mod triton_check;
mod triton_ty;
mod ty;

pub use checker::check;
pub use lower::lower;
pub use triton_check::{check_triton, KernelShapes};
pub use triton_ty::{Dim, Elem, TileTy};

#[cfg(test)]
mod tests;
