// Type checking over the `basalt-frontend-c` AST. Builds a symbol table across nested scopes
// (global -> namespace -> function -> block), resolves every identifier and type reference,
// and assigns a type to every expression, collecting problems as `basalt_diag::Diag`s instead
// of stopping at the first one. Shape inference (Triton tiles) and lowering to BIR are later
// work layered on top of this (ARCHITECTURE.md §6-7); this crate currently covers the CUDA-C
// side only.
//
// Public API: `check(&TranslationUnit) -> Vec<Diag>`. See `checker.rs` for the E-codes used,
// the scoping/redefinition rules, and the documented simplifications this pass makes.
//
// `lower(&TranslationUnit) -> (basalt_bir::Module, Vec<Diag>)` lowers a checked translation
// unit to BIR; see `lower.rs`'s module header for the lowering design (stack-slot locals,
// address-space rules, and the BIR gaps this pass had to work around).

mod checker;
mod lower;
mod scope;
mod ty;

pub use checker::check;
pub use lower::lower;

#[cfg(test)]
mod tests;
