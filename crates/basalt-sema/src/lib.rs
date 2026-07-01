// Type checking over the `basalt-frontend-c` AST. Builds a symbol table across nested scopes
// (global -> namespace -> function -> block), resolves every identifier and type reference,
// and assigns a type to every expression, collecting problems as `basalt_diag::Diag`s instead
// of stopping at the first one. Shape inference (Triton tiles) and lowering to BIR are later
// work layered on top of this (ARCHITECTURE.md §6-7); this crate currently covers the CUDA-C
// side only.
//
// Public API: `check(&TranslationUnit) -> Vec<Diag>`. See `checker.rs` for the E-codes used,
// the scoping/redefinition rules, and the documented simplifications this pass makes.

mod checker;
mod scope;
mod ty;

pub use checker::check;

#[cfg(test)]
mod tests;
