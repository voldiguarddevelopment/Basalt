// `@triton.jit` frontend: parses a Python source file and hands back every decorated
// kernel function as this crate's own AST (see `ast.rs`).
//
// Scope (ARCHITECTURE.md §6, PLAN.md Phase 10, task P10-T1): parsing only. Tile shape
// inference, `constexpr` propagation, and lowering to BIR are `basalt-sema`'s job in later
// tasks; this crate's contract ends at "decorated functions parse to a walkable AST."
//
// Built on `ruff_python_parser`/`ruff_python_ast` (see `parse.rs`) rather than a hand-rolled
// Python grammar — the same reasoning that justified `rspirv` for `basalt-spirv`: a mature,
// actively maintained, pure-Rust implementation of a large general-purpose grammar this
// project has no reason to own. Unlike `basalt-frontend-c`, there is no separate
// preprocess/lex stage here (Python has no macro preprocessor, and the parser only exposes
// one "parse this source" entry point), and problems are reported as `basalt_diag::Diag`
// directly rather than a local error type, since this crate has no earlier stage of its own
// to keep decoupled from `basalt-diag`.

pub mod ast;
mod parse;

pub use parse::parse;
