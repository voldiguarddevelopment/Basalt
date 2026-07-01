// basalt-bir: BIR, Basalt's typed-SSA intermediate representation.
//
// Three pieces:
//   - `ir`/`ty`  — the in-memory arena data model: `Module`, `Function`, `Block`,
//                 `Inst`, `Op`, `Ty`. Instructions and blocks live in per-function `Vec`
//                 arenas addressed by `InstId`/`BlockId`, never `Rc`/`Box` graphs.
//   - `print`    — `print(&Module) -> String`, BIR's textual form.
//   - `parse`    — `parse(&str) -> Result<Module, ParseError>`, the inverse.
//
// `parse(print(m)) == m` for any `Module` whose arenas are laid out in construction order
// (which is the only way this crate ever builds one) — this is the BIR round-trip invariant,
// tested in `tests/roundtrip.rs`.
//
// # Textual grammar
//
// Lexical note: besides `( ) { } [ ] , : =` and `->`, every other token is a "word" — a
// maximal run of ASCII alphanumerics, `_ . @ % -`. Types, opcodes, predicates, value
// references (`%0`, `%arg1`), and block labels (`bb0`) are all words; the grammar below
// treats them as terminals.
//
// ```text
// module    := "module" "{" meta func* "}"
// meta      := ("launch_bounds" "max_threads" "=" int "min_blocks" "=" int)?
//              "shared_mem_bytes" int
//              "target_dtypes" scalar_ty*
//
// func      := "func" "@" ident "(" (ty ("," ty)*)? ")" "->" ty "{" block+ "}"
// block     := "bb" int ":" inst* term
//
// inst      := ["%" int "="] opcode ...
//
// opcode forms (opcode name, then its operand grammar):
//   const.i <ty> <int>
//   const.f <ty> <float>
//   add|sub|mul|div|rem|fadd|fsub|fmul|fdiv|frem
//   and|or|xor|shl|lshr|ashr           <ty> <val> "," <val>
//   icmp <pred> <ty> <val> "," <val>            ; pred: eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge
//   fcmp <pred> <ty> <val> "," <val>            ; pred: oeq|one|olt|ole|ogt|oge|ord|uno
//   select <ty> <val> "," <val> "," <val>
//   trunc|zext|sext|fptrunc|fpext|fptosi|fptoui|sitofp|uitofp|bitcast
//                                       <dst_ty> <src_ty> <val>
//   load  <ty> <ptr_ty> <val> "," "align" <int> [, "volatile"]
//   store <ty> <ptr_ty> <val> "," <val> "," "align" <int> [, "volatile"]
//   phi <ty> "[" ("bb" int "->" <val> ("," "bb" int "->" <val>)*)? "]"
//   tid.x|tid.y|tid.z|bid.x|bid.y|bid.z|bdim.x|bdim.y|bdim.z|gdim.x|gdim.y|gdim.z  <ty>
//   barrier
//   shuffle.idx|shuffle.up|shuffle.down|shuffle.xor  <ty> <val> "," <val>
//   ballot|vote.any|vote.all           <ty> <val>
//   atomic.add|atomic.sub|atomic.exch|atomic.min|atomic.max|atomic.and|atomic.or|atomic.xor
//                                       <ty> <ptr_ty> <val> "," <val>
//   atomic.cas <ty> <ptr_ty> <val> "," <val> "," <val>
//
// term      := "br" "bb" int
//            | "condbr" <val> "," "bb" int "," "bb" int
//            | "switch" <val> "," "default" "bb" int
//                  "[" (int "->" "bb" int ("," int "->" "bb" int)*)? "]"
//            | "ret" [<val>]
//
// ty        := "void" | scalar_ty | "ptr." space | "v" int scalar_ty
// scalar_ty := "i1" | "i8" | "i16" | "i32" | "i64" | "f16" | "f32" | "f64"
// space     := "global" | "shared" | "constant" | "local" | "param"
// val       := "%" int | "%arg" int
// ```
//
// A `%<id>` is always the arena index of the instruction that produced it (params use
// `%arg<index>` instead, an entirely separate namespace) — this is why the arena's
// construction order matters and why `InstId`/`BlockId` double as the printed numbering.

mod ir;
mod parse;
mod print;
mod ty;

pub use ir::{
    AtomicOp, BinOp, Block, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId,
    LaunchBounds, Module, Op, ShuffleKind, Term, ValRef,
};
pub use parse::{parse, ParseError};
pub use print::print;
pub use ty::{AddrSpace, Scalar, Ty};
