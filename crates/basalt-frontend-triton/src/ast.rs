// Triton kernel AST: this project's own node types for `@triton.jit`-decorated Python
// functions, built by `parse::parse` walking the tree `ruff_python_parser`/`ruff_python_ast`
// hand back. Deliberately not a reuse of `basalt-frontend-c`'s C-shaped AST — Triton kernels
// are ordinary Python with NumPy/Triton-flavored semantics (`tl.load`, `tl.constexpr`,
// `[:,None]` reshape, attribute-style intrinsics, ...), a different grammar that deserves its
// own tree rather than being forced into C's shape.
//
// Scope (P10-T1): a decorated function's name, its parameter list (including `tl.constexpr`
// parameters), and its body as a walkable statement/expression tree covering what a small,
// real Triton kernel actually uses. Tile shape inference, `constexpr` propagation, and BIR
// lowering are later tasks (P10-T2/T3); this layer only has to hand them something to walk.
// Anything genuinely out of scope right now (comprehensions, `lambda`, `with`/`try`/`match`,
// f-strings, ...) parses to an `Error` placeholder with an accompanying `Diag` reported by
// `parse::Parser` rather than being silently dropped or causing a panic.
//
// Every node carries a `basalt_diag::Span` so diagnostics can point at source text; there is
// no separate local span type to keep in sync, since this crate hands `Diag`s back directly
// (see lib.rs).

use basalt_diag::Span;

/// Every `@triton.jit`-decorated function found in a source file. Undecorated functions,
/// module-level statements (imports, plain assignments, classes, ...), and anything else in
/// the module are not modeled at all — they simply aren't kernels, which is not an error.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub kernels: Vec<KernelFn>,
}

/// A `@triton.jit`-decorated function. `params`/`body` are always populated on a best-effort
/// basis: a problem found while lowering one statement or expression does not throw away the
/// rest of the function, matching `basalt-frontend-c`'s "report many, never stop" standard.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelFn {
    pub name: String,
    pub params: Vec<Param>,
    /// The `-> T` return annotation, if written. Kernels return nothing at runtime; this is
    /// carried through unevaluated for whatever later stage wants it.
    pub returns: Option<Expr>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// A kernel parameter. `is_constexpr` is true when `annotation` is (or ends in) `.constexpr`
/// — Triton's convention for a compile-time-constant kernel argument (`BLOCK: tl.constexpr`),
/// distinct from an ordinary runtime tensor/pointer/scalar parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub annotation: Option<Expr>,
    pub is_constexpr: bool,
    pub default: Option<Expr>,
    pub span: Span,
}

/// A named or `**`-spread argument in a call (`tl.load(ptr, mask=m)`). `name` is `None` for a
/// `**kwargs`-style spread.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyword {
    pub name: Option<String>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    MatMul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    LShift,
    RShift,
    BitOr,
    BitXor,
    BitAnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Invert,
    Not,
    UAdd,
    USub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Is,
    IsNot,
    In,
    NotIn,
}

/// A full expression. Precedence/associativity is already resolved by the underlying Python
/// parser; this tree just carries the result over into this crate's own node types.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Name {
        name: String,
        span: Span,
    },
    /// Exact source text of an integer literal (`0x10`, `1_000`, ...), value interpretation
    /// deferred to a later stage — matches `basalt-frontend-c`'s `IntLit`/`FloatLit` split.
    IntLit {
        text: String,
        span: Span,
    },
    FloatLit {
        text: String,
        span: Span,
    },
    BoolLit {
        value: bool,
        span: Span,
    },
    NoneLit {
        span: Span,
    },
    /// Exact source text of a (non-interpolated) string literal, quotes included; escape
    /// processing is not this layer's job.
    StrLit {
        text: String,
        span: Span,
    },
    /// `a and b and c` / `a or b or c` — Python's `BoolOp` is variadic, not a binary tree.
    BoolOp {
        op: BoolOp,
        values: Vec<Expr>,
        span: Span,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// A (possibly chained) comparison: `a < b < c` is `left = a`, `ops = [Lt, Lt]`,
    /// `comparators = [b, c]`, matching Python's own AST shape.
    Compare {
        left: Box<Expr>,
        ops: Vec<CmpOp>,
        comparators: Vec<Expr>,
        span: Span,
    },
    /// `body if test else orelse`.
    Ternary {
        test: Box<Expr>,
        body: Box<Expr>,
        orelse: Box<Expr>,
        span: Span,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        keywords: Vec<Keyword>,
        span: Span,
    },
    /// `value.attr`, e.g. `tl.load` before it's applied as a call.
    Attribute {
        value: Box<Expr>,
        attr: String,
        span: Span,
    },
    /// `value[index]`. `index` may itself be a `Tuple`/`Slice` (`x[:, None]`); this layer
    /// does not interpret the shape those form, only parses them.
    Subscript {
        value: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Slice {
        lower: Option<Box<Expr>>,
        upper: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
        span: Span,
    },
    Tuple {
        elts: Vec<Expr>,
        span: Span,
    },
    List {
        elts: Vec<Expr>,
        span: Span,
    },
    /// A placeholder for an expression this layer doesn't yet model (comprehensions,
    /// `lambda`, f-strings, `yield`, ...). The `Diag` reported alongside it is authoritative;
    /// this node carries no other information — mirrors `basalt-frontend-c`'s `Expr::Error`.
    Error {
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Name { span, .. }
            | Expr::IntLit { span, .. }
            | Expr::FloatLit { span, .. }
            | Expr::BoolLit { span, .. }
            | Expr::NoneLit { span }
            | Expr::StrLit { span, .. }
            | Expr::BoolOp { span, .. }
            | Expr::UnaryOp { span, .. }
            | Expr::BinOp { span, .. }
            | Expr::Compare { span, .. }
            | Expr::Ternary { span, .. }
            | Expr::Call { span, .. }
            | Expr::Attribute { span, .. }
            | Expr::Subscript { span, .. }
            | Expr::Slice { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::List { span, .. }
            | Expr::Error { span } => *span,
        }
    }
}

/// A statement. `If`'s `elif`/`else` chain is flattened into nested `If`/`body` in `orelse`,
/// matching plain Python `ast`'s own shape (an `elif` is just an `If` nested in `orelse`).
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr {
        value: Expr,
        span: Span,
    },
    Assign {
        targets: Vec<Expr>,
        value: Expr,
        span: Span,
    },
    AugAssign {
        target: Expr,
        op: BinOp,
        value: Expr,
        span: Span,
    },
    /// `target: annotation = value` (`value` is `None` for a bare annotation with no
    /// initializer). Triton's `BLOCK: tl.constexpr = 256`-style module/local constants use
    /// this shape as well as `Param`'s own annotation.
    AnnAssign {
        target: Expr,
        annotation: Expr,
        value: Option<Expr>,
        span: Span,
    },
    If {
        test: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        span: Span,
    },
    For {
        target: Expr,
        iter: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        span: Span,
    },
    While {
        test: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Assert {
        test: Expr,
        msg: Option<Expr>,
        span: Span,
    },
    Pass {
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    /// See `Expr::Error`.
    Error {
        span: Span,
    },
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Expr { span, .. }
            | Stmt::Assign { span, .. }
            | Stmt::AugAssign { span, .. }
            | Stmt::AnnAssign { span, .. }
            | Stmt::If { span, .. }
            | Stmt::For { span, .. }
            | Stmt::While { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::Assert { span, .. }
            | Stmt::Pass { span }
            | Stmt::Break { span }
            | Stmt::Continue { span }
            | Stmt::Error { span } => *span,
        }
    }
}

/// True when `annotation` is Triton's `constexpr` marker: a bare `constexpr` name (the
/// `from triton.language import constexpr` import style) or an attribute path ending in
/// `.constexpr` (the usual `tl.constexpr`, or a longer path like `triton.language.constexpr`).
///
/// `pub`: `basalt-sema`'s tile-shape inference (P10-T2) reuses this exact predicate for
/// `AnnAssign`-bound locals rather than re-deriving it, so the two crates agree on what counts
/// as a `constexpr` marker.
pub fn is_constexpr_annotation(annotation: &Expr) -> bool {
    match annotation {
        Expr::Name { name, .. } => name == "constexpr",
        Expr::Attribute { attr, .. } => attr == "constexpr",
        _ => false,
    }
}
