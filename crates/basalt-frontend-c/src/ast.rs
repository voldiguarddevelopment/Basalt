// AST for declarations, types, expressions, and statements: structs/enums/unions/typedefs/
// namespaces, the type grammar needed to describe them (scalars, pointers, arrays, tag/named
// types), function signatures and bodies, full expressions, and full C control flow.
//
// `TokenRange` remains only where something is deliberately out of scope even now: a
// `template<...>` body (structural recognition only, no instantiation — see `TemplateDecl`)
// and function-pointer declarators (not parsed at all — see `parse::Parser::parse_declarator`).
//
// Every node carries a `Span` so diagnostics can point at source text without re-deriving it.

use crate::token::{CharLit, FloatLit, IntLit, Span, StrLit};

/// A half-open range of indices into the token slice the parser was given, plus the `Span`
/// that range covers, so a later pass can either re-slice the tokens or just point at source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenRange {
    pub start: usize,
    pub end: usize,
    pub span: Span,
}

/// Assignment operators, `=` and its compound forms. All are right-associative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    AndAssign,
    OrAssign,
    XorAssign,
    ShlAssign,
    ShrAssign,
}

/// Binary operators below assignment/ternary in precedence; every level is left-associative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    LogOr,
    LogAnd,
    BitOr,
    BitXor,
    BitAnd,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Shl,
    Shr,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// Prefix unary operators that bind at cast-expression precedence (the C grammar's
/// `unary-operator cast-expression`). Prefix `++`/`--` and `sizeof` are their own `Expr`
/// variants below since they recurse into a unary-expression, not a cast-expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Neg,
    Not,
    BitNot,
    Deref,
    Addr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncDecOp {
    Inc,
    Dec,
}

/// A full expression, precedence and associativity already resolved during parsing (see
/// `parse::Parser`'s cascade from `parse_expr` down through `parse_primary`) — the tree shape
/// itself encodes precedence, there is no separate resolution pass later.
///
/// A parenthesized sub-expression `(e)` is not kept as its own node; the parser returns `e`
/// directly, since nothing downstream needs to distinguish "was parenthesized" once precedence
/// is baked into the tree shape.
///
/// Known gap: boolean literals (`true`/`false`), `nullptr`, and `this` are not modeled as their
/// own literal kinds yet (out of the scope this layer targets); they parse as `Error`
/// placeholders. Add dedicated variants if/when a caller needs them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    IntLit {
        value: IntLit,
        span: Span,
    },
    FloatLit {
        value: FloatLit,
        span: Span,
    },
    CharLit {
        value: CharLit,
        span: Span,
    },
    StrLit {
        value: StrLit,
        span: Span,
    },
    Ident {
        name: String,
        span: Span,
    },
    /// `a, b, c` — left-associative, source order (the comma operator).
    Comma {
        exprs: Vec<Expr>,
        span: Span,
    },
    Assign {
        op: AssignOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// `cond ? then_branch : else_branch`, right-associative.
    Ternary {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// `(Type)expr`. See `parse::Parser::next_starts_type` for the heuristic used to tell a
    /// cast apart from a parenthesized expression, and its known gap: a bare identifier right
    /// after `(` is never treated as a type, since this layer has no symbol table to know
    /// whether it names a typedef.
    Cast {
        ty: Type,
        expr: Box<Expr>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
        span: Span,
    },
    /// Prefix `++`/`--`.
    PreIncDec {
        op: IncDecOp,
        expr: Box<Expr>,
        span: Span,
    },
    /// Postfix `++`/`--`.
    PostIncDec {
        op: IncDecOp,
        expr: Box<Expr>,
        span: Span,
    },
    SizeofExpr {
        expr: Box<Expr>,
        span: Span,
    },
    SizeofType {
        ty: Type,
        span: Span,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    /// `base.name` (`arrow: false`) or `base->name` (`arrow: true`). CUDA builtins like
    /// `threadIdx.x` are ordinary instances of this — no special-casing needed here.
    Member {
        base: Box<Expr>,
        name: String,
        arrow: bool,
        span: Span,
    },
    /// A placeholder for an expression the parser couldn't make sense of, so the surrounding
    /// statement/declaration still has a well-formed tree instead of nothing. The accompanying
    /// `ParseError` is authoritative; this node carries no other information.
    Error {
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit { span, .. }
            | Expr::FloatLit { span, .. }
            | Expr::CharLit { span, .. }
            | Expr::StrLit { span, .. }
            | Expr::Ident { span, .. }
            | Expr::Comma { span, .. }
            | Expr::Assign { span, .. }
            | Expr::Ternary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Cast { span, .. }
            | Expr::Unary { span, .. }
            | Expr::PreIncDec { span, .. }
            | Expr::PostIncDec { span, .. }
            | Expr::SizeofExpr { span, .. }
            | Expr::SizeofType { span, .. }
            | Expr::Call { span, .. }
            | Expr::Index { span, .. }
            | Expr::Member { span, .. }
            | Expr::Error { span } => *span,
        }
    }
}

/// A statement, covering full C control flow (ARCHITECTURE.md §6). `case`/`default`/labeled
/// statements wrap the statement they label, matching the C grammar's labeled-statement
/// production: `case`/`default`/a label bind tighter than the statement sequence around them,
/// so `case 1: case 2: return x;` parses as `Case(1, Case(2, Return(x)))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Expr {
        expr: Expr,
        span: Span,
    },
    /// A bare `;`.
    Empty {
        span: Span,
    },
    Block {
        stmts: Vec<Stmt>,
        span: Span,
    },
    /// One or more declarators sharing a decl-specifier sequence (`int a = 1, *b;`).
    Decl {
        decls: Vec<VarDecl>,
        span: Span,
    },
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: Expr,
        span: Span,
    },
    /// `init` is `None` for a bare leading `;`; otherwise it is an expression- or
    /// declaration-statement that has already consumed its own trailing `;`. `cond`/`step` are
    /// `None` when that clause was left empty.
    For {
        init: Option<Box<Stmt>>,
        cond: Option<Expr>,
        step: Option<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    Switch {
        expr: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    /// `case value: stmt`. `value` is a parsed constant-expression, not constant-folded — that
    /// is a later stage's job.
    Case {
        value: Expr,
        stmt: Box<Stmt>,
        span: Span,
    },
    Default {
        stmt: Box<Stmt>,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    Return {
        expr: Option<Expr>,
        span: Span,
    },
    Label {
        name: String,
        stmt: Box<Stmt>,
        span: Span,
    },
    Goto {
        label: String,
        span: Span,
    },
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Expr { span, .. }
            | Stmt::Empty { span }
            | Stmt::Block { span, .. }
            | Stmt::Decl { span, .. }
            | Stmt::If { span, .. }
            | Stmt::While { span, .. }
            | Stmt::DoWhile { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Switch { span, .. }
            | Stmt::Case { span, .. }
            | Stmt::Default { span, .. }
            | Stmt::Break { span }
            | Stmt::Continue { span }
            | Stmt::Return { span, .. }
            | Stmt::Label { span, .. }
            | Stmt::Goto { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationUnit {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Struct(StructDecl),
    Union(UnionDecl),
    Enum(EnumDecl),
    Typedef(TypedefDecl),
    Namespace(NamespaceDecl),
    Function(FunctionDecl),
    Var(VarDecl),
    Template(TemplateDecl),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Struct(d) => d.span,
            Item::Union(d) => d.span,
            Item::Enum(d) => d.span,
            Item::Typedef(d) => d.span,
            Item::Namespace(d) => d.span,
            Item::Function(d) => d.span,
            Item::Var(d) => d.span,
            Item::Template(d) => d.span,
        }
    }
}

/// `const`/`volatile` qualifiers. Applies to a `Type` node (the pointee) or, on a `Pointer`
/// node, to the pointer itself (`T* const` vs `const T*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Qualifiers {
    pub is_const: bool,
    pub is_volatile: bool,
}

/// Resolved scalar type, after combining the multiset of specifier keywords a declaration
/// used (`unsigned long long int`, `signed char`, ...) into one canonical kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Void,
    Bool,
    Char,
    SChar,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    Float,
    Double,
    LongDouble,
    WcharT,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    Struct,
    Union,
    Enum,
}

/// A type as written in source. Pointers and arrays compose onto a base type via the C
/// declarator syntax (`int *p`, `int arr[10]`); see `parse::Parser::parse_declarator`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Scalar {
        kind: ScalarKind,
        quals: Qualifiers,
        span: Span,
    },
    /// `struct Foo` / `union Foo` / `enum Foo`. `name` is empty for an anonymous tag (the
    /// definition itself, if one was present, is still emitted as its own top-level `Item`).
    Tag {
        kind: TagKind,
        name: String,
        quals: Qualifiers,
        span: Span,
    },
    /// A plain identifier used as a type: a typedef'd name, a template type, or (since this
    /// layer has no symbol table) any other bare identifier in type position.
    Named {
        name: String,
        quals: Qualifiers,
        span: Span,
    },
    Pointer {
        pointee: Box<Type>,
        quals: Qualifiers,
        span: Span,
    },
    /// `size` is `None` for an incomplete array (`T[]`); otherwise the parsed size expression.
    Array {
        elem: Box<Type>,
        size: Option<Box<Expr>>,
        span: Span,
    },
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Scalar { span, .. }
            | Type::Tag { span, .. }
            | Type::Named { span, .. }
            | Type::Pointer { span, .. }
            | Type::Array { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDecl {
    pub ty: Type,
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructDecl {
    pub name: Option<String>,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnionDecl {
    pub name: Option<String>,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: String,
    /// The parsed initializer expression, if `= expr` was present. A constant-expression per
    /// the C grammar (no top-level assignment or comma); not constant-folded here.
    pub init: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDecl {
    pub name: Option<String>,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedefDecl {
    pub ty: Type,
    pub alias: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceDecl {
    pub name: String,
    pub items: Vec<Item>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamDecl {
    pub ty: Type,
    pub name: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub ret: Type,
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub variadic: bool,
    /// `None` for a prototype (`;`-terminated); `Some` is the parsed statement list of a
    /// `{ ... }` body (possibly empty).
    pub body: Option<Vec<Stmt>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDecl {
    pub ty: Type,
    pub name: String,
    /// The parsed initializer expression, if `= expr` was present (an assignment-expression
    /// per the C grammar — no top-level comma, since that separates declarators instead).
    pub init: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateParam {
    pub name: String,
    pub span: Span,
}

/// A `template<...> ...` declaration, recognized structurally only: the parameter list is
/// parsed, but the templated item itself (struct/function/etc.) is captured as an opaque
/// `body` range rather than parsed and instantiated. `name` is a best-effort guess (see
/// `parse::Parser::guess_template_item_name`), not a guarantee.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateDecl {
    pub params: Vec<TemplateParam>,
    pub name: Option<String>,
    pub body: TokenRange,
    pub span: Span,
}
