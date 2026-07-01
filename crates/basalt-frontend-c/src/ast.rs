// AST for the declarations-and-types layer: structs/enums/unions/typedefs/namespaces, plus
// the type grammar needed to describe them (scalars, pointers, arrays, tag/named types) and
// function *signatures*. Statement/expression grammar is a later stage; anywhere this layer
// would need one (a function body, an array-size expression, an enum initializer) it stores
// an opaque `TokenRange` instead of attempting to parse it.
//
// Every node carries a `Span` so diagnostics can point at source text without re-deriving it.

use crate::token::Span;

/// A half-open range of indices into the token slice the parser was given, plus the `Span`
/// that range covers, so a later pass can either re-slice the tokens or just point at source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenRange {
    pub start: usize,
    pub end: usize,
    pub span: Span,
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
    /// `size` is `None` for an incomplete array (`T[]`); otherwise the token range of the
    /// (unparsed) size expression.
    Array {
        elem: Box<Type>,
        size: Option<TokenRange>,
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
    /// Token range of the (unparsed) initializer expression, if `= expr` was present.
    pub init: Option<TokenRange>,
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
    /// `None` for a prototype (`;`-terminated); `Some` captures the `{ ... }` body span,
    /// braces included, for a later stage to parse.
    pub body: Option<TokenRange>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDecl {
    pub ty: Type,
    pub name: String,
    /// Token range of the (unparsed) initializer expression, if `= expr` was present.
    pub init: Option<TokenRange>,
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
