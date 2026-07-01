// Token model shared by the lexer and (eventually) the parser. Kept self-contained: no
// dependency on `basalt-diag` here, since that integration lands with the parser/sema stage.
//
// `Loc`/`Span` carry a byte offset in addition to line/col so callers can slice the original
// source directly instead of re-scanning to find a token's text.

use std::fmt;

/// A single point in source text. `line`/`col` are 1-based (matches editor conventions);
/// `offset` is the 0-based byte offset into the source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Loc {
    pub offset: u32,
    pub line: u32,
    pub col: u32,
}

impl Loc {
    pub fn new(offset: u32, line: u32, col: u32) -> Loc {
        Loc { offset, line, col }
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// A source range, `start` inclusive and `end` exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub start: Loc,
    pub end: Loc,
}

impl Span {
    pub fn new(start: Loc, end: Loc) -> Span {
        Span { start, end }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

/// A lexed token: its kind plus the span of source text it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Token {
        Token { kind, span }
    }
}

/// One lexed unit. Keywords are their own variant rather than folded into `Ident` — the
/// keyword set is fixed and small, so classifying it in the lexer saves every later stage
/// from re-doing a string match. CUDA qualifiers (`__global__`, `__device__`, `__host__`,
/// `__shared__`, `__constant__`, ...) and builtins (`threadIdx`, `blockIdx`, `blockDim`,
/// `gridDim`, `__syncthreads`, ...) are deliberately left as plain `Ident` — they are not
/// reserved words in the grammar (a user can shadow `threadIdx` in a non-kernel scope, and
/// `__global__` reads more like an attribute than a keyword), so tagging them here would bake
/// a parser-level decision into the lexer for no benefit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Ident(String),
    Keyword(Keyword),
    IntLit(IntLit),
    FloatLit(FloatLit),
    CharLit(CharLit),
    StrLit(StrLit),
    Punct(Punct),
    /// Emitted once at the end of the token stream so the parser never has to special-case
    /// running off the end of a `Vec<Token>`.
    Eof,
}

/// Reserved words recognized by the lexer. Scope matches the frontend's documented subset
/// (ARCHITECTURE.md §6: structs/enums/typedefs/namespaces, basic templates, operator
/// overloading, full control flow) rather than the entire C++ grammar; alternate-token
/// keywords (`and`, `or`, `xor`, ...) and newer C++ keywords are left out until the parser
/// needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    Alignas,
    Alignof,
    Auto,
    Bool,
    Break,
    Case,
    Catch,
    Char,
    Class,
    Const,
    Constexpr,
    ConstCast,
    Continue,
    Decltype,
    Default,
    Delete,
    Do,
    Double,
    DynamicCast,
    Else,
    Enum,
    Explicit,
    Extern,
    False,
    Float,
    For,
    Friend,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Mutable,
    Namespace,
    New,
    Noexcept,
    Nullptr,
    Operator,
    Private,
    Protected,
    Public,
    Register,
    ReinterpretCast,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    StaticAssert,
    StaticCast,
    Struct,
    Switch,
    Template,
    This,
    ThreadLocal,
    Throw,
    True,
    Try,
    Typedef,
    Typeid,
    Typename,
    Union,
    Unsigned,
    Using,
    Virtual,
    Void,
    Volatile,
    WcharT,
    While,
}

impl Keyword {
    /// Classifies `s` as a keyword, if it is one. Called after an identifier has already
    /// been scanned, so `s` is guaranteed to be a valid identifier shape.
    ///
    /// Named to mirror `std::str::FromStr` for familiarity, but returns `Option` rather than
    /// `Result` (there's no error to report — a non-keyword is simply `None`), so it isn't
    /// that trait; silence clippy's name-collision heuristic accordingly.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Keyword> {
        Some(match s {
            "alignas" => Keyword::Alignas,
            "alignof" => Keyword::Alignof,
            "auto" => Keyword::Auto,
            "bool" => Keyword::Bool,
            "break" => Keyword::Break,
            "case" => Keyword::Case,
            "catch" => Keyword::Catch,
            "char" => Keyword::Char,
            "class" => Keyword::Class,
            "const" => Keyword::Const,
            "constexpr" => Keyword::Constexpr,
            "const_cast" => Keyword::ConstCast,
            "continue" => Keyword::Continue,
            "decltype" => Keyword::Decltype,
            "default" => Keyword::Default,
            "delete" => Keyword::Delete,
            "do" => Keyword::Do,
            "double" => Keyword::Double,
            "dynamic_cast" => Keyword::DynamicCast,
            "else" => Keyword::Else,
            "enum" => Keyword::Enum,
            "explicit" => Keyword::Explicit,
            "extern" => Keyword::Extern,
            "false" => Keyword::False,
            "float" => Keyword::Float,
            "for" => Keyword::For,
            "friend" => Keyword::Friend,
            "goto" => Keyword::Goto,
            "if" => Keyword::If,
            "inline" => Keyword::Inline,
            "int" => Keyword::Int,
            "long" => Keyword::Long,
            "mutable" => Keyword::Mutable,
            "namespace" => Keyword::Namespace,
            "new" => Keyword::New,
            "noexcept" => Keyword::Noexcept,
            "nullptr" => Keyword::Nullptr,
            "operator" => Keyword::Operator,
            "private" => Keyword::Private,
            "protected" => Keyword::Protected,
            "public" => Keyword::Public,
            "register" => Keyword::Register,
            "reinterpret_cast" => Keyword::ReinterpretCast,
            "return" => Keyword::Return,
            "short" => Keyword::Short,
            "signed" => Keyword::Signed,
            "sizeof" => Keyword::Sizeof,
            "static" => Keyword::Static,
            "static_assert" => Keyword::StaticAssert,
            "static_cast" => Keyword::StaticCast,
            "struct" => Keyword::Struct,
            "switch" => Keyword::Switch,
            "template" => Keyword::Template,
            "this" => Keyword::This,
            "thread_local" => Keyword::ThreadLocal,
            "throw" => Keyword::Throw,
            "true" => Keyword::True,
            "try" => Keyword::Try,
            "typedef" => Keyword::Typedef,
            "typeid" => Keyword::Typeid,
            "typename" => Keyword::Typename,
            "union" => Keyword::Union,
            "unsigned" => Keyword::Unsigned,
            "using" => Keyword::Using,
            "virtual" => Keyword::Virtual,
            "void" => Keyword::Void,
            "volatile" => Keyword::Volatile,
            "wchar_t" => Keyword::WcharT,
            "while" => Keyword::While,
            _ => return None,
        })
    }
}

/// Radix an integer literal was written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBase {
    Dec,
    Oct,
    Hex,
    Bin,
}

/// An integer literal. `text` is the exact source slice (digits and suffix, including any
/// `0x`/`0b` prefix) so a later stage can re-derive the value with whatever width/overflow
/// rules it needs; the lexer only classifies the shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntLit {
    pub text: String,
    pub base: IntBase,
    pub unsigned: bool,
    /// 0 for no `l`/`L` suffix, 1 for `l`/`L`, 2 for `ll`/`LL`.
    pub long_len: u8,
}

/// Suffix on a floating-point literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSuffix {
    None,
    F,
    L,
}

/// A floating-point literal (always base 10; hex floating constants are not part of this
/// subset). `text` is the exact source slice, suffix included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloatLit {
    pub text: String,
    pub suffix: FloatSuffix,
}

/// A character literal. `value` is the decoded scalar value (post-escape-processing); for a
/// malformed literal (empty, or one the lexer had to recover from) it is best-effort and the
/// accompanying `LexError` is authoritative. `raw` is the exact source slice between (and
/// including) the quotes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharLit {
    pub value: u32,
    pub raw: String,
}

/// A string literal. `value` is the decoded content (escapes resolved); `raw` is the exact
/// source slice between (and including) the quotes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrLit {
    pub value: String,
    pub raw: String,
}

/// Operators and punctuation. Named after their shape, not their most common meaning, since
/// e.g. `*` and `&` are both binary operators and declarator decoration depending on parse
/// context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Punct {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    PlusPlus,
    MinusMinus,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    Amp,
    Pipe,
    Caret,
    Tilde,
    AmpAmp,
    PipePipe,
    Bang,
    AmpEq,
    PipeEq,
    CaretEq,
    Shl,
    Shr,
    ShlEq,
    ShrEq,
    Eq,
    EqEq,
    NotEq,
    Lt,
    Gt,
    Le,
    Ge,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    ColonColon,
    Question,
    Dot,
    Ellipsis,
    Arrow,
    Hash,
    HashHash,
}
