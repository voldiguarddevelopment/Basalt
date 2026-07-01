// Recursive-descent parser over an already-lexed `&[Token]` (see `token.rs`/`lex.rs`); this
// stage never touches source text directly, so it stays decoupled from lexing/preprocessing.
//
// Scope (ARCHITECTURE.md §6): structs/enums/unions/typedefs/namespaces, the scalar/pointer/
// array type grammar, function signatures and bodies, full C-subset expressions, and full
// control flow. A `template<...>` header is still recognized structurally only — the
// templated item's body is captured as an opaque `TokenRange` rather than parsed and
// instantiated — and function-pointer declarators are still not parsed at all; both are
// documented gaps in `ast.rs`.
//
// Expression parsing is precedence climbing written out level by level (`parse_expr` down to
// `parse_primary`), matching the standard C precedence/associativity table; the resulting tree
// shape already encodes precedence, so there is no separate resolution pass later. A cast
// `(Type)expr` and a parenthesized expression `(expr)` are told apart by a lookahead heuristic
// with a known, documented gap — see `next_starts_type`. Declaration-statements reuse the same
// `parse_decl_specifiers`/`parse_declarator` machinery as top-level declarations.
//
// Like `lex`, this never aborts on the first error: `parse` always consumes the full token
// stream and returns whatever `ParseError`s it collected alongside the tree. A malformed item
// is resynchronized at the next plausible boundary (`;` or `}` at the current depth, or a
// keyword that starts a new item) so one bad declaration can't take the rest of the file with
// it or hang the parser; a malformed statement resynchronizes the same way at statement
// granularity (see `synchronize_stmt`).

use std::fmt;

use crate::ast::*;
use crate::token::{Keyword, Punct, Span, Token, TokenKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Expected one kind of token (`what` names it, e.g. `"identifier"` or `"';'"`) but found
    /// something else.
    Expected {
        what: String,
        found: String,
        span: Span,
    },
    /// A decl-specifier sequence combined type keywords in a way no C scalar type has (e.g.
    /// `float long`, `signed unsigned int`).
    InvalidTypeSpec { detail: String, span: Span },
    /// A declarator had no name where one was required (a struct field, a typedef alias, a
    /// plain variable declaration).
    MissingDeclaratorName { span: Span },
    /// A `}` with no matching open construct at this depth.
    UnmatchedBrace { span: Span },
}

impl ParseError {
    pub fn span(&self) -> Span {
        match self {
            ParseError::Expected { span, .. } => *span,
            ParseError::InvalidTypeSpec { span, .. } => *span,
            ParseError::MissingDeclaratorName { span } => *span,
            ParseError::UnmatchedBrace { span } => *span,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Expected { what, found, span } => {
                write!(f, "expected {what}, found {found} ({span})")
            }
            ParseError::InvalidTypeSpec { detail, span } => {
                write!(f, "invalid type specifier combination: {detail} ({span})")
            }
            ParseError::MissingDeclaratorName { span } => {
                write!(f, "declarator has no name ({span})")
            }
            ParseError::UnmatchedBrace { span } => write!(f, "unmatched '}}' ({span})"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parses `tokens` (must be `Eof`-terminated, as `lex` always produces) into a
/// `TranslationUnit` plus every error hit along the way.
pub fn parse(tokens: &[Token]) -> (TranslationUnit, Vec<ParseError>) {
    let mut p = Parser::new(tokens);
    let mut items = Vec::new();
    while !p.at_eof() {
        p.parse_item(&mut items);
    }
    (TranslationUnit { items }, p.errors)
}

fn describe(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Ident(s) => format!("identifier '{s}'"),
        TokenKind::Keyword(k) => format!("keyword '{k:?}'"),
        TokenKind::IntLit(_) => "integer literal".to_string(),
        TokenKind::FloatLit(_) => "floating-point literal".to_string(),
        TokenKind::CharLit(_) => "character literal".to_string(),
        TokenKind::StrLit(_) => "string literal".to_string(),
        TokenKind::Punct(p) => format!("'{p:?}'"),
        TokenKind::Eof => "end of file".to_string(),
    }
}

/// The source spelling of a `Punct`, used to synthesize an `operator+`-style function name
/// (see `Parser::parse_operator_function_name`).
fn punct_text(p: Punct) -> &'static str {
    match p {
        Punct::Plus => "+",
        Punct::Minus => "-",
        Punct::Star => "*",
        Punct::Slash => "/",
        Punct::Percent => "%",
        Punct::PlusPlus => "++",
        Punct::MinusMinus => "--",
        Punct::PlusEq => "+=",
        Punct::MinusEq => "-=",
        Punct::StarEq => "*=",
        Punct::SlashEq => "/=",
        Punct::PercentEq => "%=",
        Punct::Amp => "&",
        Punct::Pipe => "|",
        Punct::Caret => "^",
        Punct::Tilde => "~",
        Punct::AmpAmp => "&&",
        Punct::PipePipe => "||",
        Punct::Bang => "!",
        Punct::AmpEq => "&=",
        Punct::PipeEq => "|=",
        Punct::CaretEq => "^=",
        Punct::Shl => "<<",
        Punct::Shr => ">>",
        Punct::ShlEq => "<<=",
        Punct::ShrEq => ">>=",
        Punct::Eq => "=",
        Punct::EqEq => "==",
        Punct::NotEq => "!=",
        Punct::Lt => "<",
        Punct::Gt => ">",
        Punct::Le => "<=",
        Punct::Ge => ">=",
        Punct::LParen => "(",
        Punct::RParen => ")",
        Punct::LBrace => "{",
        Punct::RBrace => "}",
        Punct::LBracket => "[",
        Punct::RBracket => "]",
        Punct::Comma => ",",
        Punct::Semi => ";",
        Punct::Colon => ":",
        Punct::ColonColon => "::",
        Punct::Question => "?",
        Punct::Dot => ".",
        Punct::Ellipsis => "...",
        Punct::Arrow => "->",
        Punct::Hash => "#",
        Punct::HashHash => "##",
    }
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    errors: Vec<ParseError>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Parser<'a> {
        Parser {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.cur().kind, TokenKind::Eof)
    }

    fn cur(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn prev_span_end(&self) -> Span {
        let i = self.pos.saturating_sub(1);
        self.tokens[i].span
    }

    /// Advances past the current token and returns it. Pinned at `Eof` once reached, so
    /// callers never have to guard against running off the end of the slice.
    fn bump(&mut self) -> Token {
        let t = self.cur().clone();
        if !matches!(t.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        t
    }

    fn check_punct(&self, p: Punct) -> bool {
        matches!(self.cur().kind, TokenKind::Punct(pp) if pp == p)
    }

    fn eat_punct(&mut self, p: Punct) -> bool {
        if self.check_punct(p) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn peek_keyword(&self) -> Option<Keyword> {
        match self.cur().kind {
            TokenKind::Keyword(k) => Some(k),
            _ => None,
        }
    }

    fn check_keyword(&self, k: Keyword) -> bool {
        self.peek_keyword() == Some(k)
    }

    fn eat_keyword(&mut self, k: Keyword) -> bool {
        if self.check_keyword(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn ident_here(&self) -> Option<String> {
        match &self.cur().kind {
            TokenKind::Ident(s) => Some(s.clone()),
            _ => None,
        }
    }

    fn error_expected(&mut self, what: &str) {
        self.errors.push(ParseError::Expected {
            what: what.to_string(),
            found: describe(&self.cur().kind),
            span: self.cur().span,
        });
    }

    fn expect_punct(&mut self, p: Punct, what: &str) -> Option<Span> {
        if self.check_punct(p) {
            Some(self.bump().span)
        } else {
            self.error_expected(what);
            None
        }
    }

    /// Resynchronizes after an error inside a top-level (or namespace-level) item: skips
    /// tokens, tracking brace nesting, until a `;` at depth 0 (consumed), a `}` that closes
    /// the enclosing construct (left unconsumed, for the caller to see), a keyword that
    /// clearly starts a new item, or `Eof`.
    fn synchronize(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Eof => return,
                TokenKind::Punct(Punct::LBrace) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RBrace) => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::Semi) if depth == 0 => {
                    self.bump();
                    return;
                }
                TokenKind::Keyword(k)
                    if depth == 0
                        && matches!(
                            k,
                            Keyword::Struct
                                | Keyword::Union
                                | Keyword::Enum
                                | Keyword::Typedef
                                | Keyword::Namespace
                                | Keyword::Template
                        ) =>
                {
                    return;
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Skips whatever declaration follows a `template<...>` header: a brace-delimited
    /// definition (trailing `;`, if present, consumed too) or a `;`-terminated prototype.
    fn skip_one_declaration(&mut self) {
        let mut depth = 0i32;
        let mut had_brace = false;
        loop {
            match self.cur().kind {
                TokenKind::Punct(Punct::LBrace) => {
                    had_brace = true;
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::LParen | Punct::LBracket) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RBrace | Punct::RParen | Punct::RBracket) => {
                    if depth == 0 {
                        self.bump();
                        continue;
                    }
                    depth -= 1;
                    self.bump();
                    if depth == 0 && had_brace {
                        self.eat_punct(Punct::Semi);
                        break;
                    }
                }
                TokenKind::Punct(Punct::Semi) if depth == 0 => {
                    self.bump();
                    break;
                }
                TokenKind::Eof => break,
                _ => {
                    self.bump();
                }
            }
        }
    }

    // ---- items -----------------------------------------------------------------------

    fn parse_item(&mut self, out: &mut Vec<Item>) {
        match self.cur().kind {
            TokenKind::Eof => {}
            // A stray `;` between declarations is harmless.
            TokenKind::Punct(Punct::Semi) => {
                self.bump();
            }
            TokenKind::Punct(Punct::RBrace) => {
                let span = self.cur().span;
                self.errors.push(ParseError::UnmatchedBrace { span });
                self.bump();
            }
            TokenKind::Keyword(Keyword::Namespace) => self.parse_namespace(out),
            TokenKind::Keyword(Keyword::Typedef) => self.parse_typedef(out),
            TokenKind::Keyword(Keyword::Template) => self.parse_template(out),
            _ => self.parse_decl_or_def(out),
        }
    }

    fn parse_namespace(&mut self, out: &mut Vec<Item>) {
        let start = self.cur().span.start;
        self.bump(); // `namespace`
        let name = match self.ident_here() {
            Some(n) => {
                self.bump();
                n
            }
            None => {
                self.error_expected("identifier");
                self.synchronize();
                return;
            }
        };
        if self.expect_punct(Punct::LBrace, "'{'").is_none() {
            self.synchronize();
            return;
        }
        let mut items = Vec::new();
        while !self.check_punct(Punct::RBrace) && !self.at_eof() {
            self.parse_item(&mut items);
        }
        let end = if self.check_punct(Punct::RBrace) {
            self.bump().span.end
        } else {
            self.error_expected("'}'");
            self.prev_span_end().end
        };
        out.push(Item::Namespace(NamespaceDecl {
            name,
            items,
            span: Span::new(start, end),
        }));
    }

    fn parse_typedef(&mut self, out: &mut Vec<Item>) {
        let start = self.cur().span.start;
        self.bump(); // `typedef`
        let base = match self.parse_decl_specifiers(out) {
            Some((ty, _)) => ty,
            None => {
                self.synchronize();
                return;
            }
        };
        loop {
            let (ty, name, dspan) = self.parse_declarator(base.clone());
            match name {
                Some(alias) => out.push(Item::Typedef(TypedefDecl {
                    ty,
                    alias,
                    span: Span::new(start, dspan.end),
                })),
                None => self
                    .errors
                    .push(ParseError::MissingDeclaratorName { span: dspan }),
            }
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        if self.expect_punct(Punct::Semi, "';'").is_none() {
            self.synchronize();
        }
    }

    /// A `template<...>` header followed by whatever it templates. Recognized structurally
    /// only: the parameter list is parsed, the templated item is captured whole as a
    /// `TokenRange` (see the module header for why).
    fn parse_template(&mut self, out: &mut Vec<Item>) {
        let start = self.cur().span.start;
        self.bump(); // `template`
        if self.expect_punct(Punct::Lt, "'<'").is_none() {
            self.synchronize();
            return;
        }
        let params = self.parse_template_params();
        let name = self.guess_template_item_name();
        let body_start = self.pos;
        let body_start_span = self.cur().span.start;
        self.skip_one_declaration();
        let body_end = self.pos;
        let body_span_end = self.prev_span_end().end;
        out.push(Item::Template(TemplateDecl {
            params,
            name,
            body: TokenRange {
                start: body_start,
                end: body_end,
                span: Span::new(body_start_span, body_span_end),
            },
            span: Span::new(start, body_span_end),
        }));
    }

    /// Parses the comma-separated parameter list of a `template< ... >` header (`self.pos`
    /// must be positioned right after the opening `<`, at depth 1). Each parameter's "name"
    /// is a heuristic: the last identifier in its token run before any `= default`, which
    /// covers both type parameters (`typename T`, `class T`) and simple non-type parameters
    /// (`int N`) without needing full type/expression parsing.
    fn parse_template_params(&mut self) -> Vec<TemplateParam> {
        let mut params = Vec::new();
        let mut depth = 1i32;
        let mut seg_start = self.pos;
        loop {
            match self.cur().kind {
                TokenKind::Punct(Punct::Lt) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::Gt) => {
                    depth -= 1;
                    let closing = depth == 0;
                    let seg_end = self.pos;
                    self.bump();
                    if closing {
                        self.finish_template_param(&mut params, seg_start, seg_end);
                        break;
                    }
                }
                TokenKind::Punct(Punct::Shr) => {
                    // `>>` closes two angle-bracket levels at once (the classic
                    // `vector<vector<int>>` lexing snag).
                    depth -= 2;
                    let seg_end = self.pos;
                    let closing = depth <= 0;
                    self.bump();
                    if closing {
                        self.finish_template_param(&mut params, seg_start, seg_end);
                        break;
                    }
                }
                TokenKind::Punct(Punct::Comma) if depth == 1 => {
                    let seg_end = self.pos;
                    self.bump();
                    self.finish_template_param(&mut params, seg_start, seg_end);
                    seg_start = self.pos;
                }
                TokenKind::Eof => break,
                _ => {
                    self.bump();
                }
            }
        }
        params
    }

    fn finish_template_param(
        &mut self,
        params: &mut Vec<TemplateParam>,
        seg_start: usize,
        seg_end: usize,
    ) {
        if seg_start >= seg_end {
            return;
        }
        let mut limit = seg_end;
        for i in seg_start..seg_end {
            if matches!(self.tokens[i].kind, TokenKind::Punct(Punct::Eq)) {
                limit = i;
                break;
            }
        }
        let mut name = None;
        let mut span = None;
        for tok in &self.tokens[seg_start..limit] {
            if let TokenKind::Ident(s) = &tok.kind {
                name = Some(s.clone());
                span = Some(tok.span);
            }
        }
        if let Some(name) = name {
            params.push(TemplateParam {
                name,
                span: span.unwrap_or(self.tokens[seg_start].span),
            });
        }
    }

    /// Best-effort guess at the name of the item following a `template<...>` header, without
    /// parsing it: `struct`/`class`/`union` followed by an identifier, or (for a function
    /// template) the identifier immediately before the parameter-list `(`. Bounded so a
    /// malformed header can't scan the rest of the file.
    fn guess_template_item_name(&self) -> Option<String> {
        if let TokenKind::Keyword(k) = self.cur().kind {
            if matches!(k, Keyword::Struct | Keyword::Class | Keyword::Union) {
                if let Some(next) = self.tokens.get(self.pos + 1) {
                    if let TokenKind::Ident(n) = &next.kind {
                        return Some(n.clone());
                    }
                }
            }
        }
        let mut last_ident = None;
        let mut i = self.pos;
        let limit = (self.pos + 64).min(self.tokens.len());
        while i < limit {
            match &self.tokens[i].kind {
                TokenKind::Punct(Punct::LParen | Punct::LBrace | Punct::Semi) => break,
                TokenKind::Ident(n) => last_ident = Some(n.clone()),
                TokenKind::Eof => break,
                _ => {}
            }
            i += 1;
        }
        last_ident
    }

    /// The common path: decl-specifiers, then a declarator, then either a function
    /// (prototype or definition) or one or more variable declarators.
    fn parse_decl_or_def(&mut self, out: &mut Vec<Item>) {
        let start = self.cur().span.start;
        let (base, is_forward_tag) = match self.parse_decl_specifiers(out) {
            Some(r) => r,
            None => {
                self.synchronize();
                return;
            }
        };

        if self.eat_punct(Punct::Semi) {
            if is_forward_tag {
                self.push_forward_tag_decl(out, &base);
            }
            return;
        }

        let (ty, name, dspan) = self.parse_declarator(base.clone());

        if self.eat_punct(Punct::LParen) {
            let (params, variadic) = self.parse_param_list(out);
            let fn_name = name.unwrap_or_default();
            if self.check_punct(Punct::LBrace) {
                let (stmts, body_span) = self.parse_block_stmts();
                let span = Span::new(start, body_span.end);
                out.push(Item::Function(FunctionDecl {
                    ret: ty,
                    name: fn_name,
                    params,
                    variadic,
                    body: Some(stmts),
                    span,
                }));
            } else if self.eat_punct(Punct::Semi) {
                out.push(Item::Function(FunctionDecl {
                    ret: ty,
                    name: fn_name,
                    params,
                    variadic,
                    body: None,
                    span: Span::new(start, self.prev_span_end().end),
                }));
            } else {
                self.error_expected("';' or '{'");
                self.synchronize();
            }
            return;
        }

        let mut cur_name = name;
        let mut cur_ty = ty;
        let mut cur_span = dspan;
        loop {
            let init = if self.eat_punct(Punct::Eq) {
                Some(self.parse_assign())
            } else {
                None
            };
            match cur_name {
                Some(n) => out.push(Item::Var(VarDecl {
                    ty: cur_ty,
                    name: n,
                    init,
                    span: cur_span,
                })),
                None => self
                    .errors
                    .push(ParseError::MissingDeclaratorName { span: cur_span }),
            }
            if !self.eat_punct(Punct::Comma) {
                break;
            }
            let (ty2, name2, span2) = self.parse_declarator(base.clone());
            cur_ty = ty2;
            cur_name = name2;
            cur_span = span2;
        }
        if self.expect_punct(Punct::Semi, "';'").is_none() {
            self.synchronize();
        }
    }

    fn push_forward_tag_decl(&mut self, out: &mut Vec<Item>, base: &Type) {
        if let Type::Tag {
            kind, name, span, ..
        } = base
        {
            let nm = if name.is_empty() {
                None
            } else {
                Some(name.clone())
            };
            match kind {
                TagKind::Struct => out.push(Item::Struct(StructDecl {
                    name: nm,
                    fields: Vec::new(),
                    span: *span,
                })),
                TagKind::Union => out.push(Item::Union(UnionDecl {
                    name: nm,
                    fields: Vec::new(),
                    span: *span,
                })),
                TagKind::Enum => out.push(Item::Enum(EnumDecl {
                    name: nm,
                    variants: Vec::new(),
                    span: *span,
                })),
            }
        }
    }

    // ---- types -------------------------------------------------------------------------

    fn consume_quals(&mut self, quals: &mut Qualifiers) {
        loop {
            match self.peek_keyword() {
                Some(Keyword::Const) => {
                    quals.is_const = true;
                    self.bump();
                }
                Some(Keyword::Volatile) => {
                    quals.is_volatile = true;
                    self.bump();
                }
                _ => break,
            }
        }
    }

    /// Parses a decl-specifier sequence: qualifiers plus exactly one of a scalar-keyword run,
    /// a tag (`struct`/`union`/`enum`), or a plain identifier naming a type. Returns the
    /// resulting `Type` and whether it was a bare tag reference with no body (used by the
    /// caller to tell a forward declaration like `struct Foo;` apart from a vacuous one).
    fn parse_decl_specifiers(&mut self, out: &mut Vec<Item>) -> Option<(Type, bool)> {
        let start = self.cur().span.start;
        let mut quals = Qualifiers::default();
        self.consume_quals(&mut quals);

        let (mut ty, is_forward_tag) = match self.peek_keyword() {
            Some(Keyword::Struct | Keyword::Union | Keyword::Enum) => self.parse_tag(out),
            Some(k) if is_scalar_keyword(k) => (self.parse_scalar(start), false),
            _ => match self.ident_here() {
                Some(name) => {
                    let span = self.bump().span;
                    (
                        Type::Named {
                            name,
                            quals: Qualifiers::default(),
                            span,
                        },
                        false,
                    )
                }
                None => {
                    self.error_expected("type specifier");
                    return None;
                }
            },
        };

        self.consume_quals(&mut quals);
        apply_quals(&mut ty, quals);
        Some((ty, is_forward_tag))
    }

    fn parse_scalar(&mut self, start: crate::token::Loc) -> Type {
        let mut c = ScalarCounts::default();
        loop {
            match self.peek_keyword() {
                Some(Keyword::Void) => {
                    c.void += 1;
                    self.bump();
                }
                Some(Keyword::Bool) => {
                    c.bool_ += 1;
                    self.bump();
                }
                Some(Keyword::Char) => {
                    c.char_ += 1;
                    self.bump();
                }
                Some(Keyword::Short) => {
                    c.short += 1;
                    self.bump();
                }
                Some(Keyword::Int) => {
                    c.int += 1;
                    self.bump();
                }
                Some(Keyword::Long) => {
                    c.long += 1;
                    self.bump();
                }
                Some(Keyword::Float) => {
                    c.float += 1;
                    self.bump();
                }
                Some(Keyword::Double) => {
                    c.double += 1;
                    self.bump();
                }
                Some(Keyword::Signed) => {
                    c.signed += 1;
                    self.bump();
                }
                Some(Keyword::Unsigned) => {
                    c.unsigned += 1;
                    self.bump();
                }
                Some(Keyword::WcharT) => {
                    c.wchar += 1;
                    self.bump();
                }
                _ => break,
            }
        }
        let end = self.prev_span_end().end;
        let span = Span::new(start, end);
        let kind = resolve_scalar(&c, span, &mut self.errors);
        Type::Scalar {
            kind,
            quals: Qualifiers::default(),
            span,
        }
    }

    /// Parses `struct`/`union`/`enum` [name] [`{` body `}`]. If a body is present, the
    /// resulting declaration is pushed onto `out` immediately; the returned `Type::Tag` only
    /// ever carries the tag's name, never its members.
    fn parse_tag(&mut self, out: &mut Vec<Item>) -> (Type, bool) {
        let start = self.cur().span.start;
        let kind = match self.bump().kind {
            TokenKind::Keyword(Keyword::Struct) => TagKind::Struct,
            TokenKind::Keyword(Keyword::Union) => TagKind::Union,
            TokenKind::Keyword(Keyword::Enum) => TagKind::Enum,
            _ => unreachable!("parse_tag called on a non-tag keyword"),
        };
        let name = self.ident_here();
        if name.is_some() {
            self.bump();
        }
        let mut had_body = false;
        if self.check_punct(Punct::LBrace) {
            had_body = true;
            match kind {
                TagKind::Enum => {
                    let variants = self.parse_enum_variants();
                    let span = Span::new(start, self.prev_span_end().end);
                    out.push(Item::Enum(EnumDecl {
                        name: name.clone(),
                        variants,
                        span,
                    }));
                }
                TagKind::Struct | TagKind::Union => {
                    let fields = self.parse_field_list(out);
                    let span = Span::new(start, self.prev_span_end().end);
                    if kind == TagKind::Struct {
                        out.push(Item::Struct(StructDecl {
                            name: name.clone(),
                            fields,
                            span,
                        }));
                    } else {
                        out.push(Item::Union(UnionDecl {
                            name: name.clone(),
                            fields,
                            span,
                        }));
                    }
                }
            }
        }
        let end = self.prev_span_end().end;
        (
            Type::Tag {
                kind,
                name: name.unwrap_or_default(),
                quals: Qualifiers::default(),
                span: Span::new(start, end),
            },
            !had_body,
        )
    }

    /// Parses the `{ ... }` field list of a struct/union. `self.pos` must be at the opening
    /// `{`.
    fn parse_field_list(&mut self, out: &mut Vec<Item>) -> Vec<FieldDecl> {
        self.bump(); // `{`
        let mut fields = Vec::new();
        while !self.check_punct(Punct::RBrace) && !self.at_eof() {
            let before_pos = self.pos;
            self.parse_field(out, &mut fields);
            if self.pos == before_pos {
                // Safety net: guarantee progress even if a future edit adds a path in
                // `parse_field` that doesn't consume or resynchronize.
                self.bump();
            }
        }
        self.expect_punct(Punct::RBrace, "'}'");
        fields
    }

    fn parse_field(&mut self, out: &mut Vec<Item>, fields: &mut Vec<FieldDecl>) {
        let base = match self.parse_decl_specifiers(out) {
            Some((ty, _)) => ty,
            None => {
                self.synchronize_field();
                return;
            }
        };
        if self.eat_punct(Punct::Semi) {
            // A nested tag definition with no member of that type, e.g. `struct { ... };`.
            return;
        }
        loop {
            let (ty, name, span) = self.parse_declarator(base.clone());
            match name {
                Some(n) => fields.push(FieldDecl { ty, name: n, span }),
                None => self.errors.push(ParseError::MissingDeclaratorName { span }),
            }
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        if self.expect_punct(Punct::Semi, "';'").is_none() {
            self.synchronize_field();
        }
    }

    /// Like `synchronize`, but stops at a `}` too (field lists don't have their own nested
    /// item-start keywords worth special-casing beyond that).
    fn synchronize_field(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Eof => return,
                TokenKind::Punct(Punct::LBrace) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RBrace) => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::Semi) if depth == 0 => {
                    self.bump();
                    return;
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    fn parse_enum_variants(&mut self) -> Vec<EnumVariant> {
        self.bump(); // `{`
        let mut variants = Vec::new();
        while !self.check_punct(Punct::RBrace) && !self.at_eof() {
            let name = match self.ident_here() {
                Some(n) => {
                    self.bump();
                    n
                }
                None => {
                    self.error_expected("enumerator name");
                    self.synchronize_field();
                    continue;
                }
            };
            let vstart = self.prev_span_end();
            let init = if self.eat_punct(Punct::Eq) {
                Some(self.parse_conditional())
            } else {
                None
            };
            let span = Span::new(vstart.start, self.prev_span_end().end);
            variants.push(EnumVariant { name, init, span });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RBrace, "'}'");
        variants
    }

    /// Parses pointer/array declarator syntax around `base`, plus the declared name if
    /// present. `base` composes as: leading `*`s wrap it first (innermost to outermost, in
    /// the order written), then trailing `[...]`s wrap that result (also in written order) —
    /// matching C's "arrays of pointers, not pointers to arrays" default precedence. A
    /// parenthesized declarator (`int (*fp)(int)`, function pointers) is not supported; see
    /// the crate-level gap note.
    fn parse_declarator(&mut self, base: Type) -> (Type, Option<String>, Span) {
        let start = self.cur().span.start;
        let mut star_quals = Vec::new();
        while self.eat_punct(Punct::Star) {
            let mut q = Qualifiers::default();
            self.consume_quals(&mut q);
            star_quals.push(q);
        }
        let mut ty = base;
        for q in star_quals {
            let span = Span::new(start, self.prev_span_end().end);
            ty = Type::Pointer {
                pointee: Box::new(ty),
                quals: q,
                span,
            };
        }
        let name = if self.check_keyword(Keyword::Operator) {
            Some(self.parse_operator_function_name())
        } else {
            let n = self.ident_here();
            if n.is_some() {
                self.bump();
            }
            n
        };
        while self.check_punct(Punct::LBracket) {
            let lb_span = self.cur().span;
            self.bump();
            let size = if self.check_punct(Punct::RBracket) {
                None
            } else {
                Some(Box::new(self.parse_conditional()))
            };
            let rb_end = self
                .expect_punct(Punct::RBracket, "']'")
                .map(|s| s.end)
                .unwrap_or(self.cur().span.start);
            ty = Type::Array {
                elem: Box::new(ty),
                size,
                span: Span::new(lb_span.start, rb_end),
            };
        }
        let end = self.prev_span_end().end;
        (ty, name, Span::new(start, end))
    }

    /// Parses `operator` followed by the operator token(s) it overloads (`operator+`,
    /// `operator==`, `operator[]`, `operator()`, ...) into a synthesized function name, so a
    /// declarator like `T operator+(T other)` gets a stable name to hang a `FunctionDecl` off
    /// of. Overload *resolution* — matching `a + b` back to the right `operator+` — is a later
    /// stage's problem; the parser only needs the name to exist. Only operators this frontend
    /// actually lexes as `Punct` are recognized; the (unusual) `operator new`/`operator delete`
    /// forms are not.
    fn parse_operator_function_name(&mut self) -> String {
        self.bump(); // `operator`
        let mut name = String::from("operator");
        match self.cur().kind {
            TokenKind::Punct(p) => {
                name.push_str(punct_text(p));
                self.bump();
                if p == Punct::LParen {
                    self.expect_punct(Punct::RParen, "')'");
                    name.push(')');
                } else if p == Punct::LBracket {
                    self.expect_punct(Punct::RBracket, "']'");
                    name.push(']');
                }
            }
            _ => self.error_expected("overloadable operator"),
        }
        name
    }

    fn parse_param_list(&mut self, out: &mut Vec<Item>) -> (Vec<ParamDecl>, bool) {
        let mut params = Vec::new();
        if self.check_punct(Punct::RParen) {
            self.bump();
            return (params, false);
        }
        if matches!(self.peek_keyword(), Some(Keyword::Void))
            && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::Punct(Punct::RParen))
            )
        {
            self.bump();
            self.bump();
            return (params, false);
        }
        let mut variadic = false;
        loop {
            if self.eat_punct(Punct::Ellipsis) {
                variadic = true;
                break;
            }
            match self.parse_decl_specifiers(out) {
                Some((base, _)) => {
                    let (ty, name, span) = self.parse_declarator(base);
                    params.push(ParamDecl { ty, name, span });
                }
                None => {
                    self.synchronize_param();
                    if self.check_punct(Punct::RParen) || self.at_eof() {
                        break;
                    }
                }
            }
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RParen, "')'");
        (params, variadic)
    }

    /// Skips to the next `,` or `)` at depth 0, for recovering from a malformed parameter.
    fn synchronize_param(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Eof => return,
                TokenKind::Punct(Punct::LParen | Punct::LBracket | Punct::LBrace) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RParen | Punct::RBracket | Punct::RBrace) => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::Comma) if depth == 0 => return,
                _ => {
                    self.bump();
                }
            }
        }
    }

    // ---- statements --------------------------------------------------------------------

    /// True if the decl-specifier sequence starting here is unambiguous without a symbol
    /// table: a scalar keyword, a qualifier, or a tag keyword. A bare identifier is never
    /// treated as a declaration start, even if it names a typedef — this layer has no symbol
    /// table to know that, so `Foo bar;` where `Foo` is a typedef parses as an (ill-formed,
    /// but not hung-on) expression-statement instead. Same documented gap as
    /// `next_starts_type`, applied at statement granularity.
    fn stmt_starts_decl(&self) -> bool {
        match self.peek_keyword() {
            Some(k) if is_scalar_keyword(k) => true,
            Some(Keyword::Const | Keyword::Volatile) => true,
            Some(Keyword::Struct | Keyword::Union | Keyword::Enum) => true,
            _ => false,
        }
    }

    fn stmt_is_label(&self) -> bool {
        matches!(self.cur().kind, TokenKind::Ident(_))
            && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::Punct(Punct::Colon))
            )
    }

    /// Skips tokens, tracking brace nesting, until a `;` at depth 0 (consumed) or a `}` that
    /// closes the enclosing block (left unconsumed), or `Eof` — the statement-level analogue
    /// of `synchronize`/`synchronize_field`/`synchronize_param`, so one malformed statement
    /// can't take the rest of the block (or the parse) down with it.
    fn synchronize_stmt(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Eof => return,
                TokenKind::Punct(Punct::LBrace) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RBrace) => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::Semi) if depth == 0 => {
                    self.bump();
                    return;
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Parses the `{ ... }` statement list of a block or function body. `self.pos` must be at
    /// the opening `{`. Returns the parsed statements and the span of the whole `{ ... }`.
    fn parse_block_stmts(&mut self) -> (Vec<Stmt>, Span) {
        let start = self.cur().span.start;
        self.bump(); // `{`
        let mut stmts = Vec::new();
        while !self.check_punct(Punct::RBrace) && !self.at_eof() {
            let before_pos = self.pos;
            stmts.push(self.parse_stmt());
            if self.pos == before_pos {
                // Safety net: guarantee progress even if a future edit adds a path in
                // `parse_stmt` that doesn't consume or resynchronize.
                self.bump();
            }
        }
        let end = if self.check_punct(Punct::RBrace) {
            self.bump().span.end
        } else {
            self.error_expected("'}'");
            self.prev_span_end().end
        };
        (stmts, Span::new(start, end))
    }

    fn parse_block(&mut self) -> Stmt {
        let (stmts, span) = self.parse_block_stmts();
        Stmt::Block { stmts, span }
    }

    /// Parses one or more declarators sharing a decl-specifier sequence as a statement,
    /// consuming the trailing `;`. Shares `parse_decl_specifiers`/`parse_declarator` with the
    /// top-level declaration path in `parse_decl_or_def`; a tag body defined inline here (e.g.
    /// `struct { int x; } p;` as a local declaration) is parsed structurally but its `Item` is
    /// discarded rather than retained in the `Stmt` tree — local tag definitions are out of
    /// this stage's scope (expressions/statements/control-flow), consistent with the crate's
    /// documented gaps elsewhere.
    fn parse_decl_stmt(&mut self) -> Stmt {
        let start = self.cur().span.start;
        let mut scratch = Vec::new();
        let base = match self.parse_decl_specifiers(&mut scratch) {
            Some((ty, _)) => ty,
            None => {
                self.synchronize_stmt();
                return Stmt::Empty {
                    span: Span::new(start, self.prev_span_end().end),
                };
            }
        };
        if self.eat_punct(Punct::Semi) {
            return Stmt::Decl {
                decls: Vec::new(),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        let mut decls = Vec::new();
        loop {
            let (ty, name, dspan) = self.parse_declarator(base.clone());
            let init = if self.eat_punct(Punct::Eq) {
                Some(self.parse_assign())
            } else {
                None
            };
            match name {
                Some(n) => decls.push(VarDecl {
                    ty,
                    name: n,
                    init,
                    span: dspan,
                }),
                None => self
                    .errors
                    .push(ParseError::MissingDeclaratorName { span: dspan }),
            }
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        if self.expect_punct(Punct::Semi, "';'").is_none() {
            self.synchronize_stmt();
        }
        Stmt::Decl {
            decls,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_expr_stmt(&mut self) -> Stmt {
        let start = self.cur().span.start;
        let expr = self.parse_expr();
        if self.expect_punct(Punct::Semi, "';'").is_none() {
            self.synchronize_stmt();
        }
        Stmt::Expr {
            expr,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_if(&mut self) -> Stmt {
        let start = self.bump().span.start; // `if`
        self.expect_punct(Punct::LParen, "'('");
        let cond = self.parse_expr();
        self.expect_punct(Punct::RParen, "')'");
        let then_branch = Box::new(self.parse_stmt());
        let else_branch = if self.eat_keyword(Keyword::Else) {
            Some(Box::new(self.parse_stmt()))
        } else {
            None
        };
        Stmt::If {
            cond,
            then_branch,
            else_branch,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_while(&mut self) -> Stmt {
        let start = self.bump().span.start; // `while`
        self.expect_punct(Punct::LParen, "'('");
        let cond = self.parse_expr();
        self.expect_punct(Punct::RParen, "')'");
        let body = Box::new(self.parse_stmt());
        Stmt::While {
            cond,
            body,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_do_while(&mut self) -> Stmt {
        let start = self.bump().span.start; // `do`
        let body = Box::new(self.parse_stmt());
        if !self.eat_keyword(Keyword::While) {
            self.error_expected("'while'");
        }
        self.expect_punct(Punct::LParen, "'('");
        let cond = self.parse_expr();
        self.expect_punct(Punct::RParen, "')'");
        self.expect_punct(Punct::Semi, "';'");
        Stmt::DoWhile {
            body,
            cond,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_for(&mut self) -> Stmt {
        let start = self.bump().span.start; // `for`
        self.expect_punct(Punct::LParen, "'('");
        let init = if self.eat_punct(Punct::Semi) {
            None
        } else if self.stmt_starts_decl() {
            Some(Box::new(self.parse_decl_stmt()))
        } else {
            Some(Box::new(self.parse_expr_stmt()))
        };
        let cond = if self.check_punct(Punct::Semi) {
            None
        } else {
            Some(self.parse_expr())
        };
        self.expect_punct(Punct::Semi, "';'");
        let step = if self.check_punct(Punct::RParen) {
            None
        } else {
            Some(self.parse_expr())
        };
        self.expect_punct(Punct::RParen, "')'");
        let body = Box::new(self.parse_stmt());
        Stmt::For {
            init,
            cond,
            step,
            body,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_switch(&mut self) -> Stmt {
        let start = self.bump().span.start; // `switch`
        self.expect_punct(Punct::LParen, "'('");
        let expr = self.parse_expr();
        self.expect_punct(Punct::RParen, "')'");
        let body = Box::new(self.parse_stmt());
        Stmt::Switch {
            expr,
            body,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_case(&mut self) -> Stmt {
        let start = self.bump().span.start; // `case`
        let value = self.parse_conditional();
        self.expect_punct(Punct::Colon, "':'");
        let stmt = Box::new(self.parse_stmt());
        Stmt::Case {
            value,
            stmt,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_default(&mut self) -> Stmt {
        let start = self.bump().span.start; // `default`
        self.expect_punct(Punct::Colon, "':'");
        let stmt = Box::new(self.parse_stmt());
        Stmt::Default {
            stmt,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_label(&mut self) -> Stmt {
        let start = self.cur().span.start;
        let name = self.ident_here().unwrap_or_default();
        self.bump(); // name
        self.bump(); // `:`
        let stmt = Box::new(self.parse_stmt());
        Stmt::Label {
            name,
            stmt,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_stmt(&mut self) -> Stmt {
        match self.cur().kind {
            TokenKind::Punct(Punct::LBrace) => self.parse_block(),
            TokenKind::Punct(Punct::Semi) => {
                let span = self.bump().span;
                Stmt::Empty { span }
            }
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::Keyword(Keyword::While) => self.parse_while(),
            TokenKind::Keyword(Keyword::Do) => self.parse_do_while(),
            TokenKind::Keyword(Keyword::For) => self.parse_for(),
            TokenKind::Keyword(Keyword::Switch) => self.parse_switch(),
            TokenKind::Keyword(Keyword::Case) => self.parse_case(),
            TokenKind::Keyword(Keyword::Default) => self.parse_default(),
            TokenKind::Keyword(Keyword::Break) => {
                let start = self.bump().span.start;
                self.expect_punct(Punct::Semi, "';'");
                Stmt::Break {
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Keyword(Keyword::Continue) => {
                let start = self.bump().span.start;
                self.expect_punct(Punct::Semi, "';'");
                Stmt::Continue {
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Keyword(Keyword::Return) => {
                let start = self.bump().span.start;
                let expr = if self.check_punct(Punct::Semi) {
                    None
                } else {
                    Some(self.parse_expr())
                };
                self.expect_punct(Punct::Semi, "';'");
                Stmt::Return {
                    expr,
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Keyword(Keyword::Goto) => {
                let start = self.bump().span.start;
                let label = match self.ident_here() {
                    Some(n) => {
                        self.bump();
                        n
                    }
                    None => {
                        self.error_expected("label name");
                        String::new()
                    }
                };
                self.expect_punct(Punct::Semi, "';'");
                Stmt::Goto {
                    label,
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Punct(Punct::RBrace) => {
                // Let the caller (parse_block_stmts) see this rather than error on it here.
                let span = self.cur().span;
                self.errors.push(ParseError::UnmatchedBrace { span });
                Stmt::Empty { span }
            }
            _ if self.stmt_is_label() => self.parse_label(),
            _ if self.stmt_starts_decl() => self.parse_decl_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    // ---- expressions --------------------------------------------------------------------

    /// True if the token right after the `(` at `self.pos` unambiguously starts a type: a
    /// scalar/qualifier/tag keyword. A bare identifier is deliberately never treated as a type
    /// here — this layer has no symbol table, so it can't tell a typedef name from an ordinary
    /// variable — meaning `(Foo)x` where `Foo` is a typedef parses as `Foo` grouped in
    /// parentheses used as the left side of... nothing (the token after the `)` decides what
    /// happens next), not as a cast. This is the parser's one documented ambiguity gap for
    /// cast-expressions; a later stage with a symbol table can re-disambiguate if needed.
    fn next_starts_type(&self) -> bool {
        match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
            Some(TokenKind::Keyword(k)) if is_scalar_keyword(*k) => true,
            Some(TokenKind::Keyword(Keyword::Const | Keyword::Volatile)) => true,
            Some(TokenKind::Keyword(Keyword::Struct | Keyword::Union | Keyword::Enum)) => true,
            _ => false,
        }
    }

    /// Parses a type-name (decl-specifiers plus an optional abstract declarator, no name) for
    /// a cast or `sizeof(Type)`. A tag body defined inline here (rare, and not valid C in the
    /// first place for a cast) is parsed structurally but its `Item` is discarded, same as
    /// `parse_decl_stmt`'s scratch vec. A declarator name that shows up anyway (malformed
    /// input, e.g. `(int x)`) is silently ignored rather than erroring twice.
    fn parse_cast_type(&mut self) -> Type {
        let mut scratch = Vec::new();
        let base = match self.parse_decl_specifiers(&mut scratch) {
            Some((ty, _)) => ty,
            None => {
                return Type::Scalar {
                    kind: ScalarKind::Int,
                    quals: Qualifiers::default(),
                    span: self.cur().span,
                }
            }
        };
        let (ty, _name, _span) = self.parse_declarator(base);
        ty
    }

    /// Comma expression: `a, b, c` (left-associative). This is the top of the expression
    /// grammar; every narrower context (call arguments, initializers, array sizes, ...) parses
    /// at a tighter level instead so a bare `,` there means what it looks like it means.
    fn parse_expr(&mut self) -> Expr {
        let start = self.cur().span.start;
        let first = self.parse_assign();
        if !self.check_punct(Punct::Comma) {
            return first;
        }
        let mut exprs = vec![first];
        while self.eat_punct(Punct::Comma) {
            exprs.push(self.parse_assign());
        }
        Expr::Comma {
            exprs,
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn peek_assign_op(&self) -> Option<AssignOp> {
        match self.cur().kind {
            TokenKind::Punct(Punct::Eq) => Some(AssignOp::Assign),
            TokenKind::Punct(Punct::PlusEq) => Some(AssignOp::AddAssign),
            TokenKind::Punct(Punct::MinusEq) => Some(AssignOp::SubAssign),
            TokenKind::Punct(Punct::StarEq) => Some(AssignOp::MulAssign),
            TokenKind::Punct(Punct::SlashEq) => Some(AssignOp::DivAssign),
            TokenKind::Punct(Punct::PercentEq) => Some(AssignOp::RemAssign),
            TokenKind::Punct(Punct::AmpEq) => Some(AssignOp::AndAssign),
            TokenKind::Punct(Punct::PipeEq) => Some(AssignOp::OrAssign),
            TokenKind::Punct(Punct::CaretEq) => Some(AssignOp::XorAssign),
            TokenKind::Punct(Punct::ShlEq) => Some(AssignOp::ShlAssign),
            TokenKind::Punct(Punct::ShrEq) => Some(AssignOp::ShrAssign),
            _ => None,
        }
    }

    /// Assignment-expression: `lhs = rhs` and its compound forms, right-associative. The
    /// grammar restricts `lhs` to a unary-expression; this parser accepts any
    /// conditional-expression there instead and leaves rejecting a non-lvalue lhs (`a + b = c`)
    /// to a later stage, which matches the rest of this crate's "structurally permissive, defer
    /// validation" style (e.g. `resolve_scalar`).
    fn parse_assign(&mut self) -> Expr {
        let start = self.cur().span.start;
        let lhs = self.parse_conditional();
        match self.peek_assign_op() {
            Some(op) => {
                self.bump();
                let rhs = self.parse_assign();
                Expr::Assign {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            None => lhs,
        }
    }

    /// Conditional (ternary) expression: `cond ? then : else`, right-associative. Per the C
    /// grammar the middle branch is a full expression (comma allowed) and the else-branch
    /// recurses back into this same level, which is what makes `a ? b : c ? d : e` group as
    /// `a ? b : (c ? d : e)`.
    fn parse_conditional(&mut self) -> Expr {
        let start = self.cur().span.start;
        let cond = self.parse_logor();
        if !self.eat_punct(Punct::Question) {
            return cond;
        }
        let then_branch = self.parse_expr();
        self.expect_punct(Punct::Colon, "':'");
        let else_branch = self.parse_conditional();
        Expr::Ternary {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
            span: Span::new(start, self.prev_span_end().end),
        }
    }

    fn parse_logor(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_logand();
        while self.eat_punct(Punct::PipePipe) {
            let rhs = self.parse_logand();
            lhs = Expr::Binary {
                op: BinOp::LogOr,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_logand(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_bitor();
        while self.eat_punct(Punct::AmpAmp) {
            let rhs = self.parse_bitor();
            lhs = Expr::Binary {
                op: BinOp::LogAnd,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_bitor(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_bitxor();
        while self.eat_punct(Punct::Pipe) {
            let rhs = self.parse_bitxor();
            lhs = Expr::Binary {
                op: BinOp::BitOr,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_bitxor(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_bitand();
        while self.eat_punct(Punct::Caret) {
            let rhs = self.parse_bitand();
            lhs = Expr::Binary {
                op: BinOp::BitXor,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_bitand(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_equality();
        while self.eat_punct(Punct::Amp) {
            let rhs = self.parse_equality();
            lhs = Expr::Binary {
                op: BinOp::BitAnd,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_equality(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_relational();
        loop {
            let op = if self.check_punct(Punct::EqEq) {
                BinOp::Eq
            } else if self.check_punct(Punct::NotEq) {
                BinOp::Ne
            } else {
                break;
            };
            self.bump();
            let rhs = self.parse_relational();
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_relational(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_shift();
        loop {
            let op = if self.check_punct(Punct::Lt) {
                BinOp::Lt
            } else if self.check_punct(Punct::Gt) {
                BinOp::Gt
            } else if self.check_punct(Punct::Le) {
                BinOp::Le
            } else if self.check_punct(Punct::Ge) {
                BinOp::Ge
            } else {
                break;
            };
            self.bump();
            let rhs = self.parse_shift();
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_shift(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_additive();
        loop {
            let op = if self.check_punct(Punct::Shl) {
                BinOp::Shl
            } else if self.check_punct(Punct::Shr) {
                BinOp::Shr
            } else {
                break;
            };
            self.bump();
            let rhs = self.parse_additive();
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_additive(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_multiplicative();
        loop {
            let op = if self.check_punct(Punct::Plus) {
                BinOp::Add
            } else if self.check_punct(Punct::Minus) {
                BinOp::Sub
            } else {
                break;
            };
            self.bump();
            let rhs = self.parse_multiplicative();
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    fn parse_multiplicative(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut lhs = self.parse_cast();
        loop {
            let op = if self.check_punct(Punct::Star) {
                BinOp::Mul
            } else if self.check_punct(Punct::Slash) {
                BinOp::Div
            } else if self.check_punct(Punct::Percent) {
                BinOp::Rem
            } else {
                break;
            };
            self.bump();
            let rhs = self.parse_cast();
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        lhs
    }

    /// Cast-expression: `(Type)expr` or a plain unary-expression. See `next_starts_type` for
    /// the disambiguation heuristic.
    fn parse_cast(&mut self) -> Expr {
        if self.check_punct(Punct::LParen) && self.next_starts_type() {
            let start = self.cur().span.start;
            self.bump(); // `(`
            let ty = self.parse_cast_type();
            self.expect_punct(Punct::RParen, "')'");
            let expr = self.parse_cast();
            return Expr::Cast {
                ty,
                expr: Box::new(expr),
                span: Span::new(start, self.prev_span_end().end),
            };
        }
        self.parse_unary()
    }

    fn parse_unary(&mut self) -> Expr {
        let start = self.cur().span.start;
        match self.cur().kind {
            TokenKind::Punct(Punct::PlusPlus) => {
                self.bump();
                let expr = Box::new(self.parse_unary());
                Expr::PreIncDec {
                    op: IncDecOp::Inc,
                    expr,
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Punct(Punct::MinusMinus) => {
                self.bump();
                let expr = Box::new(self.parse_unary());
                Expr::PreIncDec {
                    op: IncDecOp::Dec,
                    expr,
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Punct(
                p @ (Punct::Plus
                | Punct::Minus
                | Punct::Bang
                | Punct::Tilde
                | Punct::Star
                | Punct::Amp),
            ) => {
                self.bump();
                let op = match p {
                    Punct::Plus => UnaryOp::Plus,
                    Punct::Minus => UnaryOp::Neg,
                    Punct::Bang => UnaryOp::Not,
                    Punct::Tilde => UnaryOp::BitNot,
                    Punct::Star => UnaryOp::Deref,
                    Punct::Amp => UnaryOp::Addr,
                    _ => unreachable!(),
                };
                let expr = Box::new(self.parse_cast());
                Expr::Unary {
                    op,
                    expr,
                    span: Span::new(start, self.prev_span_end().end),
                }
            }
            TokenKind::Keyword(Keyword::Sizeof) => {
                self.bump();
                if self.check_punct(Punct::LParen) && self.next_starts_type() {
                    self.bump(); // `(`
                    let ty = self.parse_cast_type();
                    self.expect_punct(Punct::RParen, "')'");
                    Expr::SizeofType {
                        ty,
                        span: Span::new(start, self.prev_span_end().end),
                    }
                } else {
                    let expr = Box::new(self.parse_unary());
                    Expr::SizeofExpr {
                        expr,
                        span: Span::new(start, self.prev_span_end().end),
                    }
                }
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Expr {
        let start = self.cur().span.start;
        let mut e = self.parse_primary();
        loop {
            match self.cur().kind {
                TokenKind::Punct(Punct::LParen) => {
                    self.bump();
                    let args = self.parse_arg_list();
                    e = Expr::Call {
                        callee: Box::new(e),
                        args,
                        span: Span::new(start, self.prev_span_end().end),
                    };
                }
                TokenKind::Punct(Punct::LBracket) => {
                    self.bump();
                    let index = self.parse_expr();
                    self.expect_punct(Punct::RBracket, "']'");
                    e = Expr::Index {
                        base: Box::new(e),
                        index: Box::new(index),
                        span: Span::new(start, self.prev_span_end().end),
                    };
                }
                TokenKind::Punct(Punct::Dot) | TokenKind::Punct(Punct::Arrow) => {
                    let arrow = self.check_punct(Punct::Arrow);
                    self.bump();
                    let name = match self.ident_here() {
                        Some(n) => {
                            self.bump();
                            n
                        }
                        None => {
                            self.error_expected("member name");
                            String::new()
                        }
                    };
                    e = Expr::Member {
                        base: Box::new(e),
                        name,
                        arrow,
                        span: Span::new(start, self.prev_span_end().end),
                    };
                }
                TokenKind::Punct(Punct::PlusPlus) => {
                    self.bump();
                    e = Expr::PostIncDec {
                        op: IncDecOp::Inc,
                        expr: Box::new(e),
                        span: Span::new(start, self.prev_span_end().end),
                    };
                }
                TokenKind::Punct(Punct::MinusMinus) => {
                    self.bump();
                    e = Expr::PostIncDec {
                        op: IncDecOp::Dec,
                        expr: Box::new(e),
                        span: Span::new(start, self.prev_span_end().end),
                    };
                }
                _ => break,
            }
        }
        e
    }

    /// Comma-separated call arguments, each at assignment-expression level (a bare `,` inside
    /// an argument list separates arguments, not a comma-expression; write `f((a, b))` for
    /// that).
    fn parse_arg_list(&mut self) -> Vec<Expr> {
        let mut args = Vec::new();
        if self.check_punct(Punct::RParen) {
            self.bump();
            return args;
        }
        loop {
            args.push(self.parse_assign());
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RParen, "')'");
        args
    }

    fn parse_primary(&mut self) -> Expr {
        let span = self.cur().span;
        match self.cur().kind.clone() {
            TokenKind::IntLit(value) => {
                self.bump();
                Expr::IntLit { value, span }
            }
            TokenKind::FloatLit(value) => {
                self.bump();
                Expr::FloatLit { value, span }
            }
            TokenKind::CharLit(value) => {
                self.bump();
                Expr::CharLit { value, span }
            }
            TokenKind::StrLit(value) => {
                self.bump();
                Expr::StrLit { value, span }
            }
            TokenKind::Ident(name) => {
                self.bump();
                Expr::Ident { name, span }
            }
            TokenKind::Punct(Punct::LParen) => {
                self.bump();
                let e = self.parse_expr();
                self.expect_punct(Punct::RParen, "')'");
                e
            }
            _ => {
                self.error_expected("expression");
                Expr::Error { span }
            }
        }
    }
}

fn is_scalar_keyword(k: Keyword) -> bool {
    matches!(
        k,
        Keyword::Void
            | Keyword::Bool
            | Keyword::Char
            | Keyword::Short
            | Keyword::Int
            | Keyword::Long
            | Keyword::Float
            | Keyword::Double
            | Keyword::Signed
            | Keyword::Unsigned
            | Keyword::WcharT
    )
}

fn apply_quals(ty: &mut Type, quals: Qualifiers) {
    let target = match ty {
        Type::Scalar { quals, .. }
        | Type::Tag { quals, .. }
        | Type::Named { quals, .. }
        | Type::Pointer { quals, .. } => quals,
        Type::Array { .. } => return,
    };
    target.is_const |= quals.is_const;
    target.is_volatile |= quals.is_volatile;
}

#[derive(Default)]
struct ScalarCounts {
    void: u32,
    bool_: u32,
    char_: u32,
    short: u32,
    int: u32,
    long: u32,
    float: u32,
    double: u32,
    signed: u32,
    unsigned: u32,
    wchar: u32,
}

/// Resolves a multiset of scalar specifier keywords into one `ScalarKind`, matching the
/// combinations C actually allows (`unsigned long long int`, `signed char`, `long double`,
/// ...). Validation is best-effort, not an exhaustive check of every invalid combination C
/// forbids: nonsensical input degrades to a plausible kind (usually `Int`) plus a recorded
/// error, rather than failing to produce a type at all.
fn resolve_scalar(c: &ScalarCounts, span: Span, errors: &mut Vec<ParseError>) -> ScalarKind {
    let total = c.void
        + c.bool_
        + c.char_
        + c.short
        + c.int
        + c.long
        + c.float
        + c.double
        + c.signed
        + c.unsigned
        + c.wchar;
    let bad = |detail: &str, errors: &mut Vec<ParseError>| {
        errors.push(ParseError::InvalidTypeSpec {
            detail: detail.to_string(),
            span,
        });
    };
    if total == 0 {
        bad("missing type specifier", errors);
        return ScalarKind::Int;
    }
    if c.void > 0 {
        if total > 1 {
            bad("'void' combined with other type specifiers", errors);
        }
        return ScalarKind::Void;
    }
    if c.bool_ > 0 {
        if total > 1 {
            bad("'bool' combined with other type specifiers", errors);
        }
        return ScalarKind::Bool;
    }
    if c.wchar > 0 {
        if total > 1 {
            bad("'wchar_t' combined with other type specifiers", errors);
        }
        return ScalarKind::WcharT;
    }
    if c.signed > 0 && c.unsigned > 0 {
        bad("'signed' and 'unsigned' together", errors);
    }
    if c.float > 0 {
        if total > 1 {
            bad("'float' combined with other type specifiers", errors);
        }
        return ScalarKind::Float;
    }
    if c.double > 0 {
        if c.long == 1 && total == 2 {
            return ScalarKind::LongDouble;
        }
        if total > 1 {
            bad("'double' combined with other type specifiers", errors);
        }
        return ScalarKind::Double;
    }
    if c.char_ > 0 {
        if c.short > 0 || c.long > 0 {
            bad("'char' combined with 'short'/'long'", errors);
        }
        if c.signed > 0 {
            return ScalarKind::SChar;
        }
        if c.unsigned > 0 {
            return ScalarKind::UChar;
        }
        return ScalarKind::Char;
    }
    if c.short > 0 {
        if c.long > 0 {
            bad("'short' combined with 'long'", errors);
        }
        return if c.unsigned > 0 {
            ScalarKind::UShort
        } else {
            ScalarKind::Short
        };
    }
    if c.long >= 2 {
        return if c.unsigned > 0 {
            ScalarKind::ULongLong
        } else {
            ScalarKind::LongLong
        };
    }
    if c.long == 1 {
        return if c.unsigned > 0 {
            ScalarKind::ULong
        } else {
            ScalarKind::Long
        };
    }
    if c.unsigned > 0 {
        return ScalarKind::UInt;
    }
    ScalarKind::Int
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::lex;

    fn parse_ok(src: &str) -> TranslationUnit {
        let (tokens, lex_errs) = lex(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        let (tu, errs) = parse(&tokens);
        assert!(errs.is_empty(), "parse errors: {errs:?}");
        tu
    }

    #[test]
    fn struct_with_varied_field_types() {
        let tu = parse_ok(
            r#"
            struct Point {
                int x;
                float y;
                char *name;
                int coords[3];
            };
            "#,
        );
        assert_eq!(tu.items.len(), 1);
        let Item::Struct(s) = &tu.items[0] else {
            panic!("expected struct, got {:?}", tu.items[0]);
        };
        assert_eq!(s.name.as_deref(), Some("Point"));
        assert_eq!(s.fields.len(), 4);
        assert_eq!(s.fields[0].name, "x");
        assert!(matches!(
            s.fields[0].ty,
            Type::Scalar {
                kind: ScalarKind::Int,
                ..
            }
        ));
        assert_eq!(s.fields[1].name, "y");
        assert!(matches!(
            s.fields[1].ty,
            Type::Scalar {
                kind: ScalarKind::Float,
                ..
            }
        ));
        assert_eq!(s.fields[2].name, "name");
        match &s.fields[2].ty {
            Type::Pointer { pointee, .. } => {
                assert!(matches!(
                    **pointee,
                    Type::Scalar {
                        kind: ScalarKind::Char,
                        ..
                    }
                ));
            }
            other => panic!("expected pointer, got {other:?}"),
        }
        assert_eq!(s.fields[3].name, "coords");
        match &s.fields[3].ty {
            Type::Array { elem, size, .. } => {
                assert!(matches!(
                    **elem,
                    Type::Scalar {
                        kind: ScalarKind::Int,
                        ..
                    }
                ));
                assert!(size.is_some());
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn nested_struct_in_namespace() {
        let tu = parse_ok(
            r#"
            namespace gpu {
                struct Vec3 {
                    float x;
                    float y;
                    float z;
                };
            }
            "#,
        );
        assert_eq!(tu.items.len(), 1);
        let Item::Namespace(ns) = &tu.items[0] else {
            panic!("expected namespace");
        };
        assert_eq!(ns.name, "gpu");
        assert_eq!(ns.items.len(), 1);
        let Item::Struct(s) = &ns.items[0] else {
            panic!("expected nested struct");
        };
        assert_eq!(s.name.as_deref(), Some("Vec3"));
        assert_eq!(s.fields.len(), 3);
    }

    #[test]
    fn enum_with_and_without_initializers() {
        let tu = parse_ok(
            r#"
            enum Color { Red, Green = 5, Blue };
            "#,
        );
        let Item::Enum(e) = &tu.items[0] else {
            panic!("expected enum");
        };
        assert_eq!(e.name.as_deref(), Some("Color"));
        assert_eq!(e.variants.len(), 3);
        assert_eq!(e.variants[0].name, "Red");
        assert!(e.variants[0].init.is_none());
        assert_eq!(e.variants[1].name, "Green");
        assert!(e.variants[1].init.is_some());
        assert_eq!(e.variants[2].name, "Blue");
        assert!(e.variants[2].init.is_none());
    }

    #[test]
    fn typedef_decl() {
        let tu = parse_ok("typedef unsigned long my_ulong;");
        let Item::Typedef(t) = &tu.items[0] else {
            panic!("expected typedef");
        };
        assert_eq!(t.alias, "my_ulong");
        assert!(matches!(
            t.ty,
            Type::Scalar {
                kind: ScalarKind::ULong,
                ..
            }
        ));
    }

    #[test]
    fn function_prototype_with_pointer_params() {
        let tu = parse_ok("void kernel(float *a, const float *b, int n);");
        let Item::Function(f) = &tu.items[0] else {
            panic!("expected function");
        };
        assert_eq!(f.name, "kernel");
        assert!(f.body.is_none());
        assert!(!f.variadic);
        assert_eq!(f.params.len(), 3);
        assert_eq!(f.params[0].name.as_deref(), Some("a"));
        match &f.params[0].ty {
            Type::Pointer { pointee, .. } => {
                assert!(matches!(
                    **pointee,
                    Type::Scalar {
                        kind: ScalarKind::Float,
                        ..
                    }
                ))
            }
            other => panic!("expected pointer, got {other:?}"),
        }
        assert_eq!(f.params[1].name.as_deref(), Some("b"));
        match &f.params[1].ty {
            Type::Pointer { pointee, .. } => match &**pointee {
                Type::Scalar {
                    kind: ScalarKind::Float,
                    quals,
                    ..
                } => assert!(quals.is_const),
                other => panic!("expected const float, got {other:?}"),
            },
            other => panic!("expected pointer, got {other:?}"),
        }
        assert_eq!(f.params[2].name.as_deref(), Some("n"));
    }

    #[test]
    fn function_with_body_captures_nested_braces() {
        let src = r#"
        int add(int a, int b) {
            if (a > 0) {
                return a + b;
            }
            return b;
        }
        "#;
        let (tokens, lex_errs) = lex(src);
        assert!(lex_errs.is_empty());
        let (tu, errs) = parse(&tokens);
        assert!(errs.is_empty(), "{errs:?}");
        let Item::Function(f) = &tu.items[0] else {
            panic!("expected function");
        };
        assert_eq!(f.name, "add");
        let body = f.body.as_ref().expect("expected body");
        assert_eq!(body.len(), 2);
        assert!(matches!(body[0], Stmt::If { .. }));
        assert!(matches!(body[1], Stmt::Return { expr: Some(_), .. }));
        let Stmt::If {
            then_branch, cond, ..
        } = &body[0]
        else {
            panic!();
        };
        assert!(matches!(cond, Expr::Binary { op: BinOp::Gt, .. }));
        let Stmt::Block { stmts, .. } = then_branch.as_ref() else {
            panic!("expected block then-branch");
        };
        assert_eq!(stmts.len(), 1);
        assert!(matches!(stmts[0], Stmt::Return { expr: Some(_), .. }));
    }

    #[test]
    fn combined_keyword_scalar_types() {
        let tu = parse_ok("unsigned long long int a; const int *b; int * const c;");
        let Item::Var(a) = &tu.items[0] else { panic!() };
        assert!(matches!(
            a.ty,
            Type::Scalar {
                kind: ScalarKind::ULongLong,
                ..
            }
        ));

        let Item::Var(b) = &tu.items[1] else { panic!() };
        match &b.ty {
            Type::Pointer { pointee, quals, .. } => {
                assert!(!quals.is_const);
                assert!(
                    matches!(**pointee, Type::Scalar { kind: ScalarKind::Int, quals, .. } if quals.is_const)
                );
            }
            other => panic!("{other:?}"),
        }

        let Item::Var(c) = &tu.items[2] else { panic!() };
        match &c.ty {
            Type::Pointer { pointee, quals, .. } => {
                assert!(quals.is_const);
                assert!(
                    matches!(**pointee, Type::Scalar { kind: ScalarKind::Int, quals, .. } if !quals.is_const)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn namespace_with_multiple_nested_items() {
        let tu = parse_ok(
            r#"
            namespace ops {
                struct Handle { int id; };
                typedef int index_t;
                void run(int n);
            }
            "#,
        );
        let Item::Namespace(ns) = &tu.items[0] else {
            panic!("expected namespace");
        };
        assert_eq!(ns.items.len(), 3);
        assert!(matches!(ns.items[0], Item::Struct(_)));
        assert!(matches!(ns.items[1], Item::Typedef(_)));
        assert!(matches!(ns.items[2], Item::Function(_)));
    }

    #[test]
    fn template_decl_recognized_structurally() {
        let tu = parse_ok(
            r#"
            template<typename T>
            struct Box {
                T value;
            };
            "#,
        );
        assert_eq!(tu.items.len(), 1);
        let Item::Template(t) = &tu.items[0] else {
            panic!("expected template, got {:?}", tu.items[0]);
        };
        assert_eq!(t.params.len(), 1);
        assert_eq!(t.params[0].name, "T");
        assert_eq!(t.name.as_deref(), Some("Box"));
        assert!(t.body.start < t.body.end);
    }

    #[test]
    fn malformed_declaration_recovers_and_continues() {
        let (tokens, lex_errs) = lex(r#"
            int &&& broken;
            int good;
            "#);
        assert!(lex_errs.is_empty());
        let (tu, errs) = parse(&tokens);
        assert!(!errs.is_empty(), "expected at least one parse error");
        // the well-formed declaration after the broken one must still show up
        let found_good = tu
            .items
            .iter()
            .any(|i| matches!(i, Item::Var(v) if v.name == "good"));
        assert!(found_good, "items: {tu:?}");
    }

    #[test]
    fn does_not_hang_on_unterminated_input() {
        let (tokens, _) = lex("struct Foo { int x");
        let (_tu, errs) = parse(&tokens);
        assert!(!errs.is_empty());
    }

    // ---- expressions ---------------------------------------------------------------------

    /// Parses `void f() { EXPR; }` and returns the parsed expression of that one statement.
    fn parse_expr_src(expr_src: &str) -> Expr {
        let src = format!("void f() {{ {expr_src}; }}");
        let tu = parse_ok(&src);
        let body = first_fn_body(&tu);
        let Stmt::Expr { expr, .. } = &body[0] else {
            panic!("expected expression-statement, got {:?}", body[0]);
        };
        expr.clone()
    }

    fn first_fn_body(tu: &TranslationUnit) -> &[Stmt] {
        let Item::Function(f) = &tu.items[0] else {
            panic!("expected function, got {:?}", tu.items[0]);
        };
        f.body.as_ref().expect("expected body")
    }

    #[test]
    fn expr_multiplicative_binds_tighter_than_additive() {
        let e = parse_expr_src("a + b * c");
        let Expr::Binary {
            op: BinOp::Add,
            lhs,
            rhs,
            ..
        } = &e
        else {
            panic!("{e:?}");
        };
        assert!(matches!(&**lhs, Expr::Ident{name, ..} if name == "a"));
        assert!(matches!(&**rhs, Expr::Binary { op: BinOp::Mul, .. }));
    }

    #[test]
    fn expr_assignment_is_right_associative() {
        let e = parse_expr_src("a = b = c");
        let Expr::Assign { lhs, rhs, .. } = &e else {
            panic!("{e:?}");
        };
        assert!(matches!(&**lhs, Expr::Ident{name, ..} if name == "a"));
        assert!(matches!(&**rhs, Expr::Assign { .. }));
    }

    #[test]
    fn expr_ternary_is_right_associative() {
        let e = parse_expr_src("a ? b : c ? d : e");
        let Expr::Ternary { else_branch, .. } = &e else {
            panic!("{e:?}");
        };
        assert!(matches!(&**else_branch, Expr::Ternary { .. }));
    }

    #[test]
    fn expr_comma_expression_collects_all_operands() {
        let e = parse_expr_src("a, b, c");
        let Expr::Comma { exprs, .. } = &e else {
            panic!("{e:?}");
        };
        assert_eq!(exprs.len(), 3);
    }

    #[test]
    fn expr_unary_prefix_and_postfix_operators() {
        assert!(matches!(
            parse_expr_src("-a"),
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("!a"),
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("~a"),
            Expr::Unary {
                op: UnaryOp::BitNot,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("*a"),
            Expr::Unary {
                op: UnaryOp::Deref,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("&a"),
            Expr::Unary {
                op: UnaryOp::Addr,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("++a"),
            Expr::PreIncDec {
                op: IncDecOp::Inc,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("--a"),
            Expr::PreIncDec {
                op: IncDecOp::Dec,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("a++"),
            Expr::PostIncDec {
                op: IncDecOp::Inc,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_src("a--"),
            Expr::PostIncDec {
                op: IncDecOp::Dec,
                ..
            }
        ));
    }

    #[test]
    fn expr_sizeof_expr_and_sizeof_type_forms() {
        assert!(matches!(
            parse_expr_src("sizeof a"),
            Expr::SizeofExpr { .. }
        ));
        assert!(matches!(
            parse_expr_src("sizeof(int)"),
            Expr::SizeofType {
                ty: Type::Scalar {
                    kind: ScalarKind::Int,
                    ..
                },
                ..
            }
        ));
        // no symbol table: a bare identifier after `(` is an expression, not a type, so this
        // is `sizeof` applied to the parenthesized expression `a`, not `sizeof(Type)`.
        assert!(matches!(
            parse_expr_src("sizeof(a)"),
            Expr::SizeofExpr { .. }
        ));
    }

    #[test]
    fn expr_cast_of_builtin_scalar_type() {
        let e = parse_expr_src("(int)x");
        let Expr::Cast { ty, expr, .. } = &e else {
            panic!("{e:?}");
        };
        assert!(matches!(
            ty,
            Type::Scalar {
                kind: ScalarKind::Int,
                ..
            }
        ));
        assert!(matches!(&**expr, Expr::Ident{name, ..} if name == "x"));
    }

    #[test]
    fn expr_postfix_call_index_and_member() {
        let e = parse_expr_src("f(a, b)");
        let Expr::Call { args, .. } = &e else {
            panic!("{e:?}");
        };
        assert_eq!(args.len(), 2);

        assert!(matches!(parse_expr_src("a[i]"), Expr::Index { .. }));

        let e = parse_expr_src("threadIdx.x");
        let Expr::Member { name, arrow, .. } = &e else {
            panic!("{e:?}");
        };
        assert_eq!(name, "x");
        assert!(!arrow);

        assert!(matches!(
            parse_expr_src("p->x"),
            Expr::Member { arrow: true, .. }
        ));
    }

    // ---- statements ------------------------------------------------------------------------

    #[test]
    fn if_else_statement() {
        let tu = parse_ok("void f() { if (a) b; else c; }");
        let body = first_fn_body(&tu);
        assert_eq!(body.len(), 1);
        let Stmt::If {
            then_branch,
            else_branch,
            ..
        } = &body[0]
        else {
            panic!("{:?}", body[0]);
        };
        assert!(matches!(**then_branch, Stmt::Expr { .. }));
        assert!(else_branch.is_some());
    }

    #[test]
    fn while_statement() {
        let tu = parse_ok("void f() { while (a < b) a++; }");
        let body = first_fn_body(&tu);
        assert!(matches!(body[0], Stmt::While { .. }));
    }

    #[test]
    fn do_while_statement() {
        let tu = parse_ok("void f() { do { a++; } while (a < 10); }");
        let body = first_fn_body(&tu);
        assert!(matches!(body[0], Stmt::DoWhile { .. }));
    }

    #[test]
    fn for_statement_all_clauses_present() {
        let tu = parse_ok("void f() { for (int i = 0; i < 10; i++) sum += i; }");
        let body = first_fn_body(&tu);
        let Stmt::For {
            init, cond, step, ..
        } = &body[0]
        else {
            panic!("{:?}", body[0]);
        };
        assert!(matches!(init.as_deref(), Some(Stmt::Decl { .. })));
        assert!(cond.is_some());
        assert!(step.is_some());
    }

    #[test]
    fn for_statement_with_omitted_clauses() {
        let tu = parse_ok("void f() { for (;;) { break; } }");
        let body = first_fn_body(&tu);
        let Stmt::For {
            init, cond, step, ..
        } = &body[0]
        else {
            panic!("{:?}", body[0]);
        };
        assert!(init.is_none());
        assert!(cond.is_none());
        assert!(step.is_none());
    }

    #[test]
    fn switch_case_default_statement() {
        let tu = parse_ok(
            r#"
            void f() {
                switch (x) {
                    case 1:
                        y = 1;
                        break;
                    case 2:
                    case 3:
                        y = 2;
                        break;
                    default:
                        y = 0;
                }
            }
            "#,
        );
        let body = first_fn_body(&tu);
        let Stmt::Switch { body: sbody, .. } = &body[0] else {
            panic!("{:?}", body[0]);
        };
        let Stmt::Block { stmts, .. } = sbody.as_ref() else {
            panic!("expected block switch body");
        };
        let Stmt::Case { value, stmt, .. } = &stmts[0] else {
            panic!("expected case, got {:?}", stmts[0]);
        };
        assert!(matches!(value, Expr::IntLit { .. }));
        assert!(matches!(**stmt, Stmt::Expr { .. }));
        // `case 2: case 3: ...` chains as nested Case nodes.
        let Stmt::Case { stmt: inner, .. } = &stmts[2] else {
            panic!("expected case, got {:?}", stmts[2]);
        };
        assert!(matches!(**inner, Stmt::Case { .. }));
        assert!(matches!(stmts[4], Stmt::Default { .. }));
    }

    #[test]
    fn break_continue_return_with_and_without_value() {
        let tu = parse_ok("void f() { while (1) { break; continue; } }");
        let body = first_fn_body(&tu);
        let Stmt::While { body: wbody, .. } = &body[0] else {
            panic!("{:?}", body[0]);
        };
        let Stmt::Block { stmts, .. } = wbody.as_ref() else {
            panic!();
        };
        assert!(matches!(stmts[0], Stmt::Break { .. }));
        assert!(matches!(stmts[1], Stmt::Continue { .. }));

        let tu = parse_ok("int f() { return 5; }");
        let body = first_fn_body(&tu);
        assert!(matches!(body[0], Stmt::Return { expr: Some(_), .. }));

        let tu = parse_ok("void f() { return; }");
        let body = first_fn_body(&tu);
        assert!(matches!(body[0], Stmt::Return { expr: None, .. }));
    }

    #[test]
    fn labeled_statement_and_goto() {
        let tu = parse_ok("void f() { start: x++; goto start; }");
        let body = first_fn_body(&tu);
        assert!(matches!(body[0], Stmt::Label{ref name, ..} if name == "start"));
        assert!(matches!(body[1], Stmt::Goto{ref label, ..} if label == "start"));
    }

    #[test]
    fn nested_blocks_with_own_declarations() {
        let tu = parse_ok("void f() { int x = 1; { int x = 2; x = x + 1; } }");
        let body = first_fn_body(&tu);
        assert!(matches!(body[0], Stmt::Decl { .. }));
        let Stmt::Block { stmts, .. } = &body[1] else {
            panic!("{:?}", body[1]);
        };
        assert!(matches!(stmts[0], Stmt::Decl { .. }));
        assert!(matches!(stmts[1], Stmt::Expr { .. }));
    }

    #[test]
    fn full_function_body_with_mixed_statements() {
        let tu = parse_ok(
            r#"
            int sum(int n) {
                int total = 0;
                for (int i = 0; i < n; i++) {
                    if (i % 2 == 0) {
                        total += i;
                    } else {
                        continue;
                    }
                }
                return total;
            }
            "#,
        );
        let body = first_fn_body(&tu);
        assert_eq!(body.len(), 3);
        assert!(matches!(body[0], Stmt::Decl { .. }));
        assert!(matches!(body[1], Stmt::For { .. }));
        assert!(matches!(body[2], Stmt::Return { expr: Some(_), .. }));
    }

    #[test]
    fn array_size_is_a_parsed_expression_not_folded() {
        let tu = parse_ok("void f(int n) { int arr[n + 1]; }");
        let body = first_fn_body(&tu);
        let Stmt::Decl { decls, .. } = &body[0] else {
            panic!("{:?}", body[0]);
        };
        let Type::Array { size, .. } = &decls[0].ty else {
            panic!("expected array type, got {:?}", decls[0].ty);
        };
        let size = size.as_ref().expect("expected array size expression");
        assert!(matches!(**size, Expr::Binary { op: BinOp::Add, .. }));
    }

    #[test]
    fn enum_variant_with_expression_initializer() {
        let tu = parse_ok("enum E { A = 1 + 2 };");
        let Item::Enum(e) = &tu.items[0] else {
            panic!("expected enum");
        };
        let init = e.variants[0].init.as_ref().expect("expected initializer");
        assert!(matches!(init, Expr::Binary { op: BinOp::Add, .. }));
    }

    #[test]
    fn operator_overload_function_names() {
        let tu = parse_ok("Vec operator+(Vec other);");
        let Item::Function(f) = &tu.items[0] else {
            panic!("expected function");
        };
        assert_eq!(f.name, "operator+");

        let tu = parse_ok("bool operator==(Vec other);");
        let Item::Function(f) = &tu.items[0] else {
            panic!("expected function");
        };
        assert_eq!(f.name, "operator==");
    }

    #[test]
    fn malformed_statement_recovers_via_synchronize_stmt() {
        let (tokens, lex_errs) = lex(r#"
            void f() {
                ) ;
                int y = 2;
            }
            "#);
        assert!(lex_errs.is_empty());
        let (tu, errs) = parse(&tokens);
        assert!(!errs.is_empty(), "expected at least one parse error");
        let body = first_fn_body(&tu);
        let found_y = body
            .iter()
            .any(|s| matches!(s, Stmt::Decl { decls, .. } if decls.iter().any(|d| d.name == "y")));
        assert!(found_y, "body: {body:?}");
    }
}
