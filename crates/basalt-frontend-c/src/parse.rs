// Recursive-descent parser over an already-lexed `&[Token]` (see `token.rs`/`lex.rs`); this
// stage never touches source text directly, so it stays decoupled from lexing/preprocessing.
//
// Scope is declarations and types only (ARCHITECTURE.md §6): structs/enums/unions/typedefs/
// namespaces, the scalar/pointer/array type grammar, and function *signatures*. Statement and
// expression grammar land in a later stage; wherever this parser would need them (a function
// body, an array-size expression, an enum initializer, a template's instantiated body) it
// records an opaque `TokenRange` instead.
//
// Like `lex`, this never aborts on the first error: `parse` always consumes the full token
// stream and returns whatever `ParseError`s it collected alongside the tree. A malformed item
// is resynchronized at the next plausible boundary (`;` or `}` at the current depth, or a
// keyword that starts a new item) so one bad declaration can't take the rest of the file with
// it or hang the parser.

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

    /// Skips a `{ ... }` block, braces included, tracking nested braces. `self.pos` must be
    /// at the opening `{`. Reports (but recovers from) an unterminated block.
    fn skip_balanced_braces(&mut self) -> TokenRange {
        let start_idx = self.pos;
        let start_span = self.cur().span;
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Punct(Punct::LBrace) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RBrace) => {
                    depth -= 1;
                    self.bump();
                    if depth == 0 {
                        break;
                    }
                }
                TokenKind::Eof => {
                    self.errors.push(ParseError::Expected {
                        what: "'}'".to_string(),
                        found: "end of file".to_string(),
                        span: self.cur().span,
                    });
                    break;
                }
                _ => {
                    self.bump();
                }
            }
        }
        TokenRange {
            start: start_idx,
            end: self.pos,
            span: Span::new(start_span.start, self.prev_span_end().end),
        }
    }

    /// Skips forward to (not past) the `]` matching the `[` just consumed, tracking bracket
    /// nesting so a nested index expression doesn't stop the scan early.
    fn skip_until_matching_bracket(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Punct(Punct::LBracket) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RBracket) => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Eof => break,
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Skips an initializer expression up to (not past) the next `,` or `;` at depth 0,
    /// tracking `(`/`[`/`{` nesting so those don't get mistaken for the terminator.
    fn skip_until_decl_separator(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.cur().kind {
                TokenKind::Punct(Punct::LParen | Punct::LBracket | Punct::LBrace) => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::RParen | Punct::RBracket | Punct::RBrace) => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Punct(Punct::Comma | Punct::Semi) if depth == 0 => break,
                TokenKind::Eof => break,
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
                let body = self.skip_balanced_braces();
                let span = Span::new(start, body.span.end);
                out.push(Item::Function(FunctionDecl {
                    ret: ty,
                    name: fn_name,
                    params,
                    variadic,
                    body: Some(body),
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
                let s = self.pos;
                let s_span = self.cur().span.start;
                self.skip_until_decl_separator();
                Some(TokenRange {
                    start: s,
                    end: self.pos,
                    span: Span::new(s_span, self.prev_span_end().end),
                })
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
                let s = self.pos;
                let s_span = self.cur().span.start;
                self.skip_until_decl_separator();
                Some(TokenRange {
                    start: s,
                    end: self.pos,
                    span: Span::new(s_span, self.prev_span_end().end),
                })
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
        let name = self.ident_here();
        if name.is_some() {
            self.bump();
        }
        while self.check_punct(Punct::LBracket) {
            let lb_span = self.cur().span;
            self.bump();
            let size = if self.check_punct(Punct::RBracket) {
                None
            } else {
                let s = self.pos;
                let s_span = self.cur().span.start;
                self.skip_until_matching_bracket();
                Some(TokenRange {
                    start: s,
                    end: self.pos,
                    span: Span::new(s_span, self.prev_span_end().end),
                })
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
        let body = f.body.expect("expected body");
        assert!(matches!(
            tokens[body.start].kind,
            TokenKind::Punct(Punct::LBrace)
        ));
        assert!(matches!(
            tokens[body.end - 1].kind,
            TokenKind::Punct(Punct::RBrace)
        ));
        // depth-balance sanity check over the captured range
        let mut depth = 0i32;
        for tok in &tokens[body.start..body.end] {
            match tok.kind {
                TokenKind::Punct(Punct::LBrace) => depth += 1,
                TokenKind::Punct(Punct::RBrace) => depth -= 1,
                _ => {}
            }
        }
        assert_eq!(depth, 0);
        // and it should not have swallowed the Eof sentinel or run past the real end
        assert!(body.end <= tokens.len());
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
}
