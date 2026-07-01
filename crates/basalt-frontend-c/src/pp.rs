// The C preprocessor: `#include`/`#define`/`#undef`, function-like macros, conditional
// compilation (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`), `#pragma`/`#error`, and
// command-line-style `-I`/`-D` seeding. Operates on the already-lexed `Token` stream from
// `lex::lex` rather than raw source, so it inherits that scanner's Eof-terminated,
// error-tolerant shape.
//
// `lex.rs` has no notion of "first token on a physical line" or "newline", so directives are
// found by comparing `Span` line numbers between adjacent tokens (see `is_line_initial_hash`),
// and a directive's extent is every following token whose `span.start.line` matches the `#`.
// This means backslash-newline line continuation (splicing two physical lines into one
// logical line before a directive is recognized) is not supported: `lex.rs` doesn't perform
// that splice either, and it can't be added here without touching the scanner.
//
// Macro self-reference protection uses the standard "painted blue" technique: expanding a
// macro adds its name to an `active` set threaded through the rescan of its own replacement
// list, so a self-referential body prints the invoking name literally instead of recursing
// forever. This is scoped per rescan (the body's own token slice), not a full per-token
// hide-set spanning the whole file, so a macro that expands to a *partial* invocation of a
// function-like macro whose closing parenthesis and arguments live outside its own
// replacement list (a rare, exotic macro-chaining trick) will not see those outer tokens.
// Ordinary macro use, including nested and mutually-recursive expansion, works as expected.
//
// `-D NAME` with no `=value` seeds `NAME` as `1` (matching gcc/clang's documented behavior
// for a valueless `-D`), not an empty replacement list.
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::lex::{lex, LexError};
use crate::token::{IntBase, IntLit, Keyword, Loc, Punct, Span, Token, TokenKind};

/// Preprocessor configuration: search path for `#include` and command-line-style defines,
/// applied before the first token of `src` is seen.
#[derive(Debug, Clone, Default)]
pub struct PpOpts {
    pub include_dirs: Vec<PathBuf>,
    /// `(name, value)` pairs, in `-D` order. `value == None` means a bare `-D NAME`.
    pub defines: Vec<(String, Option<String>)>,
    /// Directory `src` is considered to live in, used as a last-resort search location for
    /// quoted includes (real callers pass the source file's own directory here).
    pub base_dir: Option<PathBuf>,
}

/// A problem found while preprocessing. Every variant carries the `Span` of the offending
/// directive or token; `Lex` wraps a scanning error hit while lexing `src` itself or an
/// included file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PpError {
    BadDirective(Span),
    UnknownDirective(Span),
    BadMacroDef(Span),
    BadInclude(Span),
    IncludeNotFound(Span),
    IncludeReadError(Span),
    IncludeCycle(Span),
    UnterminatedIf(Span),
    UnmatchedElif(Span),
    UnmatchedElse(Span),
    UnmatchedEndif(Span),
    ElseAfterElse(Span),
    ElifAfterElse(Span),
    BadConstExpr(Span),
    MacroArity(Span),
    UserError(Span, String),
    Lex(LexError),
}

impl PpError {
    pub fn span(&self) -> Span {
        match self {
            PpError::BadDirective(s)
            | PpError::UnknownDirective(s)
            | PpError::BadMacroDef(s)
            | PpError::BadInclude(s)
            | PpError::IncludeNotFound(s)
            | PpError::IncludeReadError(s)
            | PpError::IncludeCycle(s)
            | PpError::UnterminatedIf(s)
            | PpError::UnmatchedElif(s)
            | PpError::UnmatchedElse(s)
            | PpError::UnmatchedEndif(s)
            | PpError::ElseAfterElse(s)
            | PpError::ElifAfterElse(s)
            | PpError::BadConstExpr(s)
            | PpError::MacroArity(s)
            | PpError::UserError(s, _) => *s,
            PpError::Lex(e) => e.span(),
        }
    }
}

impl fmt::Display for PpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PpError::BadDirective(s) => write!(f, "malformed preprocessor directive ({s})"),
            PpError::UnknownDirective(s) => write!(f, "unknown preprocessor directive ({s})"),
            PpError::BadMacroDef(s) => write!(f, "malformed macro definition ({s})"),
            PpError::BadInclude(s) => write!(f, "malformed #include directive ({s})"),
            PpError::IncludeNotFound(s) => write!(f, "include file not found ({s})"),
            PpError::IncludeReadError(s) => write!(f, "could not read include file ({s})"),
            PpError::IncludeCycle(s) => write!(f, "circular #include ({s})"),
            PpError::UnterminatedIf(s) => write!(f, "unterminated #if at end of file ({s})"),
            PpError::UnmatchedElif(s) => write!(f, "#elif without matching #if ({s})"),
            PpError::UnmatchedElse(s) => write!(f, "#else without matching #if ({s})"),
            PpError::UnmatchedEndif(s) => write!(f, "#endif without matching #if ({s})"),
            PpError::ElseAfterElse(s) => write!(f, "#else after #else ({s})"),
            PpError::ElifAfterElse(s) => write!(f, "#elif after #else ({s})"),
            PpError::BadConstExpr(s) => write!(f, "invalid constant expression ({s})"),
            PpError::MacroArity(s) => write!(f, "macro argument count mismatch ({s})"),
            PpError::UserError(s, msg) => write!(f, "#error {msg} ({s})"),
            PpError::Lex(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PpError {}

/// A macro's replacement definition. `params == None` marks an object-like macro;
/// `Some(names)` marks function-like (an empty `Vec` is a valid zero-parameter macro).
#[derive(Debug, Clone)]
struct MacroDef {
    params: Option<Vec<String>>,
    body: Vec<Token>,
}

/// One open `#if`/`#ifdef`/`#ifndef` group.
#[derive(Debug)]
struct CondFrame {
    /// Set when an enclosing group was already inactive at the time this one was opened;
    /// nothing in a dead group is ever evaluated or made active, regardless of its own
    /// `#elif`/`#else` conditions.
    dead: bool,
    /// Whether any branch in this group has been active so far.
    any_taken: bool,
    /// Whether the branch currently open is the one being emitted.
    branch_active: bool,
    seen_else: bool,
    start: Span,
}

impl CondFrame {
    fn new(cond: bool, start: Span) -> CondFrame {
        CondFrame {
            dead: false,
            any_taken: cond,
            branch_active: cond,
            seen_else: false,
            start,
        }
    }

    fn dead(start: Span) -> CondFrame {
        CondFrame {
            dead: true,
            any_taken: true,
            branch_active: false,
            seen_else: false,
            start,
        }
    }
}

fn stack_active(stack: &[CondFrame]) -> bool {
    stack.iter().all(|f| f.branch_active)
}

/// Preprocesses `src`, returning the fully macro-expanded, `#include`-spliced, conditional-
/// resolved token stream (always Eof-terminated) plus every problem found along the way.
/// Never aborts early: a malformed directive, missing include, or dangling conditional is
/// recorded as a `PpError` and preprocessing continues to the end of the input.
pub fn preprocess(src: &str, opts: &PpOpts) -> (Vec<Token>, Vec<PpError>) {
    let mut errors = Vec::new();
    let (tokens, lex_errors) = lex(src);
    for e in lex_errors {
        errors.push(PpError::Lex(e));
    }
    let eof_span = tokens.last().map(|t| t.span).unwrap_or_else(|| {
        let z = Loc::new(0, 1, 1);
        Span::new(z, z)
    });
    let body = &tokens[..tokens.len().saturating_sub(1)];

    let mut macros: HashMap<String, MacroDef> = HashMap::new();
    seed_defines(&mut macros, &opts.defines, &mut errors);

    let base_dir = opts.base_dir.clone().unwrap_or_else(|| PathBuf::from("."));
    let mut out = Vec::new();
    let mut include_stack = Vec::new();
    process_file(
        src,
        body,
        &base_dir,
        opts,
        &mut macros,
        &mut out,
        &mut errors,
        &mut include_stack,
    );
    out.push(Token::new(TokenKind::Eof, eof_span));
    (out, errors)
}

fn seed_defines(
    macros: &mut HashMap<String, MacroDef>,
    defines: &[(String, Option<String>)],
    errors: &mut Vec<PpError>,
) {
    let synth = {
        let z = Loc::new(0, 0, 0);
        Span::new(z, z)
    };
    for (name, val) in defines {
        let body = match val {
            None => vec![Token::new(
                TokenKind::IntLit(IntLit {
                    text: "1".to_string(),
                    base: IntBase::Dec,
                    unsigned: false,
                    long_len: 0,
                }),
                synth,
            )],
            Some(text) => {
                let (toks, lex_errors) = lex(text);
                for e in lex_errors {
                    errors.push(PpError::Lex(e));
                }
                toks.into_iter()
                    .filter(|t| !matches!(t.kind, TokenKind::Eof))
                    .collect()
            }
        };
        macros.insert(name.clone(), MacroDef { params: None, body });
    }
}

/// Classifies a token as a directive keyword name, if it is one. `if`/`else` lex as
/// `Keyword` (they're reserved words in the grammar) rather than `Ident`, so both shapes are
/// checked here; every other directive name (`include`, `define`, `ifdef`, ...) is not a
/// keyword and lexes as a plain `Ident`.
fn directive_word(tok: &Token) -> Option<&str> {
    match &tok.kind {
        TokenKind::Ident(s) => Some(s.as_str()),
        TokenKind::Keyword(Keyword::If) => Some("if"),
        TokenKind::Keyword(Keyword::Else) => Some("else"),
        _ => None,
    }
}

fn is_line_initial_hash(tokens: &[Token], idx: usize) -> bool {
    if !matches!(tokens[idx].kind, TokenKind::Punct(Punct::Hash)) {
        return false;
    }
    idx == 0 || tokens[idx].span.start.line != tokens[idx - 1].span.end.line
}

/// Returns the index one past the last token sharing `tokens[i]`'s physical line — the
/// directive's extent, since there is no explicit newline token to stop at.
fn find_line_end(tokens: &[Token], i: usize) -> usize {
    let line = tokens[i].span.start.line;
    let mut j = i + 1;
    while j < tokens.len() && tokens[j].span.start.line == line {
        j += 1;
    }
    j
}

/// Scans one file's worth of tokens (the top-level source, or a spliced-in `#include`),
/// splitting it into directive lines and the runs of ordinary tokens between them. Ordinary
/// runs are macro-expanded and appended to `out` only while the conditional stack is active;
/// directives are always recognized (even while skipping) so nesting stays correct. Shares
/// `macros` and `out` with any enclosing file so definitions and emitted tokens cross
/// `#include` boundaries the way real C requires; the conditional stack itself is local to
/// this file, so a dangling `#if` is reported against the file that opened it.
#[allow(clippy::too_many_arguments)]
fn process_file(
    src: &str,
    tokens: &[Token],
    cur_dir: &Path,
    opts: &PpOpts,
    macros: &mut HashMap<String, MacroDef>,
    out: &mut Vec<Token>,
    errors: &mut Vec<PpError>,
    include_stack: &mut Vec<PathBuf>,
) {
    let n = tokens.len();
    let mut stack: Vec<CondFrame> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if is_line_initial_hash(tokens, i) {
            let j = find_line_end(tokens, i);
            handle_directive(
                src,
                tokens,
                i,
                j,
                cur_dir,
                opts,
                macros,
                &mut stack,
                out,
                errors,
                include_stack,
            );
            i = j;
            continue;
        }
        let mut k = i + 1;
        while k < n && !is_line_initial_hash(tokens, k) {
            k += 1;
        }
        if stack_active(&stack) {
            let expanded = expand_run(&tokens[i..k], macros, &HashSet::new(), errors);
            out.extend(expanded);
        }
        i = k;
    }
    if let Some(top) = stack.last() {
        errors.push(PpError::UnterminatedIf(top.start));
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_directive(
    src: &str,
    tokens: &[Token],
    i: usize,
    j: usize,
    cur_dir: &Path,
    opts: &PpOpts,
    macros: &mut HashMap<String, MacroDef>,
    stack: &mut Vec<CondFrame>,
    out: &mut Vec<Token>,
    errors: &mut Vec<PpError>,
    include_stack: &mut Vec<PathBuf>,
) {
    let hash_span = tokens[i].span;
    if i + 1 >= j {
        // A lone `#` on its own line: the null directive, a documented no-op.
        return;
    }
    let word = directive_word(&tokens[i + 1]);
    let rest = &tokens[i + 2..j];
    let is_active = stack_active(stack);

    match word {
        Some("include") => {
            if is_active {
                handle_include(
                    src,
                    rest,
                    hash_span,
                    cur_dir,
                    opts,
                    macros,
                    out,
                    errors,
                    include_stack,
                );
            }
        }
        Some("define") => {
            if is_active {
                handle_define(rest, hash_span, macros, errors);
            }
        }
        Some("undef") => {
            if is_active {
                match rest.first().map(|t| &t.kind) {
                    Some(TokenKind::Ident(name)) => {
                        macros.remove(name);
                    }
                    _ => errors.push(PpError::BadDirective(hash_span)),
                }
            }
        }
        Some("ifdef") | Some("ifndef") => {
            let negate = word == Some("ifndef");
            if !is_active {
                stack.push(CondFrame::dead(hash_span));
            } else {
                match rest.first().map(|t| &t.kind) {
                    Some(TokenKind::Ident(name)) => {
                        let defined = macros.contains_key(name);
                        let cond = if negate { !defined } else { defined };
                        stack.push(CondFrame::new(cond, hash_span));
                    }
                    _ => {
                        errors.push(PpError::BadDirective(hash_span));
                        stack.push(CondFrame::dead(hash_span));
                    }
                }
            }
        }
        Some("if") => {
            if !is_active {
                stack.push(CondFrame::dead(hash_span));
            } else {
                let cond = eval_constant_expr(rest, macros, errors, hash_span);
                stack.push(CondFrame::new(cond != 0, hash_span));
            }
        }
        Some("elif") => match stack.last_mut() {
            None => errors.push(PpError::UnmatchedElif(hash_span)),
            Some(top) => {
                if top.seen_else {
                    errors.push(PpError::ElifAfterElse(hash_span));
                    top.branch_active = false;
                } else if top.dead || top.any_taken {
                    top.branch_active = false;
                } else {
                    let cond = eval_constant_expr(rest, macros, errors, hash_span);
                    top.branch_active = cond != 0;
                    top.any_taken = top.branch_active;
                }
            }
        },
        Some("else") => match stack.last_mut() {
            None => errors.push(PpError::UnmatchedElse(hash_span)),
            Some(top) => {
                if top.seen_else {
                    errors.push(PpError::ElseAfterElse(hash_span));
                }
                top.seen_else = true;
                if top.dead || top.any_taken {
                    top.branch_active = false;
                } else {
                    top.branch_active = true;
                    top.any_taken = true;
                }
            }
        },
        Some("endif") => {
            if stack.pop().is_none() {
                errors.push(PpError::UnmatchedEndif(hash_span));
            }
        }
        Some("pragma") => {
            // No pragma has target-specific semantics yet; accept and drop silently.
        }
        Some("error") => {
            if is_active {
                let msg = if rest.is_empty() {
                    String::new()
                } else {
                    let start = rest[0].span.start.offset as usize;
                    let end = rest[rest.len() - 1].span.end.offset as usize;
                    src[start..end].to_string()
                };
                errors.push(PpError::UserError(hash_span, msg));
            }
        }
        _ => {
            if is_active {
                errors.push(PpError::UnknownDirective(hash_span));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_include(
    src: &str,
    rest: &[Token],
    hash_span: Span,
    cur_dir: &Path,
    opts: &PpOpts,
    macros: &mut HashMap<String, MacroDef>,
    out: &mut Vec<Token>,
    errors: &mut Vec<PpError>,
    include_stack: &mut Vec<PathBuf>,
) {
    if rest.is_empty() {
        errors.push(PpError::BadInclude(hash_span));
        return;
    }
    let (rel_path, quoted) = match &rest[0].kind {
        TokenKind::StrLit(s) => {
            let raw = s.raw.as_str();
            let inner = if raw.len() >= 2 {
                raw[1..raw.len() - 1].to_string()
            } else {
                String::new()
            };
            (inner, true)
        }
        TokenKind::Punct(Punct::Lt) => {
            let gt_idx = rest
                .iter()
                .position(|t| matches!(t.kind, TokenKind::Punct(Punct::Gt)));
            match gt_idx {
                Some(gi) => {
                    let start_off = rest[0].span.end.offset as usize;
                    let end_off = rest[gi].span.start.offset as usize;
                    (src[start_off..end_off].to_string(), false)
                }
                None => {
                    errors.push(PpError::BadInclude(hash_span));
                    return;
                }
            }
        }
        _ => {
            errors.push(PpError::BadInclude(hash_span));
            return;
        }
    };
    if rel_path.is_empty() {
        errors.push(PpError::BadInclude(hash_span));
        return;
    }

    let resolved = resolve_include(quoted, &rel_path, cur_dir, opts);
    let path = match resolved {
        Some(p) => p,
        None => {
            errors.push(PpError::IncludeNotFound(hash_span));
            return;
        }
    };
    let key = path.canonicalize().unwrap_or_else(|_| path.clone());
    if include_stack.contains(&key) {
        errors.push(PpError::IncludeCycle(hash_span));
        return;
    }
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => {
            errors.push(PpError::IncludeReadError(hash_span));
            return;
        }
    };
    let (inc_tokens, lex_errors) = lex(&contents);
    for e in lex_errors {
        errors.push(PpError::Lex(e));
    }
    let inc_body = &inc_tokens[..inc_tokens.len().saturating_sub(1)];
    let inc_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| cur_dir.to_path_buf());

    include_stack.push(key);
    process_file(
        &contents,
        inc_body,
        &inc_dir,
        opts,
        macros,
        out,
        errors,
        include_stack,
    );
    include_stack.pop();
}

fn resolve_include(quoted: bool, rel: &str, cur_dir: &Path, opts: &PpOpts) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if quoted {
        candidates.push(cur_dir.join(rel));
    }
    for d in &opts.include_dirs {
        candidates.push(d.join(rel));
    }
    if quoted {
        if let Some(b) = &opts.base_dir {
            candidates.push(b.join(rel));
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn handle_define(
    rest: &[Token],
    hash_span: Span,
    macros: &mut HashMap<String, MacroDef>,
    errors: &mut Vec<PpError>,
) {
    if rest.is_empty() {
        errors.push(PpError::BadMacroDef(hash_span));
        return;
    }
    let name = match &rest[0].kind {
        TokenKind::Ident(s) => s.clone(),
        _ => {
            errors.push(PpError::BadMacroDef(hash_span));
            return;
        }
    };
    let name_end = rest[0].span.end.offset;

    let is_func_like = rest.len() > 1
        && matches!(rest[1].kind, TokenKind::Punct(Punct::LParen))
        && rest[1].span.start.offset == name_end;

    if is_func_like {
        let mut idx = 2usize;
        let mut params: Vec<String> = Vec::new();
        if idx < rest.len() && matches!(rest[idx].kind, TokenKind::Punct(Punct::RParen)) {
            idx += 1;
        } else {
            loop {
                match rest.get(idx).map(|t| &t.kind) {
                    Some(TokenKind::Ident(p)) => {
                        params.push(p.clone());
                        idx += 1;
                    }
                    _ => {
                        errors.push(PpError::BadMacroDef(hash_span));
                        return;
                    }
                }
                match rest.get(idx).map(|t| &t.kind) {
                    Some(TokenKind::Punct(Punct::Comma)) => idx += 1,
                    Some(TokenKind::Punct(Punct::RParen)) => {
                        idx += 1;
                        break;
                    }
                    _ => {
                        errors.push(PpError::BadMacroDef(hash_span));
                        return;
                    }
                }
            }
        }
        let body = rest[idx..].to_vec();
        macros.insert(
            name,
            MacroDef {
                params: Some(params),
                body,
            },
        );
    } else {
        let body = rest[1..].to_vec();
        macros.insert(name, MacroDef { params: None, body });
    }
}

/// Macro-expands `tokens` (one run between directives, a macro body being rescanned, or the
/// resolved text of an `#if`/`#elif` expression), never re-expanding a name in `active` —
/// the "painted blue" set that stops a self-referential macro from recursing forever.
fn expand_run(
    tokens: &[Token],
    macros: &HashMap<String, MacroDef>,
    active: &HashSet<String>,
    errors: &mut Vec<PpError>,
) -> Vec<Token> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let tok = &tokens[i];
        let mut consumed = false;
        if let TokenKind::Ident(name) = &tok.kind {
            if !active.contains(name) {
                if let Some(def) = macros.get(name) {
                    match &def.params {
                        Some(params) => {
                            if i + 1 < tokens.len()
                                && matches!(tokens[i + 1].kind, TokenKind::Punct(Punct::LParen))
                            {
                                if let Some((args, after)) = parse_args(tokens, i + 2) {
                                    if args.len() != params.len() {
                                        errors.push(PpError::MacroArity(tok.span));
                                    }
                                    let subst = substitute(&def.body, params, &args);
                                    let mut next_active = active.clone();
                                    next_active.insert(name.clone());
                                    out.extend(expand_run(&subst, macros, &next_active, errors));
                                    i = after;
                                    consumed = true;
                                }
                            }
                        }
                        None => {
                            let mut next_active = active.clone();
                            next_active.insert(name.clone());
                            out.extend(expand_run(&def.body, macros, &next_active, errors));
                            i += 1;
                            consumed = true;
                        }
                    }
                }
            }
        }
        if !consumed {
            out.push(tok.clone());
            i += 1;
        }
    }
    out
}

/// Parses a balanced, comma-separated argument list starting right after the opening `(`
/// (already consumed). Returns the argument token lists and the index one past the closing
/// `)`, or `None` if `tokens` runs out before it balances.
fn parse_args(tokens: &[Token], start: usize) -> Option<(Vec<Vec<Token>>, usize)> {
    if start < tokens.len() && matches!(tokens[start].kind, TokenKind::Punct(Punct::RParen)) {
        return Some((Vec::new(), start + 1));
    }
    let mut args = Vec::new();
    let mut cur = Vec::new();
    let mut depth = 0i32;
    let mut i = start;
    while i < tokens.len() {
        match &tokens[i].kind {
            TokenKind::Punct(Punct::LParen) => {
                depth += 1;
                cur.push(tokens[i].clone());
            }
            TokenKind::Punct(Punct::RParen) => {
                if depth == 0 {
                    args.push(cur);
                    return Some((args, i + 1));
                }
                depth -= 1;
                cur.push(tokens[i].clone());
            }
            TokenKind::Punct(Punct::Comma) if depth == 0 => {
                args.push(std::mem::take(&mut cur));
            }
            _ => cur.push(tokens[i].clone()),
        }
        i += 1;
    }
    None
}

fn substitute(body: &[Token], params: &[String], args: &[Vec<Token>]) -> Vec<Token> {
    let mut out = Vec::new();
    for t in body {
        if let TokenKind::Ident(name) = &t.kind {
            if let Some(idx) = params.iter().position(|p| p == name) {
                if let Some(a) = args.get(idx) {
                    out.extend(a.iter().cloned());
                }
                continue;
            }
        }
        out.push(t.clone());
    }
    out
}

/// Rewrites `defined NAME` / `defined(NAME)` into a literal `0`/`1` token, evaluated against
/// the macro table *before* general macro expansion runs on the rest of the expression — the
/// standard rule that protects `defined`'s operand from being macro-substituted itself.
fn resolve_defined(
    tokens: &[Token],
    macros: &HashMap<String, MacroDef>,
    errors: &mut Vec<PpError>,
) -> Vec<Token> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let is_defined = matches!(&tokens[i].kind, TokenKind::Ident(n) if n == "defined");
        if is_defined {
            let span = tokens[i].span;
            if i + 3 < tokens.len()
                && matches!(tokens[i + 1].kind, TokenKind::Punct(Punct::LParen))
                && matches!(tokens[i + 3].kind, TokenKind::Punct(Punct::RParen))
            {
                if let TokenKind::Ident(target) = &tokens[i + 2].kind {
                    let v = macros.contains_key(target) as i64;
                    out.push(make_int_tok(v, span));
                    i += 4;
                    continue;
                }
            }
            if i + 1 < tokens.len() {
                if let TokenKind::Ident(target) = &tokens[i + 1].kind {
                    let v = macros.contains_key(target) as i64;
                    out.push(make_int_tok(v, span));
                    i += 2;
                    continue;
                }
            }
            errors.push(PpError::BadConstExpr(span));
            out.push(make_int_tok(0, span));
            i += 1;
            continue;
        }
        out.push(tokens[i].clone());
        i += 1;
    }
    out
}

fn make_int_tok(v: i64, span: Span) -> Token {
    Token::new(
        TokenKind::IntLit(IntLit {
            text: v.to_string(),
            base: IntBase::Dec,
            unsigned: false,
            long_len: 0,
        }),
        span,
    )
}

/// Evaluates the constant-expression tail of an `#if`/`#elif` line: resolves `defined`,
/// macro-expands what's left, then parses/evaluates it as an integer constant expression.
fn eval_constant_expr(
    rest: &[Token],
    macros: &HashMap<String, MacroDef>,
    errors: &mut Vec<PpError>,
    span: Span,
) -> i64 {
    if rest.is_empty() {
        errors.push(PpError::BadConstExpr(span));
        return 0;
    }
    let resolved = resolve_defined(rest, macros, errors);
    let expanded = expand_run(&resolved, macros, &HashSet::new(), errors);
    let mut p = ExprParser {
        toks: &expanded,
        pos: 0,
        errors,
        err_span: span,
    };
    p.parse()
}

fn int_lit_value(l: &IntLit) -> i64 {
    let mut s = l.text.as_str();
    while matches!(s.as_bytes().last(), Some(b'u' | b'U' | b'l' | b'L')) {
        s = &s[..s.len() - 1];
    }
    let (radix, digits) = match l.base {
        IntBase::Dec => (10, s),
        IntBase::Oct => (8, s),
        IntBase::Hex => (16, s.get(2..).unwrap_or("")),
        IntBase::Bin => (2, s.get(2..).unwrap_or("")),
    };
    i64::from_str_radix(digits, radix).unwrap_or(0)
}

/// Small recursive-descent evaluator for `#if`/`#elif` constant expressions: integer and
/// character literals, `true`/`false`, the usual C operators (down through unary), and
/// parens. Every error path records one `PpError::BadConstExpr` and yields `0` so a malformed
/// expression can't hang or panic the pass.
struct ExprParser<'a> {
    toks: &'a [Token],
    pos: usize,
    errors: &'a mut Vec<PpError>,
    err_span: Span,
}

impl<'a> ExprParser<'a> {
    fn kind_at(&self, k: usize) -> Option<&TokenKind> {
        self.toks.get(k).map(|t| &t.kind)
    }

    fn err(&mut self) -> i64 {
        self.errors.push(PpError::BadConstExpr(self.err_span));
        0
    }

    fn parse(&mut self) -> i64 {
        if self.toks.is_empty() {
            return self.err();
        }
        let v = self.conditional();
        if self.pos != self.toks.len() {
            return self.err();
        }
        v
    }

    fn conditional(&mut self) -> i64 {
        let c = self.logor();
        if matches!(
            self.kind_at(self.pos),
            Some(TokenKind::Punct(Punct::Question))
        ) {
            self.pos += 1;
            let a = self.conditional();
            if matches!(self.kind_at(self.pos), Some(TokenKind::Punct(Punct::Colon))) {
                self.pos += 1;
            } else {
                return self.err();
            }
            let b = self.conditional();
            return if c != 0 { a } else { b };
        }
        c
    }

    fn logor(&mut self) -> i64 {
        let mut v = self.logand();
        while matches!(
            self.kind_at(self.pos),
            Some(TokenKind::Punct(Punct::PipePipe))
        ) {
            self.pos += 1;
            let rhs = self.logand();
            v = ((v != 0) || (rhs != 0)) as i64;
        }
        v
    }

    fn logand(&mut self) -> i64 {
        let mut v = self.bitor();
        while matches!(
            self.kind_at(self.pos),
            Some(TokenKind::Punct(Punct::AmpAmp))
        ) {
            self.pos += 1;
            let rhs = self.bitor();
            v = ((v != 0) && (rhs != 0)) as i64;
        }
        v
    }

    fn bitor(&mut self) -> i64 {
        let mut v = self.bitxor();
        while matches!(self.kind_at(self.pos), Some(TokenKind::Punct(Punct::Pipe))) {
            self.pos += 1;
            v |= self.bitxor();
        }
        v
    }

    fn bitxor(&mut self) -> i64 {
        let mut v = self.bitand();
        while matches!(self.kind_at(self.pos), Some(TokenKind::Punct(Punct::Caret))) {
            self.pos += 1;
            v ^= self.bitand();
        }
        v
    }

    fn bitand(&mut self) -> i64 {
        let mut v = self.equality();
        while matches!(self.kind_at(self.pos), Some(TokenKind::Punct(Punct::Amp))) {
            self.pos += 1;
            v &= self.equality();
        }
        v
    }

    fn equality(&mut self) -> i64 {
        let mut v = self.relational();
        loop {
            match self.kind_at(self.pos) {
                Some(TokenKind::Punct(Punct::EqEq)) => {
                    self.pos += 1;
                    v = (v == self.relational()) as i64;
                }
                Some(TokenKind::Punct(Punct::NotEq)) => {
                    self.pos += 1;
                    v = (v != self.relational()) as i64;
                }
                _ => break,
            }
        }
        v
    }

    fn relational(&mut self) -> i64 {
        let mut v = self.shift();
        loop {
            match self.kind_at(self.pos) {
                Some(TokenKind::Punct(Punct::Lt)) => {
                    self.pos += 1;
                    v = (v < self.shift()) as i64;
                }
                Some(TokenKind::Punct(Punct::Gt)) => {
                    self.pos += 1;
                    v = (v > self.shift()) as i64;
                }
                Some(TokenKind::Punct(Punct::Le)) => {
                    self.pos += 1;
                    v = (v <= self.shift()) as i64;
                }
                Some(TokenKind::Punct(Punct::Ge)) => {
                    self.pos += 1;
                    v = (v >= self.shift()) as i64;
                }
                _ => break,
            }
        }
        v
    }

    fn shift(&mut self) -> i64 {
        let mut v = self.additive();
        loop {
            match self.kind_at(self.pos) {
                Some(TokenKind::Punct(Punct::Shl)) => {
                    self.pos += 1;
                    let rhs = self.additive();
                    v = v.wrapping_shl(rhs as u32);
                }
                Some(TokenKind::Punct(Punct::Shr)) => {
                    self.pos += 1;
                    let rhs = self.additive();
                    v = v.wrapping_shr(rhs as u32);
                }
                _ => break,
            }
        }
        v
    }

    fn additive(&mut self) -> i64 {
        let mut v = self.multiplicative();
        loop {
            match self.kind_at(self.pos) {
                Some(TokenKind::Punct(Punct::Plus)) => {
                    self.pos += 1;
                    v = v.wrapping_add(self.multiplicative());
                }
                Some(TokenKind::Punct(Punct::Minus)) => {
                    self.pos += 1;
                    v = v.wrapping_sub(self.multiplicative());
                }
                _ => break,
            }
        }
        v
    }

    fn multiplicative(&mut self) -> i64 {
        let mut v = self.unary();
        loop {
            match self.kind_at(self.pos) {
                Some(TokenKind::Punct(Punct::Star)) => {
                    self.pos += 1;
                    v = v.wrapping_mul(self.unary());
                }
                Some(TokenKind::Punct(Punct::Slash)) => {
                    self.pos += 1;
                    let rhs = self.unary();
                    v = if rhs == 0 { self.err() } else { v / rhs };
                }
                Some(TokenKind::Punct(Punct::Percent)) => {
                    self.pos += 1;
                    let rhs = self.unary();
                    v = if rhs == 0 { self.err() } else { v % rhs };
                }
                _ => break,
            }
        }
        v
    }

    fn unary(&mut self) -> i64 {
        match self.kind_at(self.pos) {
            Some(TokenKind::Punct(Punct::Bang)) => {
                self.pos += 1;
                (self.unary() == 0) as i64
            }
            Some(TokenKind::Punct(Punct::Minus)) => {
                self.pos += 1;
                self.unary().wrapping_neg()
            }
            Some(TokenKind::Punct(Punct::Plus)) => {
                self.pos += 1;
                self.unary()
            }
            Some(TokenKind::Punct(Punct::Tilde)) => {
                self.pos += 1;
                !self.unary()
            }
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> i64 {
        match self.kind_at(self.pos).cloned() {
            Some(TokenKind::IntLit(l)) => {
                self.pos += 1;
                int_lit_value(&l)
            }
            Some(TokenKind::CharLit(c)) => {
                self.pos += 1;
                c.value as i64
            }
            Some(TokenKind::Keyword(Keyword::True)) => {
                self.pos += 1;
                1
            }
            Some(TokenKind::Keyword(Keyword::False)) => {
                self.pos += 1;
                0
            }
            // Any identifier left after macro expansion is, per the standard, not defined:
            // it evaluates to 0 rather than being an error.
            Some(TokenKind::Ident(_)) => {
                self.pos += 1;
                0
            }
            Some(TokenKind::Punct(Punct::LParen)) => {
                self.pos += 1;
                let v = self.conditional();
                if matches!(
                    self.kind_at(self.pos),
                    Some(TokenKind::Punct(Punct::RParen))
                ) {
                    self.pos += 1;
                } else {
                    return self.err();
                }
                v
            }
            _ => self.err(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenKind;

    fn run(src: &str) -> (Vec<Token>, Vec<PpError>) {
        preprocess(src, &PpOpts::default())
    }

    fn ident_names(toks: &[Token]) -> Vec<String> {
        toks.iter()
            .filter_map(|t| match &t.kind {
                TokenKind::Ident(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    fn int_values(toks: &[Token]) -> Vec<String> {
        toks.iter()
            .filter_map(|t| match &t.kind {
                TokenKind::IntLit(l) => Some(l.text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn object_like_macro_expands() {
        let (toks, errs) = run("#define FOO 42\nFOO");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(int_values(&toks), vec!["42"]);
    }

    #[test]
    fn function_like_macro_multiple_params() {
        let (toks, errs) = run("#define ADD(a, b) a + b\nADD(1, 2)");
        assert!(errs.is_empty(), "{errs:?}");
        let kinds: Vec<&TokenKind> = toks.iter().map(|t| &t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &TokenKind::IntLit(IntLit {
                    text: "1".into(),
                    base: IntBase::Dec,
                    unsigned: false,
                    long_len: 0
                }),
                &TokenKind::Punct(Punct::Plus),
                &TokenKind::IntLit(IntLit {
                    text: "2".into(),
                    base: IntBase::Dec,
                    unsigned: false,
                    long_len: 0
                }),
                &TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn self_referential_macro_does_not_hang() {
        let (toks, errs) = run("#define X X + 1\nX");
        assert!(errs.is_empty(), "{errs:?}");
        // the inner `X` is left unexpanded (painted blue), so it prints literally
        assert_eq!(ident_names(&toks), vec!["X"]);
        assert_eq!(int_values(&toks), vec!["1"]);
    }

    #[test]
    fn undef_then_reuse_as_plain_ident() {
        let (toks, errs) = run("#define FOO 1\n#undef FOO\nFOO");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["FOO"]);
    }

    #[test]
    fn ifdef_ifndef_else_endif() {
        let (toks, errs) =
            run("#define FOO\n#ifdef FOO\nA\n#else\nB\n#endif\n#ifndef FOO\nC\n#else\nD\n#endif");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["A", "D"]);
    }

    #[test]
    fn if_elif_else_with_const_expr() {
        let (toks, errs) = run("#if 1 + 1 == 2\nA\n#elif 1\nB\n#else\nC\n#endif");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["A"]);

        let (toks, errs) = run("#if 0\nA\n#elif 2 * 3 == 6\nB\n#else\nC\n#endif");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["B"]);

        let (toks, errs) = run("#if 0\nA\n#elif 0\nB\n#else\nC\n#endif");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["C"]);
    }

    #[test]
    fn defined_operator_both_forms() {
        let (toks, errs) = run("#define FOO\n#if defined(FOO) && defined BAR\nA\n#else\nB\n#endif");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["B"]);

        let (toks, errs) = run("#define FOO\n#if defined(FOO)\nA\n#else\nB\n#endif");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["A"]);
    }

    #[test]
    fn command_line_defines_seed_macro_table() {
        let opts = PpOpts {
            defines: vec![
                ("FOO".to_string(), Some("7".to_string())),
                ("BAR".to_string(), None),
            ],
            ..PpOpts::default()
        };
        let (toks, errs) = preprocess("FOO\n#ifdef BAR\nYES\n#endif", &opts);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(int_values(&toks), vec!["7"]);
        assert_eq!(ident_names(&toks), vec!["YES"]);
    }

    #[test]
    fn include_splices_tokens_and_leaves_macro_visible() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let opts = PpOpts {
            include_dirs: vec![dir],
            ..PpOpts::default()
        };
        let (toks, errs) = preprocess("#include <pp_include.h>\nINCLUDED_VALUE", &opts);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(int_values(&toks), vec!["99"]);
    }

    #[test]
    fn pragma_is_silently_accepted() {
        let (toks, errs) = run("#pragma once\n#pragma unroll 4\nX");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["X"]);
    }

    #[test]
    fn error_directive_records_error_and_continues() {
        let (toks, errs) = run("#error something went wrong\nAFTER");
        assert_eq!(errs.len(), 1);
        match &errs[0] {
            PpError::UserError(_, msg) => assert_eq!(msg, "something went wrong"),
            other => panic!("{other:?}"),
        }
        assert_eq!(ident_names(&toks), vec!["AFTER"]);
    }

    #[test]
    fn unterminated_if_at_eof_recovers() {
        let (toks, errs) = run("#if 1\nA");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], PpError::UnterminatedIf(_)));
        assert_eq!(ident_names(&toks), vec!["A"]);
    }

    #[test]
    fn function_like_macro_zero_params() {
        let (toks, errs) = run("#define ZERO() 5\nZERO()");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(int_values(&toks), vec!["5"]);
    }

    #[test]
    fn non_function_like_use_of_function_macro_name_is_plain_ident() {
        let (toks, errs) = run("#define FOO(a) a\nFOO");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(ident_names(&toks), vec!["FOO"]);
    }
}
