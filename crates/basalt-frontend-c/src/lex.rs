// The scanner: turns source text into a `Vec<Token>` (see `token.rs` for the token model).
//
// Errors never abort the scan — `lex` always runs to `Eof` and returns whatever `LexError`s
// it collected alongside the tokens, so a caller can report every problem in a file in one
// pass rather than stopping at the first one. Each error variant documents its own recovery
// rule; the one exception is an unterminated block comment, which has nowhere left to
// recover to and consumes the rest of the file.

use std::fmt;

use crate::token::{
    CharLit, FloatLit, FloatSuffix, IntBase, IntLit, Keyword, Loc, Punct, Span, StrLit, Token,
    TokenKind,
};

/// A problem found while scanning. Every variant carries the `Span` of the offending text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexError {
    UnterminatedString(Span),
    UnterminatedChar(Span),
    UnterminatedComment(Span),
    /// Empty (`''`) or otherwise malformed character literal.
    BadCharLit(Span),
    InvalidEscape(Span),
    /// A numeric literal whose digits, base, or suffix don't form a valid C literal shape.
    BadNumber(Span),
    /// A byte that doesn't start any recognized token.
    UnexpectedChar(Span),
}

impl LexError {
    pub fn span(&self) -> Span {
        match *self {
            LexError::UnterminatedString(s)
            | LexError::UnterminatedChar(s)
            | LexError::UnterminatedComment(s)
            | LexError::BadCharLit(s)
            | LexError::InvalidEscape(s)
            | LexError::BadNumber(s)
            | LexError::UnexpectedChar(s) => s,
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexError::UnterminatedString(s) => write!(f, "unterminated string literal ({s})"),
            LexError::UnterminatedChar(s) => write!(f, "unterminated character literal ({s})"),
            LexError::UnterminatedComment(s) => write!(f, "unterminated block comment ({s})"),
            LexError::BadCharLit(s) => write!(f, "invalid character literal ({s})"),
            LexError::InvalidEscape(s) => write!(f, "invalid escape sequence ({s})"),
            LexError::BadNumber(s) => write!(f, "invalid numeric literal ({s})"),
            LexError::UnexpectedChar(s) => write!(f, "unexpected character ({s})"),
        }
    }
}

impl std::error::Error for LexError {}

/// Scans `src` in full, returning every token (always Eof-terminated) plus every error hit
/// along the way.
pub fn lex(src: &str) -> (Vec<Token>, Vec<LexError>) {
    let mut lx = Lexer::new(src);
    lx.run();
    (lx.tokens, lx.errors)
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    line: u32,
    col: u32,
    tokens: Vec<Token>,
    errors: Vec<LexError>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src,
            pos: 0,
            line: 1,
            col: 1,
            tokens: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn loc(&self) -> Loc {
        Loc::new(self.pos as u32, self.line, self.col)
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek_at(&self, ahead: usize) -> Option<char> {
        self.src[self.pos..].chars().nth(ahead)
    }

    /// Advances past the current char, keeping `line`/`col` in sync. Column counts by char,
    /// not byte, so multi-byte UTF-8 doesn't distort it.
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn run(&mut self) {
        loop {
            self.skip_trivia();
            let start = self.loc();
            let Some(c) = self.peek() else {
                self.tokens
                    .push(Token::new(TokenKind::Eof, Span::new(start, start)));
                break;
            };
            if is_ident_start(c) {
                self.scan_ident(start);
            } else if c.is_ascii_digit()
                || (c == '.' && matches!(self.peek_at(1), Some(d) if d.is_ascii_digit()))
            {
                self.scan_number(start);
            } else if c == '\'' {
                self.scan_char(start);
            } else if c == '"' {
                self.scan_string(start);
            } else if let Some(p) = self.scan_punct() {
                let end = self.loc();
                self.tokens
                    .push(Token::new(TokenKind::Punct(p), Span::new(start, end)));
            } else {
                self.bump();
                let end = self.loc();
                self.errors
                    .push(LexError::UnexpectedChar(Span::new(start, end)));
            }
        }
    }

    /// Skips whitespace and comments. Line comments run to (not including) the newline;
    /// block comments may span multiple lines and, if never closed, consume to EOF.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek_at(1) == Some('/') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some('/') if self.peek_at(1) == Some('*') => {
                    let start = self.loc();
                    self.bump();
                    self.bump();
                    let mut closed = false;
                    while self.peek().is_some() {
                        if self.peek() == Some('*') && self.peek_at(1) == Some('/') {
                            self.bump();
                            self.bump();
                            closed = true;
                            break;
                        }
                        self.bump();
                    }
                    if !closed {
                        let end = self.loc();
                        self.errors
                            .push(LexError::UnterminatedComment(Span::new(start, end)));
                    }
                }
                _ => break,
            }
        }
    }

    fn scan_ident(&mut self, start: Loc) {
        let start_byte = self.pos;
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.bump();
            } else {
                break;
            }
        }
        let text = self.src[start_byte..self.pos].to_string();
        let end = self.loc();
        let kind = match Keyword::from_str(&text) {
            Some(k) => TokenKind::Keyword(k),
            None => TokenKind::Ident(text),
        };
        self.tokens.push(Token::new(kind, Span::new(start, end)));
    }

    /// Consumes the `u`/`U`/`l`/`L` run right after a digit sequence and classifies it. An
    /// unrecognized combination is reported once and decoded on a best-effort basis (any `u`
    /// present sets unsigned; up to two `l`s set the long length) so the token stream still
    /// has something plausible to hand the parser.
    fn scan_int_suffix(&mut self, start: Loc) -> (bool, u8) {
        let run_start = self.pos;
        while let Some(c) = self.peek() {
            if matches!(c, 'u' | 'U' | 'l' | 'L') {
                self.bump();
            } else {
                break;
            }
        }
        let raw = &self.src[run_start..self.pos];
        let upper: String = raw.chars().map(|c| c.to_ascii_uppercase()).collect();
        match upper.as_str() {
            "" => (false, 0),
            "U" => (true, 0),
            "L" => (false, 1),
            "LL" => (false, 2),
            "UL" | "LU" => (true, 1),
            "ULL" | "LLU" => (true, 2),
            _ => {
                self.errors
                    .push(LexError::BadNumber(Span::new(start, self.loc())));
                let unsigned = upper.contains('U');
                let long_len = upper.chars().filter(|&c| c == 'L').count().min(2) as u8;
                (unsigned, long_len)
            }
        }
    }

    /// A numeric literal directly followed by an identifier char (`123abc`) has no valid
    /// shape in C. Report it once and fold the trailing run into this token so it doesn't
    /// reappear as a bogus, unrelated identifier.
    fn check_trailing_ident(&mut self, start: Loc) {
        if let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.errors
                    .push(LexError::BadNumber(Span::new(start, self.loc())));
                while let Some(c) = self.peek() {
                    if is_ident_continue(c) {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    fn scan_number(&mut self, start: Loc) {
        let start_byte = self.pos;
        let leading_zero = self.peek() == Some('0');

        if leading_zero && matches!(self.peek_at(1), Some('x' | 'X')) {
            self.bump();
            self.bump();
            let digits_start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() {
                    self.bump();
                } else {
                    break;
                }
            }
            if self.pos == digits_start {
                self.errors
                    .push(LexError::BadNumber(Span::new(start, self.loc())));
            }
            let (unsigned, long_len) = self.scan_int_suffix(start);
            self.check_trailing_ident(start);
            let text = self.src[start_byte..self.pos].to_string();
            let end = self.loc();
            let lit = IntLit {
                text,
                base: IntBase::Hex,
                unsigned,
                long_len,
            };
            self.tokens
                .push(Token::new(TokenKind::IntLit(lit), Span::new(start, end)));
            return;
        }

        if leading_zero && matches!(self.peek_at(1), Some('b' | 'B')) {
            self.bump();
            self.bump();
            let digits_start = self.pos;
            while matches!(self.peek(), Some('0' | '1')) {
                self.bump();
            }
            if self.pos == digits_start {
                self.errors
                    .push(LexError::BadNumber(Span::new(start, self.loc())));
            }
            let (unsigned, long_len) = self.scan_int_suffix(start);
            self.check_trailing_ident(start);
            let text = self.src[start_byte..self.pos].to_string();
            let end = self.loc();
            let lit = IntLit {
                text,
                base: IntBase::Bin,
                unsigned,
                long_len,
            };
            self.tokens
                .push(Token::new(TokenKind::IntLit(lit), Span::new(start, end)));
            return;
        }

        if self.peek() != Some('.') {
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
        }

        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.bump();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
        }

        if matches!(self.peek(), Some('e' | 'E')) {
            let (has_sign, digit_after) = match self.peek_at(1) {
                Some('+') | Some('-') => {
                    (true, self.peek_at(2).is_some_and(|d| d.is_ascii_digit()))
                }
                Some(d) if d.is_ascii_digit() => (false, true),
                _ => (false, false),
            };
            if digit_after {
                is_float = true;
                self.bump();
                if has_sign {
                    self.bump();
                }
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
        }

        if is_float {
            let suffix = match self.peek() {
                Some('f' | 'F') => {
                    self.bump();
                    FloatSuffix::F
                }
                Some('l' | 'L') => {
                    self.bump();
                    FloatSuffix::L
                }
                _ => FloatSuffix::None,
            };
            self.check_trailing_ident(start);
            let text = self.src[start_byte..self.pos].to_string();
            let end = self.loc();
            self.tokens.push(Token::new(
                TokenKind::FloatLit(FloatLit { text, suffix }),
                Span::new(start, end),
            ));
        } else {
            let int_text = &self.src[start_byte..self.pos];
            let base = if leading_zero && int_text.len() > 1 {
                IntBase::Oct
            } else {
                IntBase::Dec
            };
            if base == IntBase::Oct && int_text.bytes().any(|b| b == b'8' || b == b'9') {
                self.errors
                    .push(LexError::BadNumber(Span::new(start, self.loc())));
            }
            let (unsigned, long_len) = self.scan_int_suffix(start);
            self.check_trailing_ident(start);
            let text = self.src[start_byte..self.pos].to_string();
            let end = self.loc();
            let lit = IntLit {
                text,
                base,
                unsigned,
                long_len,
            };
            self.tokens
                .push(Token::new(TokenKind::IntLit(lit), Span::new(start, end)));
        }
    }

    /// Decodes one escape sequence, having already consumed the backslash at `bs_loc`. An
    /// escape this lexer doesn't recognize is reported and recovered from by treating the
    /// character after the backslash as a literal (i.e. the backslash is dropped).
    fn scan_escape(&mut self, bs_loc: Loc) -> u32 {
        match self.peek() {
            Some('n') => {
                self.bump();
                b'\n' as u32
            }
            Some('t') => {
                self.bump();
                b'\t' as u32
            }
            Some('r') => {
                self.bump();
                b'\r' as u32
            }
            Some('a') => {
                self.bump();
                7
            }
            Some('b') => {
                self.bump();
                8
            }
            Some('f') => {
                self.bump();
                12
            }
            Some('v') => {
                self.bump();
                11
            }
            Some('\\') => {
                self.bump();
                b'\\' as u32
            }
            Some('\'') => {
                self.bump();
                b'\'' as u32
            }
            Some('"') => {
                self.bump();
                b'"' as u32
            }
            Some('x') => {
                self.bump();
                let mut val: u32 = 0;
                let mut n = 0;
                while let Some(c) = self.peek() {
                    if let Some(d) = c.to_digit(16) {
                        val = val.wrapping_mul(16).wrapping_add(d);
                        self.bump();
                        n += 1;
                    } else {
                        break;
                    }
                }
                if n == 0 {
                    self.errors
                        .push(LexError::InvalidEscape(Span::new(bs_loc, self.loc())));
                }
                val
            }
            Some(c) if c.is_digit(8) => {
                let mut val: u32 = 0;
                let mut n = 0;
                while n < 3 {
                    match self.peek().and_then(|c| c.to_digit(8)) {
                        Some(d) => {
                            val = val * 8 + d;
                            self.bump();
                            n += 1;
                        }
                        None => break,
                    }
                }
                val
            }
            Some(c) => {
                self.errors
                    .push(LexError::InvalidEscape(Span::new(bs_loc, self.loc())));
                self.bump();
                c as u32
            }
            None => {
                self.errors
                    .push(LexError::InvalidEscape(Span::new(bs_loc, self.loc())));
                0
            }
        }
    }

    fn scan_char(&mut self, start: Loc) {
        let open_byte = self.pos;
        self.bump();
        let mut first: Option<u32> = None;
        let mut count = 0u32;
        loop {
            match self.peek() {
                None => {
                    let end = self.loc();
                    self.errors
                        .push(LexError::UnterminatedChar(Span::new(start, end)));
                    break;
                }
                Some('\n') => {
                    let end = self.loc();
                    self.errors
                        .push(LexError::UnterminatedChar(Span::new(start, end)));
                    self.bump();
                    break;
                }
                Some('\'') => {
                    if count == 0 {
                        self.bump();
                        let end = self.loc();
                        self.errors
                            .push(LexError::BadCharLit(Span::new(start, end)));
                    } else {
                        self.bump();
                    }
                    break;
                }
                Some('\\') => {
                    let bs_loc = self.loc();
                    self.bump();
                    let v = self.scan_escape(bs_loc);
                    if first.is_none() {
                        first = Some(v);
                    }
                    count += 1;
                }
                Some(c) => {
                    self.bump();
                    if first.is_none() {
                        first = Some(c as u32);
                    }
                    count += 1;
                }
            }
        }
        let raw = self.src[open_byte..self.pos].to_string();
        let end = self.loc();
        let lit = CharLit {
            value: first.unwrap_or(0),
            raw,
        };
        self.tokens
            .push(Token::new(TokenKind::CharLit(lit), Span::new(start, end)));
    }

    fn scan_string(&mut self, start: Loc) {
        let open_byte = self.pos;
        self.bump();
        let mut value = String::new();
        loop {
            match self.peek() {
                None => {
                    let end = self.loc();
                    self.errors
                        .push(LexError::UnterminatedString(Span::new(start, end)));
                    break;
                }
                Some('\n') => {
                    let end = self.loc();
                    self.errors
                        .push(LexError::UnterminatedString(Span::new(start, end)));
                    self.bump();
                    break;
                }
                Some('"') => {
                    self.bump();
                    break;
                }
                Some('\\') => {
                    let bs_loc = self.loc();
                    self.bump();
                    let v = self.scan_escape(bs_loc);
                    value.push(char::from_u32(v).unwrap_or('\u{fffd}'));
                }
                Some(c) => {
                    self.bump();
                    value.push(c);
                }
            }
        }
        let raw = self.src[open_byte..self.pos].to_string();
        let end = self.loc();
        let lit = StrLit { value, raw };
        self.tokens
            .push(Token::new(TokenKind::StrLit(lit), Span::new(start, end)));
    }

    /// Maximal-munch scan of one operator/punctuation token. Longer forms are always tried
    /// before their prefixes (`<<=` before `<<` before `<`, `...` before `.`, etc).
    fn scan_punct(&mut self) -> Option<Punct> {
        let c0 = self.peek()?;
        let c1 = self.peek_at(1);
        let c2 = self.peek_at(2);
        let p = match c0 {
            '+' => match c1 {
                Some('+') => {
                    self.bump();
                    self.bump();
                    Punct::PlusPlus
                }
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::PlusEq
                }
                _ => {
                    self.bump();
                    Punct::Plus
                }
            },
            '-' => match c1 {
                Some('-') => {
                    self.bump();
                    self.bump();
                    Punct::MinusMinus
                }
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::MinusEq
                }
                Some('>') => {
                    self.bump();
                    self.bump();
                    Punct::Arrow
                }
                _ => {
                    self.bump();
                    Punct::Minus
                }
            },
            '*' => match c1 {
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::StarEq
                }
                _ => {
                    self.bump();
                    Punct::Star
                }
            },
            '/' => match c1 {
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::SlashEq
                }
                _ => {
                    self.bump();
                    Punct::Slash
                }
            },
            '%' => match c1 {
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::PercentEq
                }
                _ => {
                    self.bump();
                    Punct::Percent
                }
            },
            '&' => match c1 {
                Some('&') => {
                    self.bump();
                    self.bump();
                    Punct::AmpAmp
                }
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::AmpEq
                }
                _ => {
                    self.bump();
                    Punct::Amp
                }
            },
            '|' => match c1 {
                Some('|') => {
                    self.bump();
                    self.bump();
                    Punct::PipePipe
                }
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::PipeEq
                }
                _ => {
                    self.bump();
                    Punct::Pipe
                }
            },
            '^' => match c1 {
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::CaretEq
                }
                _ => {
                    self.bump();
                    Punct::Caret
                }
            },
            '~' => {
                self.bump();
                Punct::Tilde
            }
            '!' => match c1 {
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::NotEq
                }
                _ => {
                    self.bump();
                    Punct::Bang
                }
            },
            '<' => match (c1, c2) {
                (Some('<'), Some('=')) => {
                    self.bump();
                    self.bump();
                    self.bump();
                    Punct::ShlEq
                }
                (Some('<'), _) => {
                    self.bump();
                    self.bump();
                    Punct::Shl
                }
                (Some('='), _) => {
                    self.bump();
                    self.bump();
                    Punct::Le
                }
                _ => {
                    self.bump();
                    Punct::Lt
                }
            },
            '>' => match (c1, c2) {
                (Some('>'), Some('=')) => {
                    self.bump();
                    self.bump();
                    self.bump();
                    Punct::ShrEq
                }
                (Some('>'), _) => {
                    self.bump();
                    self.bump();
                    Punct::Shr
                }
                (Some('='), _) => {
                    self.bump();
                    self.bump();
                    Punct::Ge
                }
                _ => {
                    self.bump();
                    Punct::Gt
                }
            },
            '=' => match c1 {
                Some('=') => {
                    self.bump();
                    self.bump();
                    Punct::EqEq
                }
                _ => {
                    self.bump();
                    Punct::Eq
                }
            },
            '(' => {
                self.bump();
                Punct::LParen
            }
            ')' => {
                self.bump();
                Punct::RParen
            }
            '{' => {
                self.bump();
                Punct::LBrace
            }
            '}' => {
                self.bump();
                Punct::RBrace
            }
            '[' => {
                self.bump();
                Punct::LBracket
            }
            ']' => {
                self.bump();
                Punct::RBracket
            }
            ',' => {
                self.bump();
                Punct::Comma
            }
            ';' => {
                self.bump();
                Punct::Semi
            }
            ':' => match c1 {
                Some(':') => {
                    self.bump();
                    self.bump();
                    Punct::ColonColon
                }
                _ => {
                    self.bump();
                    Punct::Colon
                }
            },
            '?' => {
                self.bump();
                Punct::Question
            }
            '.' => match (c1, c2) {
                (Some('.'), Some('.')) => {
                    self.bump();
                    self.bump();
                    self.bump();
                    Punct::Ellipsis
                }
                _ => {
                    self.bump();
                    Punct::Dot
                }
            },
            '#' => match c1 {
                Some('#') => {
                    self.bump();
                    self.bump();
                    Punct::HashHash
                }
                _ => {
                    self.bump();
                    Punct::Hash
                }
            },
            _ => return None,
        };
        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Keyword;

    fn kinds(src: &str) -> Vec<TokenKind> {
        let (toks, errs) = lex(src);
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        toks.into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn identifiers_and_cuda_builtins_are_plain_idents() {
        let (toks, errs) = lex("foo _bar123 threadIdx __global__ blockDim");
        assert!(errs.is_empty());
        let names: Vec<&str> = toks
            .iter()
            .filter_map(|t| match &t.kind {
                TokenKind::Ident(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            names,
            vec!["foo", "_bar123", "threadIdx", "__global__", "blockDim"]
        );
    }

    #[test]
    fn every_keyword_round_trips() {
        let words = [
            "alignas",
            "alignof",
            "auto",
            "bool",
            "break",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "constexpr",
            "const_cast",
            "continue",
            "decltype",
            "default",
            "delete",
            "do",
            "double",
            "dynamic_cast",
            "else",
            "enum",
            "explicit",
            "extern",
            "false",
            "float",
            "for",
            "friend",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "mutable",
            "namespace",
            "new",
            "noexcept",
            "nullptr",
            "operator",
            "private",
            "protected",
            "public",
            "register",
            "reinterpret_cast",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "static_assert",
            "static_cast",
            "struct",
            "switch",
            "template",
            "this",
            "thread_local",
            "throw",
            "true",
            "try",
            "typedef",
            "typeid",
            "typename",
            "union",
            "unsigned",
            "using",
            "virtual",
            "void",
            "volatile",
            "wchar_t",
            "while",
        ];
        for w in words {
            let (toks, errs) = lex(w);
            assert!(errs.is_empty(), "{w}: {errs:?}");
            assert_eq!(toks.len(), 2, "{w}: {toks:?}");
            match &toks[0].kind {
                TokenKind::Keyword(k) => assert_eq!(Some(*k), Keyword::from_str(w), "{w}"),
                other => panic!("{w} lexed as {other:?}, not a keyword"),
            }
        }
    }

    #[test]
    fn decimal_int() {
        let (toks, errs) = lex("42");
        assert!(errs.is_empty());
        match &toks[0].kind {
            TokenKind::IntLit(l) => {
                assert_eq!(l.text, "42");
                assert_eq!(l.base, IntBase::Dec);
                assert!(!l.unsigned);
                assert_eq!(l.long_len, 0);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn hex_int() {
        let (toks, errs) = lex("0xFF");
        assert!(errs.is_empty());
        match &toks[0].kind {
            TokenKind::IntLit(l) => {
                assert_eq!(l.text, "0xFF");
                assert_eq!(l.base, IntBase::Hex);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn octal_int() {
        let (toks, errs) = lex("0755");
        assert!(errs.is_empty());
        match &toks[0].kind {
            TokenKind::IntLit(l) => {
                assert_eq!(l.text, "0755");
                assert_eq!(l.base, IntBase::Oct);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn binary_int() {
        let (toks, errs) = lex("0b1010");
        assert!(errs.is_empty());
        match &toks[0].kind {
            TokenKind::IntLit(l) => {
                assert_eq!(l.text, "0b1010");
                assert_eq!(l.base, IntBase::Bin);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn every_int_suffix_combination() {
        let cases: &[(&str, bool, u8)] = &[
            ("1", false, 0),
            ("1u", true, 0),
            ("1U", true, 0),
            ("1l", false, 1),
            ("1L", false, 1),
            ("1ll", false, 2),
            ("1LL", false, 2),
            ("1ul", true, 1),
            ("1UL", true, 1),
            ("1lu", true, 1),
            ("1LU", true, 1),
            ("1ull", true, 2),
            ("1ULL", true, 2),
            ("1llu", true, 2),
            ("1LLU", true, 2),
        ];
        for (src, unsigned, long_len) in cases {
            let (toks, errs) = lex(src);
            assert!(errs.is_empty(), "{src}: {errs:?}");
            match &toks[0].kind {
                TokenKind::IntLit(l) => {
                    assert_eq!(l.unsigned, *unsigned, "{src}");
                    assert_eq!(l.long_len, *long_len, "{src}");
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }

    #[test]
    fn float_literals() {
        let cases: &[(&str, FloatSuffix)] = &[
            ("1.5", FloatSuffix::None),
            (".5", FloatSuffix::None),
            ("5.", FloatSuffix::None),
            ("1.5e10", FloatSuffix::None),
            ("1.5E10", FloatSuffix::None),
            ("5e-3", FloatSuffix::None),
            ("5e+3", FloatSuffix::None),
            ("1.5f", FloatSuffix::F),
            ("1.5F", FloatSuffix::F),
            ("1.5l", FloatSuffix::L),
            ("1.5L", FloatSuffix::L),
        ];
        for (src, suffix) in cases {
            let (toks, errs) = lex(src);
            assert!(errs.is_empty(), "{src}: {errs:?}");
            match &toks[0].kind {
                TokenKind::FloatLit(l) => {
                    assert_eq!(l.text, *src);
                    assert_eq!(l.suffix, *suffix);
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }

    #[test]
    fn char_lit_escapes() {
        let cases: &[(&str, u32)] = &[
            (r"'a'", 'a' as u32),
            (r"'\n'", 10),
            (r"'\t'", 9),
            (r"'\r'", 13),
            (r"'\\'", 92),
            (r"'\''", 39),
            (r#"'\"'"#, 34),
            (r"'\0'", 0),
            (r"'\x41'", 0x41),
            (r"'\101'", 0o101),
        ];
        for (src, value) in cases {
            let (toks, errs) = lex(src);
            assert!(errs.is_empty(), "{src}: {errs:?}");
            match &toks[0].kind {
                TokenKind::CharLit(l) => {
                    assert_eq!(l.value, *value, "{src}");
                    assert_eq!(l.raw, *src);
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }

    #[test]
    fn string_lit_escapes() {
        let (toks, errs) = lex(r#""hi\n\t\\\"there""#);
        assert!(errs.is_empty());
        match &toks[0].kind {
            TokenKind::StrLit(l) => {
                assert_eq!(l.value, "hi\n\t\\\"there");
                assert_eq!(l.raw, r#""hi\n\t\\\"there""#);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn empty_char_lit_errors() {
        let (toks, errs) = lex("''");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::BadCharLit(_)));
        assert!(matches!(toks[0].kind, TokenKind::CharLit(_)));
    }

    #[test]
    fn unterminated_string_recovers_at_newline() {
        let (toks, errs) = lex("\"abc\nint x;");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::UnterminatedString(_)));
        // lexing continues cleanly on the next line
        assert!(matches!(toks[1].kind, TokenKind::Keyword(Keyword::Int)));
        assert!(matches!(toks[2].kind, TokenKind::Ident(_)));
        assert!(matches!(toks[3].kind, TokenKind::Punct(Punct::Semi)));
    }

    #[test]
    fn unterminated_char_at_eof() {
        let (_, errs) = lex("'a");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::UnterminatedChar(_)));
    }

    #[test]
    fn bad_escape_recovers() {
        let (toks, errs) = lex(r"'\q'");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::InvalidEscape(_)));
        match &toks[0].kind {
            TokenKind::CharLit(l) => assert_eq!(l.value, 'q' as u32),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn bad_number_suffix_recovers_and_keeps_lexing() {
        let (toks, errs) = lex("1uu + 2");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::BadNumber(_)));
        assert!(matches!(toks[0].kind, TokenKind::IntLit(_)));
        assert!(matches!(toks[1].kind, TokenKind::Punct(Punct::Plus)));
        match &toks[2].kind {
            TokenKind::IntLit(l) => assert_eq!(l.text, "2"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn line_comment_is_discarded() {
        assert_eq!(
            kinds("// comment\n42"),
            vec![
                TokenKind::IntLit(IntLit {
                    text: "42".into(),
                    base: IntBase::Dec,
                    unsigned: false,
                    long_len: 0
                }),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn block_comment_single_and_multi_line() {
        assert_eq!(
            kinds("/* x */42"),
            vec![
                TokenKind::IntLit(IntLit {
                    text: "42".into(),
                    base: IntBase::Dec,
                    unsigned: false,
                    long_len: 0
                }),
                TokenKind::Eof,
            ]
        );
        assert_eq!(
            kinds("/* line1\nline2\nline3 */42"),
            vec![
                TokenKind::IntLit(IntLit {
                    text: "42".into(),
                    base: IntBase::Dec,
                    unsigned: false,
                    long_len: 0
                }),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn unterminated_block_comment_does_not_hang() {
        let (toks, errs) = lex("/* never closed");
        assert_eq!(errs.len(), 1);
        assert!(matches!(errs[0], LexError::UnterminatedComment(_)));
        assert_eq!(toks.len(), 1);
        assert!(matches!(toks[0].kind, TokenKind::Eof));
    }

    #[test]
    fn every_punct_variant() {
        let cases: &[(&str, Punct)] = &[
            ("+", Punct::Plus),
            ("-", Punct::Minus),
            ("*", Punct::Star),
            ("/", Punct::Slash),
            ("%", Punct::Percent),
            ("++", Punct::PlusPlus),
            ("--", Punct::MinusMinus),
            ("+=", Punct::PlusEq),
            ("-=", Punct::MinusEq),
            ("*=", Punct::StarEq),
            ("/=", Punct::SlashEq),
            ("%=", Punct::PercentEq),
            ("&", Punct::Amp),
            ("|", Punct::Pipe),
            ("^", Punct::Caret),
            ("~", Punct::Tilde),
            ("&&", Punct::AmpAmp),
            ("||", Punct::PipePipe),
            ("!", Punct::Bang),
            ("&=", Punct::AmpEq),
            ("|=", Punct::PipeEq),
            ("^=", Punct::CaretEq),
            ("<<", Punct::Shl),
            (">>", Punct::Shr),
            ("<<=", Punct::ShlEq),
            (">>=", Punct::ShrEq),
            ("=", Punct::Eq),
            ("==", Punct::EqEq),
            ("!=", Punct::NotEq),
            ("<", Punct::Lt),
            (">", Punct::Gt),
            ("<=", Punct::Le),
            (">=", Punct::Ge),
            ("(", Punct::LParen),
            (")", Punct::RParen),
            ("{", Punct::LBrace),
            ("}", Punct::RBrace),
            ("[", Punct::LBracket),
            ("]", Punct::RBracket),
            (",", Punct::Comma),
            (";", Punct::Semi),
            (":", Punct::Colon),
            ("::", Punct::ColonColon),
            ("?", Punct::Question),
            (".", Punct::Dot),
            ("...", Punct::Ellipsis),
            ("->", Punct::Arrow),
            ("#", Punct::Hash),
            ("##", Punct::HashHash),
        ];
        for (src, expect) in cases {
            let (toks, errs) = lex(src);
            assert!(errs.is_empty(), "{src}: {errs:?}");
            assert_eq!(toks.len(), 2, "{src}: {toks:?}");
            match &toks[0].kind {
                TokenKind::Punct(p) => assert_eq!(p, expect, "{src}"),
                other => panic!("{src}: {other:?}"),
            }
        }
    }

    #[test]
    fn maximal_munch_shift_assign_does_not_split() {
        assert_eq!(
            kinds("<<="),
            vec![TokenKind::Punct(Punct::ShlEq), TokenKind::Eof]
        );
        assert_eq!(
            kinds("..."),
            vec![TokenKind::Punct(Punct::Ellipsis), TokenKind::Eof]
        );
        // three dots not present -> two separate tokens, not an Ellipsis
        assert_eq!(
            kinds(". ."),
            vec![
                TokenKind::Punct(Punct::Dot),
                TokenKind::Punct(Punct::Dot),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn loc_tracks_line_and_col_across_lines() {
        let (toks, errs) = lex("int x;\nfloat y;");
        assert!(errs.is_empty());
        // `float` starts on line 2, column 1
        let float_tok = toks
            .iter()
            .find(|t| matches!(t.kind, TokenKind::Keyword(Keyword::Float)))
            .unwrap();
        assert_eq!(float_tok.span.start.line, 2);
        assert_eq!(float_tok.span.start.col, 1);
        // `y` follows at column 7
        let y_tok = toks
            .iter()
            .find(|t| matches!(&t.kind, TokenKind::Ident(s) if s == "y"))
            .unwrap();
        assert_eq!(y_tok.span.start.line, 2);
        assert_eq!(y_tok.span.start.col, 7);
    }

    #[test]
    fn multibyte_utf8_in_string_and_comment() {
        let (toks, errs) = lex("\"caf\u{e9} \u{1f600}\" // \u{2603}\n1");
        assert!(errs.is_empty());
        match &toks[0].kind {
            TokenKind::StrLit(l) => assert_eq!(l.value, "caf\u{e9} \u{1f600}"),
            other => panic!("{other:?}"),
        }
        match &toks[1].kind {
            TokenKind::IntLit(l) => assert_eq!(l.text, "1"),
            other => panic!("{other:?}"),
        }
    }
}
