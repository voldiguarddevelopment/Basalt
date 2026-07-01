// The `Diag` type: an E-code plus an optional source location plus positional format
// arguments. `Diag` never carries message text itself — rendering a user-facing string
// requires a `LangTable` (see lang.rs) so that swapping `--lang` never touches this type.

use std::fmt;

use crate::ecode::ECode;
use crate::lang::LangTable;

/// A single point in source text, 1-based (matches editor conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Loc {
    pub line: u32,
    pub col: u32,
}

impl Loc {
    pub fn new(line: u32, col: u32) -> Loc {
        Loc { line, col }
    }
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// A source range. `start == end` for a single-point diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub start: Loc,
    pub end: Loc,
}

impl Span {
    pub fn new(start: Loc, end: Loc) -> Span {
        Span { start, end }
    }

    pub fn point(loc: Loc) -> Span {
        Span {
            start: loc,
            end: loc,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == self.end {
            write!(f, "{}", self.start)
        } else {
            write!(f, "{}-{}", self.start, self.end)
        }
    }
}

/// A diagnostic: a stable code, an optional location, and positional arguments for message
/// interpolation. Construct with `Diag::new`, attach a span/args with the builder methods,
/// then `render` against a loaded `LangTable` to get the user-facing string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diag {
    pub code: ECode,
    pub span: Option<Span>,
    pub args: Vec<String>,
}

impl Diag {
    pub fn new(code: ECode) -> Diag {
        Diag {
            code,
            span: None,
            args: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_span(mut self, span: Span) -> Diag {
        self.span = Some(span);
        self
    }

    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Diag {
        self.args.push(arg.into());
        self
    }

    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Diag
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Renders this diagnostic to a user-facing string using `table`'s message template for
    /// `self.code`. Falls back to a placeholder if the table has no entry for the code
    /// (rather than panicking — a stale table should never crash the compiler).
    pub fn render(&self, table: &LangTable) -> String {
        let template = table
            .message(self.code)
            .unwrap_or("<no message registered for this E-code>");
        let body = interpolate(template, &self.args);
        match &self.span {
            Some(span) => format!("{} ({span}): {body}", self.code.as_str()),
            None => format!("{}: {body}", self.code.as_str()),
        }
    }
}

impl fmt::Display for Diag {
    /// Code-only rendering, for contexts without a `LangTable` (e.g. plumbing `Diag` through
    /// `Result<_, Diag>` as a plain `std::error::Error`). User-facing output should call
    /// `render` instead.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code.as_str())
    }
}

impl std::error::Error for Diag {}

/// Substitutes `{0}`, `{1}`, ... in `template` with the corresponding entry of `args`. An
/// out-of-range or non-numeric placeholder is left as literal text.
fn interpolate(template: &str, args: &[String]) -> String {
    let chars: Vec<char> = template.chars().collect();
    let mut out = String::with_capacity(template.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            if let Some(rel_close) = chars[i..].iter().position(|&c| c == '}') {
                let close = i + rel_close;
                let idx_str: String = chars[i + 1..close].iter().collect();
                if let Ok(idx) = idx_str.parse::<usize>() {
                    match args.get(idx) {
                        Some(a) => out.push_str(a),
                        None => {
                            out.push('{');
                            out.push_str(&idx_str);
                            out.push('}');
                        }
                    }
                    i = close + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::LangTable;

    #[test]
    fn renders_with_placeholder_interpolation() {
        let table = LangTable::parse("E100 = unknown flag: {0}\n").unwrap();
        let d = Diag::new(ECode::CliUnknownFlag).with_arg("--frobnicate");
        assert_eq!(d.render(&table), "E100: unknown flag: --frobnicate");
    }

    #[test]
    fn renders_span_when_present() {
        let table = LangTable::parse("E301 = undefined symbol: {0}\n").unwrap();
        let d = Diag::new(ECode::UndefinedSymbol)
            .with_arg("foo")
            .with_span(Span::point(Loc::new(3, 7)));
        assert_eq!(d.render(&table), "E301 (3:7): undefined symbol: foo");
    }

    #[test]
    fn missing_table_entry_falls_back_instead_of_panicking() {
        let table = LangTable::parse("").unwrap();
        let d = Diag::new(ECode::TypeError);
        assert!(d.render(&table).starts_with("E300:"));
    }

    #[test]
    fn out_of_range_placeholder_is_left_literal() {
        let table = LangTable::parse("E301 = undefined symbol: {5}\n").unwrap();
        let d = Diag::new(ECode::UndefinedSymbol);
        assert_eq!(d.render(&table), "E301: undefined symbol: {5}");
    }
}
