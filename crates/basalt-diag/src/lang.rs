// `--lang` table loader. Message strings never live in Rust source;
// they live in `lang/*.txt` tables keyed by E-code, loaded into a `LangTable` at startup and
// looked up by `Diag::render`.
//
// File format, one entry per line:
//
//   E099 = rank-2 tile requested on a backend without a matrix path
//
// - `<E-code> = <template>`, whitespace around both sides is trimmed.
// - Lines that are empty (after trim) or start with `#` are ignored.
// - `<template>` may reference a `Diag`'s positional args with `{0}`, `{1}`, ...
// - Every `<E-code>` must be a name from `ECode::ALL` and must appear at most once per file.
//
// `lang/en.txt` at the repo root is the default table, embedded at compile time so the
// compiler always has a message for every code even with no `--lang` flag given. `--lang
// <file>` loads an arbitrary table with `load_lang_file`.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use crate::diag::Diag;
use crate::ecode::ECode;

/// A loaded set of E-code -> message-template mappings.
#[derive(Debug, Clone, Default)]
pub struct LangTable {
    messages: BTreeMap<ECode, String>,
}

impl LangTable {
    /// Parses table source text per the format documented above.
    pub fn parse(source: &str) -> Result<LangTable, Diag> {
        let mut messages = BTreeMap::new();
        for (idx, raw_line) in source.lines().enumerate() {
            let lineno = idx + 1;
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (code_str, template) = line
                .split_once('=')
                .ok_or_else(|| Diag::new(ECode::LangMalformedLine).with_arg(lineno.to_string()))?;
            let code = ECode::from_code_str(code_str.trim())
                .ok_or_else(|| Diag::new(ECode::LangMalformedLine).with_arg(lineno.to_string()))?;
            let template = template.trim().to_string();
            if messages.insert(code, template).is_some() {
                return Err(Diag::new(ECode::LangDuplicateCode)
                    .with_arg(code.as_str())
                    .with_arg(lineno.to_string()));
            }
        }
        Ok(LangTable { messages })
    }

    /// Loads and parses a table from an arbitrary path — the entry point for `--lang <file>`.
    pub fn load_file(path: &Path) -> Result<LangTable, Diag> {
        let contents = fs::read_to_string(path).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                Diag::new(ECode::LangFileNotFound).with_arg(path.display().to_string())
            } else {
                Diag::new(ECode::IoError)
                    .with_arg(path.display().to_string())
                    .with_arg(e.to_string())
            }
        })?;
        Self::parse(&contents)
    }

    /// The project's default (English) table, embedded at compile time from `lang/en.txt`.
    /// Parsing this can only fail if the shipped file itself is malformed, which is a build
    /// bug caught by this crate's own tests — not a condition callers need to handle.
    pub fn default_en() -> LangTable {
        Self::parse(include_str!("../../../lang/en.txt")).expect("lang/en.txt must be well-formed")
    }

    /// The message template registered for `code`, if any.
    pub fn message(&self, code: ECode) -> Option<&str> {
        self.messages.get(&code).map(String::as_str)
    }

    /// Codes from `ECode::ALL` that have no entry in this table.
    pub fn missing_codes(&self) -> Vec<ECode> {
        ECode::ALL
            .iter()
            .copied()
            .filter(|c| !self.messages.contains_key(c))
            .collect()
    }
}

/// Loads a `--lang <file>` message table. Thin wrapper over `LangTable::load_file` kept as a
/// free function so callers (`basalt-cli`) don't need to know about the type's other
/// constructors.
pub fn load_lang_file(path: impl AsRef<Path>) -> Result<LangTable, Diag> {
    LangTable::load_file(path.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_table() {
        let table = LangTable::parse(
            "# a comment\n\nE090 = backend refusal\nE099 = matrix path unsupported\n",
        )
        .unwrap();
        assert_eq!(table.message(ECode::UnsupportedOp), Some("backend refusal"));
        assert_eq!(
            table.message(ECode::MatrixPathUnsupported),
            Some("matrix path unsupported")
        );
    }

    #[test]
    fn loads_default_en_table_with_every_code_covered() {
        let table = LangTable::default_en();
        let missing = table.missing_codes();
        assert!(
            missing.is_empty(),
            "lang/en.txt is missing entries for: {missing:?}"
        );
    }

    #[test]
    fn malformed_line_reports_e502_with_line_number() {
        let err = LangTable::parse("this line has no separator\n").unwrap_err();
        assert_eq!(err.code, ECode::LangMalformedLine);
        assert_eq!(err.args, vec!["1".to_string()]);
    }

    #[test]
    fn unknown_code_reports_e502() {
        let err = LangTable::parse("E999 = some message\n").unwrap_err();
        assert_eq!(err.code, ECode::LangMalformedLine);
    }

    #[test]
    fn duplicate_code_reports_e503() {
        let err = LangTable::parse("E090 = a\nE090 = b\n").unwrap_err();
        assert_eq!(err.code, ECode::LangDuplicateCode);
    }

    #[test]
    fn missing_file_reports_e501() {
        let err =
            LangTable::load_file(Path::new("/nonexistent/path/basalt-lang-test.txt")).unwrap_err();
        assert_eq!(err.code, ECode::LangFileNotFound);
    }

    #[test]
    fn load_lang_file_wrapper_round_trips_default_table() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../lang/en.txt");
        let table = load_lang_file(path).unwrap();
        assert!(table.missing_codes().is_empty());
    }
}
