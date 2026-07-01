// Stable E-code registry. Every user-facing diagnostic in the project carries one of these
// codes; tests assert on the code, never on message text. The
// string form ("E099" etc.) is the contract — do not renumber an existing variant, only add
// new ones. Message text for each code lives in `lang/*.txt`, never inline here.

use std::fmt;
use std::str::FromStr;

/// Language-neutral diagnostic code.
///
/// Ranges are grouped by subsystem so a code's prefix hints at where it came from; the
/// grouping is a convenience, not a contract — only the exact string (`ECode::as_str`)
/// is stable.
///
/// - `E09x` — backend/codegen refusals (`Support::Unsupported` in `basalt-backend`).
/// - `E1xx` — CLI argument handling.
/// - `E2xx` — frontend lexing/parsing.
/// - `E3xx` — sema.
/// - `E4xx` — BIR textual format.
/// - `E5xx` — I/O and `--lang` table loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ECode {
    /// Backend cannot lower a BIR op it does not recognize or has not implemented.
    UnsupportedOp,
    /// Backend cannot represent a BIR value's type (width, dtype, ...).
    UnsupportedType,
    /// Backend cannot address a BIR memory space (e.g. shared/local on a target lacking it).
    UnsupportedAddressSpace,
    /// Backend lacks a required hardware feature (e.g. an ISA extension).
    UnsupportedFeature,
    /// Reserved for a rank-2 tile requested on a GPU backend without the matrix
    /// (`mma`) path.
    MatrixPathUnsupported,

    /// CLI flag not recognized.
    CliUnknownFlag,
    /// CLI flag requires a value that was not supplied.
    CliMissingArgument,
    /// CLI flag was supplied a value it cannot parse/accept.
    CliInvalidArgument,
    /// Two or more CLI flags were given that cannot be combined.
    CliConflictingFlags,

    /// Frontend lexer could not tokenize the input.
    LexError,
    /// Frontend parser hit an unexpected token/construct.
    ParseError,
    /// A quoted literal, comment, or bracketed construct was never closed.
    UnterminatedConstruct,

    /// Sema type check failed.
    TypeError,
    /// Sema encountered a reference to an unbound identifier.
    UndefinedSymbol,
    /// Sema encountered a duplicate definition of the same identifier.
    Redefinition,
    /// Sema encountered an invalid combination of CUDA execution-space qualifiers (e.g.
    /// `__global__` combined with `__device__`), or one on a declaration it cannot apply to.
    InvalidCudaQualifier,
    /// AST-to-BIR lowering hit a construct it does not yet lower (calls, GPU intrinsics,
    /// templates, global variable storage, ...). The lowering pass still produces a
    /// best-effort placeholder so the rest of the function can be inspected, but the result
    /// must not be handed to a backend while this code was reported.
    LoweringUnsupported,

    /// BIR textual parser rejected the input.
    BirParseError,
    /// `parse(print(module)) != module` (round-trip invariant violated).
    BirRoundTripMismatch,

    /// Generic I/O failure (file read/write) outside the cases below.
    IoError,
    /// `--lang <file>` was pointed at a path that does not exist.
    LangFileNotFound,
    /// A line in a `lang/*.txt` table did not match the documented format.
    LangMalformedLine,
    /// The same E-code appeared twice in one `lang/*.txt` table.
    LangDuplicateCode,
    /// A `Diag` was rendered against a table with no entry for its code.
    LangMissingMessage,
}

impl ECode {
    /// Every variant, in the same order as declared. Used for round-trip tests and to check
    /// a `lang/*.txt` table is exhaustive.
    pub const ALL: &'static [ECode] = &[
        ECode::UnsupportedOp,
        ECode::UnsupportedType,
        ECode::UnsupportedAddressSpace,
        ECode::UnsupportedFeature,
        ECode::MatrixPathUnsupported,
        ECode::CliUnknownFlag,
        ECode::CliMissingArgument,
        ECode::CliInvalidArgument,
        ECode::CliConflictingFlags,
        ECode::LexError,
        ECode::ParseError,
        ECode::UnterminatedConstruct,
        ECode::TypeError,
        ECode::UndefinedSymbol,
        ECode::Redefinition,
        ECode::InvalidCudaQualifier,
        ECode::LoweringUnsupported,
        ECode::BirParseError,
        ECode::BirRoundTripMismatch,
        ECode::IoError,
        ECode::LangFileNotFound,
        ECode::LangMalformedLine,
        ECode::LangDuplicateCode,
        ECode::LangMissingMessage,
    ];

    /// The stable string form of this code, e.g. `"E099"`.
    pub fn as_str(self) -> &'static str {
        match self {
            ECode::UnsupportedOp => "E090",
            ECode::UnsupportedType => "E091",
            ECode::UnsupportedAddressSpace => "E092",
            ECode::UnsupportedFeature => "E093",
            ECode::MatrixPathUnsupported => "E099",
            ECode::CliUnknownFlag => "E100",
            ECode::CliMissingArgument => "E101",
            ECode::CliInvalidArgument => "E102",
            ECode::CliConflictingFlags => "E103",
            ECode::LexError => "E200",
            ECode::ParseError => "E201",
            ECode::UnterminatedConstruct => "E202",
            ECode::TypeError => "E300",
            ECode::UndefinedSymbol => "E301",
            ECode::Redefinition => "E302",
            ECode::InvalidCudaQualifier => "E303",
            ECode::LoweringUnsupported => "E304",
            ECode::BirParseError => "E400",
            ECode::BirRoundTripMismatch => "E401",
            ECode::IoError => "E500",
            ECode::LangFileNotFound => "E501",
            ECode::LangMalformedLine => "E502",
            ECode::LangDuplicateCode => "E503",
            ECode::LangMissingMessage => "E504",
        }
    }

    /// Parses a code back from its string form (e.g. `"E099"`). Returns `None` for anything
    /// not in `ECode::ALL`.
    pub fn from_code_str(s: &str) -> Option<ECode> {
        ECode::ALL.iter().copied().find(|c| c.as_str() == s)
    }
}

impl fmt::Display for ECode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned by `ECode::from_str` when the input does not name a known code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseECodeError(pub String);

impl fmt::Display for ParseECodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a known E-code: {}", self.0)
    }
}

impl std::error::Error for ParseECodeError {}

impl FromStr for ECode {
    type Err = ParseECodeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ECode::from_code_str(s).ok_or_else(|| ParseECodeError(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_code() {
        for &code in ECode::ALL {
            let s = code.as_str();
            assert!(
                s.starts_with('E') && s.len() == 4,
                "not a stable E-code form: {s}"
            );
            assert_eq!(s.parse::<ECode>().unwrap(), code);
        }
    }

    #[test]
    fn matrix_path_code_is_e099() {
        assert_eq!(ECode::MatrixPathUnsupported.as_str(), "E099");
    }

    #[test]
    fn unknown_string_does_not_parse() {
        assert!("E999".parse::<ECode>().is_err());
    }

    #[test]
    fn all_codes_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for &code in ECode::ALL {
            assert!(
                seen.insert(code.as_str()),
                "duplicate code string: {}",
                code.as_str()
            );
        }
    }
}
