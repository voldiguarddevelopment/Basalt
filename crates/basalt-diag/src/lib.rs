// basalt-diag: the diagnostics layer.
//
// Every user-facing error in the project is represented as a `Diag` carrying a stable
// `ECode` — the E-code is the contract other crates and tests depend on; message text is
// never inlined in Rust source. Message text lives in `lang/*.txt` tables (this crate's
// `LangTable`), loaded either as the shipped default (`lang/en.txt`) or from an arbitrary
// path via `--lang <file>` (`load_lang_file`).
//
// Public surface: `ECode` (registry + string round-trip), `Diag`/`Span`/`Loc` (a diagnostic
// and its optional source location), and `LangTable`/`load_lang_file` (the message-table
// loader). Downstream crates (`basalt-backend`, `basalt-cli`, frontends, sema) build `Diag`
// values and render them against a `LangTable` at the point they need user-facing text.

mod diag;
mod ecode;
mod lang;

pub use diag::{Diag, Loc, Span};
pub use ecode::{ECode, ParseECodeError};
pub use lang::{load_lang_file, LangTable};
