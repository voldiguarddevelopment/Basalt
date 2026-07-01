// CUDA-C / HIP subset frontend.
//
// Current scope: the lexer only. `lex` turns source text into the `Token` stream defined in
// `token`, reporting problems as `LexError`s local to this crate rather than aborting on the
// first one. The preprocessor and recursive-descent parser (ARCHITECTURE.md §6) land on top
// of this in later work.

pub mod lex;
pub mod token;

pub use lex::{lex, LexError};
pub use token::{
    CharLit, FloatLit, FloatSuffix, IntBase, IntLit, Keyword, Loc, Punct, Span, StrLit, Token,
    TokenKind,
};
