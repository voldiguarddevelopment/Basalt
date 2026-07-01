// CUDA-C / HIP subset frontend.
//
// Current scope: lexer, preprocessor, and a recursive-descent parser for declarations and
// types. Each stage reports problems as its own local error type rather than aborting on the
// first one. Statement/expression parsing and full template instantiation land on top of this
// in later work (ARCHITECTURE.md §6).

pub mod ast;
pub mod lex;
pub mod parse;
pub mod pp;
pub mod token;

pub use lex::{lex, LexError};
pub use parse::{parse, ParseError};
pub use pp::{preprocess, PpError, PpOpts};
pub use token::{
    CharLit, FloatLit, FloatSuffix, IntBase, IntLit, Keyword, Loc, Punct, Span, StrLit, Token,
    TokenKind,
};
