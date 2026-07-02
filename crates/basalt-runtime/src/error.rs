use std::fmt;

/// Everything that can go wrong loading or driving the CUDA driver, from a missing
/// `libcuda.so` through a failed driver call. No `basalt-diag` E-code integration here —
/// this crate has no consuming diagnostic stage yet (same reasoning as `basalt-frontend-c`'s
/// local `LexError`/`ParseError`: that wiring belongs to whichever later stage renders these
/// to a user-facing message).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// Neither `libcuda.so.1` nor `libcuda.so` could be `dlopen`ed; the string is the
    /// dynamic linker's own diagnostic for the last name tried.
    DriverNotFound(String),
    /// The driver library opened, but a required entry point isn't exported under any of
    /// the symbol names this crate knows to try.
    SymbolNotFound(&'static str),
    /// A resolved driver entry point ran and returned a non-success `CUresult`.
    DriverCallFailed {
        call: &'static str,
        code: i32,
        message: String,
    },
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::DriverNotFound(msg) => {
                write!(f, "CUDA driver library not found: {msg}")
            }
            CudaError::SymbolNotFound(sym) => {
                write!(f, "CUDA driver symbol not found: {sym}")
            }
            CudaError::DriverCallFailed {
                call,
                code,
                message,
            } => {
                write!(f, "{call} failed with CUresult {code}: {message}")
            }
        }
    }
}

impl std::error::Error for CudaError {}
