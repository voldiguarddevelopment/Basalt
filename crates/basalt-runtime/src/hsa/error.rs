use std::fmt;

/// Everything that can go wrong loading or driving the HSA runtime, from a missing
/// `libhsa-runtime64.so` through a failed runtime call or an asynchronous queue fault reported
/// through the ABEND callback (see `queue.rs`). Mirrors `../error.rs`'s `CudaError` shape and
/// the same reasoning for staying independent of `basalt-diag`'s E-codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HsaError {
    /// `libhsa-runtime64.so` could not be `dlopen`ed; the string is the dynamic linker's own
    /// diagnostic for the last name tried.
    DriverNotFound(String),
    /// The runtime library opened, but a required entry point isn't exported under any of the
    /// symbol names this crate knows to try.
    SymbolNotFound(&'static str),
    /// A resolved runtime entry point ran and returned a non-success `hsa_status_t`, or a
    /// queue's error callback reported an asynchronous fault after the fact.
    RuntimeCallFailed {
        call: &'static str,
        code: i32,
        message: String,
    },
}

impl fmt::Display for HsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HsaError::DriverNotFound(msg) => {
                write!(f, "HSA runtime library not found: {msg}")
            }
            HsaError::SymbolNotFound(sym) => {
                write!(f, "HSA runtime symbol not found: {sym}")
            }
            HsaError::RuntimeCallFailed {
                call,
                code,
                message,
            } => {
                write!(f, "{call} failed with hsa_status_t {code}: {message}")
            }
        }
    }
}

impl std::error::Error for HsaError {}
