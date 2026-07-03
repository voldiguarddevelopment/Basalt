// OPTIONAL (feature=mlir): BIR -> MLIR dialects via melior.
//
// `melior` is an optional dependency of this crate gated behind this crate's own `mlir`
// feature (not merely the workspace-level `--features mlir` on `basalt-cli`), so a bare
// `cargo build` never touches `mlir-sys`'s build script and never needs an LLVM/MLIR
// toolchain present. With the feature off this crate is empty.

#[cfg(feature = "mlir")]
mod lower;

#[cfg(feature = "mlir")]
pub use lower::{lower_module, lower_to_text};
