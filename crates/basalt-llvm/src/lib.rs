// OPTIONAL (feature=llvm): BIR -> LLVM IR via inkwell.
//
// `inkwell` is an optional dependency of this crate gated behind this crate's own `llvm`
// feature (not merely the workspace-level `--features llvm` on `basalt-cli`), so a bare
// `cargo build` never touches `llvm-sys`'s build script and never needs an LLVM toolchain
// present. With the feature off this crate is empty.

#[cfg(feature = "llvm")]
mod lower;

#[cfg(feature = "llvm")]
pub use lower::lower_module;
