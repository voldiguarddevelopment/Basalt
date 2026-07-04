// Hand-rolled RV32IM / RV64 + soft-float runtime.
//
// `enc` is the low-level RV32IM instruction encoder (no external assembler/encoder crate —
// see its own header for why), kept `pub` the same way `basalt-amdgpu`'s own encoder is:
// a from-scratch ISA encoder naturally covers a little more of the format space than any one
// lowering pass currently exercises (every R/I/S/B/U/J-format instruction here was verified
// against a real assembler regardless of whether `lower`/`softfloat` happen to need it yet).
// `softfloat` is the hand-rolled soft-float runtime `enc` builds on (RV32IM has no F/D
// extension); `softfloat_ref` is its pure-Rust, vector-validated reference algorithm, kept
// test-only since it is never linked into an emitted object. `lower` is the `Backend` impl
// (`Rv32`) tying it all together — see that module's own header for the full design
// (threading model, calling convention, frame layout, and this backend's documented scope
// cuts).
pub mod enc;
mod lower;
pub mod softfloat;
#[cfg(test)]
mod softfloat_ref;

pub use lower::Rv32;
