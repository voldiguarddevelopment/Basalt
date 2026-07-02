// basalt-amdgpu: hand-rolled RDNA3 (gfx1100) instruction encoder and HSACO container writer.
//
// `enc` is encoder primitives only, the AMDGCN equivalent of `basalt-x86/src/enc.rs`: one
// Rust function per real instruction *form*, each producing exact machine-code bytes. It does
// not decide how to lower BIR into instruction sequences (that is a later, separate crate
// module) and it is not a general-purpose AMDGPU assembler — only the subset of
// SOP2/SOP1/SOPK/SOPC/SOPP/VOP1/VOP2/VOP3/VOPC/SMEM/DS/FLAT-GLOBAL encodings a lowering pass
// would plausibly need is implemented, and only for gfx1100 (RDNA3, wave32).
//
// Every encoding here was derived and checked against a real, independent assembler for this
// exact target (LLVM's MC layer, `-mcpu=gfx1100 -show-encoding`) during development; the
// resulting bytes are hard-coded into this crate's tests and carry no runtime dependency on
// any external tool. See `enc`'s module header for the encoding-format notes and register
// model, and the `tests` module at the bottom of `enc.rs` for the derivation of every
// hard-coded byte sequence.
//
// `hsaco` wraps `enc`'s output (or, once it exists, a real lowering pass's) in the ELF
// container a real AMDGPU loader expects: kernel descriptor, entry-point relocation, and
// NT_AMDGPU_METADATA. See `hsaco`'s own module header for how each byte-level detail was
// pinned down against a real reference object.

pub mod enc;
pub mod hsaco;
