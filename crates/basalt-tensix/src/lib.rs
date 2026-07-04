// Hand-rolled Tenstorrent Metalium backend: unlike every other target in this tree, Tensix has
// no register machine or ISA in the usual sense — a "kernel" is C++ compiled against the real
// `tt_metal` device-kernel API and run on one of a Tensix tile's own RISC-V cores. This crate's
// job is text emission (BIR -> Metalium C++), not byte encoding, closer in spirit to
// `basalt-ptx`'s "emit text, not bytes" stance than to `basalt-x86`/`basalt-rv`. See `emit.rs`'s
// header for the real design: scope, refusal surface, and the toolchain this was verified
// against.
//
// The TDF layer (regions/channels/NoC arcs, multi-core fission) named in ARCHITECTURE §2 is
// P12-T4, a separate, later task; this crate currently covers only the single-core kernel
// bring-up.

mod emit;

pub use emit::Tensix;
