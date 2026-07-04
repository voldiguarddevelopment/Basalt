// Hand-rolled Tenstorrent Metalium backend: unlike every other target in this tree, Tensix has
// no register machine or ISA in the usual sense — a "kernel" is C++ compiled against the real
// `tt_metal` device-kernel API and run on one of a Tensix tile's own RISC-V cores. This crate's
// job is text emission (BIR -> Metalium C++), not byte encoding, closer in spirit to
// `basalt-ptx`'s "emit text, not bytes" stance than to `basalt-x86`/`basalt-rv`. See `emit.rs`'s
// header for the real design: scope, refusal surface, and the toolchain this was verified
// against.
//
// The TDF layer (regions/channels/NoC arcs, multi-core fission) named in ARCHITECTURE §2 is
// `fission.rs`/`tdf.rs` (P12-T4): a real fission pass splitting a validated single-core-shaped
// BIR kernel into a reader/compute_writer Metalium kernel pair wired through real `tt_metal`
// circular buffers and NoC transfers, plus `dump_tdf`'s text dump of the resulting layout
// (`basalt-cli`'s `--tdf` flag). See `fission.rs`'s own header for the exact scope this task
// settled on and why.

mod emit;
mod fission;
mod tdf;

pub use emit::Tensix;

use basalt_bir::Module;
use basalt_diag::Diag;

/// `--tdf`: runs the TDF fission pass over `module` and renders the resulting layout
/// (regions/channels/NoC arcs, plus the two kernel bodies the pass produced) as text. Returns
/// the same kind of `Diag` `Tensix::emit` would for a module this pass cannot fission — see
/// `fission::check_fissionable`.
pub fn dump_tdf(module: &Module) -> Result<String, Diag> {
    let prog = fission::build_tdf(module)?;
    Ok(tdf::print_tdf(&prog))
}
