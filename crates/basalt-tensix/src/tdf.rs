// The Tile-DataFlow (TDF) representation: the explicit regions/channels/NoC-arc structure
// `fission.rs` builds on top of a validated BIR module, and `--tdf`'s own text dump of it. This
// file is pure data plus a deterministic printer — no BIR analysis and no C++ generation live
// here (see `fission.rs` for both).
//
// Grounded in real, working multi-core `tt-metal` example programs, not invented: a `Region`
// is one core in the grid `tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp`
// partitions work across (`CoreRange`/`split_work_to_cores`); a `Channel` is one real
// `tt_metal::CircularBufferConfig`-backed circular buffer, the same "reader pushes, compute
// pops" primitive that example's own `interleaved_tile_read_multi_core.cpp` /
// `add_multi_core.cpp` pair (and `add_2_integers_in_compute`'s reader/compute/writer triplet)
// uses to hand tiles from one kernel to another; a `NocArc` is one real `noc_async_read`/
// `noc_async_write` transfer between a DRAM buffer and a channel's L1 backing.
//
// Every region shares the same two kernel bodies (`reader`/`compute_writer` in the printed
// dump) — real `tt-metal` multi-core examples compile one kernel *file* per role and reuse it
// across every core in the grid, varying only the per-core runtime arguments
// (`SetRuntimeArgs`); this crate does not emit that host-side wiring (see `fission.rs`'s module
// header for why), so the dump documents the intended per-region runtime-argument contract in
// prose rather than in generated host C++.

/// One core in the fixed grid this fission pass targets. `tt_metal::CoreCoord` is `(x, y)`;
/// see `fission.rs` for why the grid size itself is a fixed constant rather than derived from
/// a real device query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoreCoord {
    pub(crate) x: u32,
    pub(crate) y: u32,
}

/// A region: one core, running both kernel roles (see the module header on why a genuine third,
/// TRISC-resident role is out of this task's real-compile-verified scope). `slice` is the
/// symbolic, runtime-resolved range of the flat BIR thread space this region is responsible
/// for — never a literal number, since `nthreads` is itself a runtime value this backend has
/// never had access to at compile time (see `emit.rs`'s own `bdim.x` handling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) id: u32,
    pub(crate) core: CoreCoord,
}

/// Which physical kernel role produces or consumes a channel. `Reader` is the BRISC
/// (`DataMovementProcessor::RISCV_0`) kernel; `ComputeWriter` is the NCRISC
/// (`DataMovementProcessor::RISCV_1`) kernel that both performs the BIR body's arithmetic and
/// writes the result back to DRAM — the module header explains why these two roles, not the
/// idealized three, are what a Tensix core can actually run without an unverified bfloat16/tile
/// LLK dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Reader,
    ComputeWriter,
}

impl Role {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Role::Reader => "reader",
            Role::ComputeWriter => "compute_writer",
        }
    }
}

/// A real `tt_metal::CircularBufferConfig`-backed channel, keyed by a `tt::CBIndex`. `producer`/
/// `consumer` name which role writes/reads it; a channel whose producer and consumer are the
/// same role (see `is_internal_scratch`) never crosses a kernel boundary — it exists purely to
/// obtain a valid L1 address via `cb_reserve_back`/`get_write_ptr`, since this crate emits no
/// host-side buffer allocation of its own (see `fission.rs`).
#[derive(Debug, Clone)]
pub(crate) struct Channel {
    pub(crate) cb_index: u32,
    pub(crate) name: String,
    pub(crate) param: u32,
    pub(crate) scalar_ty: String,
    pub(crate) producer: Role,
    pub(crate) consumer: Role,
}

impl Channel {
    pub(crate) fn is_internal_scratch(&self) -> bool {
        self.producer == self.consumer
    }
}

/// Direction of one real NoC transfer (`noc_async_read`/`noc_async_write`) between a DRAM-
/// resident BIR pointer parameter and a channel's L1 backing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NocDir {
    Read,
    Write,
}

/// One explicit data-movement edge: DRAM buffer `param` <-> channel `channel_name`, issued by
/// `region`'s copy of `role`'s kernel, over the region's own `[start_tile_id, start_tile_id +
/// n_tiles)` slice (see `Region::slice` — there is no literal offset to print, since it is a
/// runtime argument, not a compile-time constant).
#[derive(Debug, Clone)]
pub(crate) struct NocArc {
    pub(crate) dir: NocDir,
    pub(crate) param: u32,
    pub(crate) channel_name: String,
    pub(crate) role: Role,
}

/// The full layout one `fission::fission` call produces for one BIR function: a fixed grid of
/// regions, the channels wiring reader to compute_writer within each, the NoC arcs each role's
/// kernel issues, and the two kernel bodies themselves (shared verbatim across every region).
#[derive(Debug, Clone)]
pub(crate) struct TdfProgram {
    pub(crate) func_name: String,
    pub(crate) grid_x: u32,
    pub(crate) grid_y: u32,
    pub(crate) regions: Vec<Region>,
    pub(crate) channels: Vec<Channel>,
    pub(crate) noc_arcs: Vec<NocArc>,
    pub(crate) reader_kernel: String,
    pub(crate) compute_writer_kernel: String,
}

fn param_text(i: u32) -> String {
    format!("p{i}")
}

/// Prints a `TdfProgram` deterministically: fixed iteration order throughout (`Vec`s in the
/// order this pass built them, never a hash-keyed structure), matching every other backend's
/// determinism contract in this tree. This is `--tdf`'s entire output.
pub(crate) fn print_tdf(prog: &TdfProgram) -> String {
    let mut out = String::new();
    out.push_str(&format!("tdf module for `{}`\n", prog.func_name));
    out.push_str(&format!(
        "grid {}x{} ({} regions, one core each)\n\n",
        prog.grid_x,
        prog.grid_y,
        prog.regions.len()
    ));

    out.push_str("regions:\n");
    for r in &prog.regions {
        out.push_str(&format!(
            "  region r{} core({},{}) slice=[start_tile_id, start_tile_id + n_tiles) \
             (runtime args, per-region; not a compile-time constant)\n",
            r.id, r.core.x, r.core.y
        ));
        out.push_str(
            "    role reader          kernel=reader_kernel         processor=RISCV_0 (BRISC)\n",
        );
        out.push_str(
            "    role compute_writer  kernel=compute_writer_kernel processor=RISCV_1 (NCRISC)\n",
        );
    }
    out.push('\n');

    out.push_str("channels (real tt_metal::CircularBufferConfig, one per region, shared cb index across regions):\n");
    for c in &prog.channels {
        if c.is_internal_scratch() {
            out.push_str(&format!(
                "  {} (cb_index={}, {}) param={} ty={} internal scratch: produced and consumed \
                 within {} — never crosses a kernel boundary, used only to obtain a valid L1 \
                 address via cb_reserve_back/get_write_ptr\n",
                c.name,
                c.cb_index,
                c.scalar_ty,
                param_text(c.param),
                c.scalar_ty,
                c.producer.as_str()
            ));
        } else {
            out.push_str(&format!(
                "  {} (cb_index={}) param={} ty={} producer={} consumer={}\n",
                c.name,
                c.cb_index,
                param_text(c.param),
                c.scalar_ty,
                c.producer.as_str(),
                c.consumer.as_str()
            ));
        }
    }
    out.push('\n');

    out.push_str("noc arcs (one real noc_async_read/noc_async_write per region, over that region's own slice):\n");
    for a in &prog.noc_arcs {
        match a.dir {
            NocDir::Read => out.push_str(&format!(
                "  read  dram:{} -> channel:{}  (issued by {})\n",
                param_text(a.param),
                a.channel_name,
                a.role.as_str()
            )),
            NocDir::Write => out.push_str(&format!(
                "  write channel:{} -> dram:{}  (issued by {})\n",
                a.channel_name,
                param_text(a.param),
                a.role.as_str()
            )),
        }
    }
    out.push('\n');

    out.push_str(
        "==== reader kernel (shared verbatim by every region; only runtime args differ) ====\n",
    );
    out.push_str(&prog.reader_kernel);
    out.push('\n');
    out.push_str("==== compute_writer kernel (shared verbatim by every region; only runtime args differ) ====\n");
    out.push_str(&prog.compute_writer_kernel);
    out
}
