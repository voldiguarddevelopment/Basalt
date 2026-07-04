// The TDF fission pass: BIR -> a real, multi-core Metalium reader/compute-writer kernel pair
// wired through real `tt_metal` circular buffers and NoC transfers. See `tdf.rs` for the
// layout data this pass builds and `--tdf`'s dump of it.
//
// # Scope this task actually settled on, and why
//
// The brief this pass was built against sketches the idealized `tt-metal` shape as three
// kernel roles per core: a reader (data-movement), a compute kernel (the Tensix tensor engine,
// TRISC0-2 driven together through the LLK tile-math API), and a writer (data-movement) —
// exactly what `tt_metal/programming_examples/vecadd_multi_core/` and
// `add_2_integers_in_compute/` both build. Reading those examples end to end (their host
// program, and all three kernel-side files) turned up a real hardware constraint that changes
// the honest scope here: a Wormhole Tensix core has exactly two general-purpose data-movement
// processors (`DataMovementProcessor::RISCV_0`/`RISCV_1`, BRISC/NCRISC —
// `tt_metal/api/tt-metalium/kernel_types.hpp`; `RISCV_2..4` exist only on the unrelated Quasar
// architecture this project's toolchain does not target). The third role is not a third
// general-purpose core to place arbitrary scalar C++ on — it is the TRISC tensor engine, and
// every real compute-kernel example in this tree (`add_2_tiles.cpp`, `add_multi_core.cpp`,
// `custom_sfpi_add/kernels/compute/tiles_add.cpp`) drives it exclusively through the LLK tile-
// math API (`binary_op_init_common`/`add_tiles_init`/`tile_regs_acquire`/`add_tiles`/
// `pack_tile`/`tile_regs_release`) over data in the hardware's native 32x32 *tilized* layout,
// always `bfloat16` (`DataFormat::Float16_b`) in every example this pass could find. `emit.rs`'s
// own module header already refused `f16` as "not a safe stand-in for Tensix's native bfloat16
// format" for a single scalar value; the tilized packing a real tile op additionally requires
// is a strictly bigger, still-unverified lift on top of that, and this pass has no way to check
// its own correctness without real silicon or a simulator (neither exists in this tree — see
// `CLAUDE.md`'s tiering discipline). Guessing at it would be exactly the "no silently-wrong
// codegen" invariant this project refuses to trade away.
//
// So: this pass fissions a BIR kernel into exactly the two roles a Tensix core can actually run
// without that unverified dependency — `reader` (BRISC) and `compute_writer` (NCRISC, folding
// the idealized "compute" and "writer" roles into the one remaining general-purpose processor)
// — connected by real circular-buffer channels for every value that crosses the reader/
// compute_writer boundary. This is a hardware-forced, honestly-documented narrowing of the
// idealized three-role picture, not a shortcut around it; see this task's own report for the
// full reasoning and the real Docker-based compile verification of both kernels.
//
// # Per-region addressing: the pointer-base-shift technique
//
// `emit.rs`'s single-core backend represents every `Ty::Ptr(Global)` parameter as a plain
// `uint8_t*` over the *entire* buffer, so a BIR pointer computed from the (global) `tid.x`
// indexes it directly. Here, each region's channel only ever holds that region's own
// `[start_tile_id, start_tile_id + n_tiles)` slice, but `tid.x` inside the reused loop body
// must still read the true *global* index (other BIR values — `vector_add.cu`'s own `n`-bound
// guard, for one — compare it against a global count). Rather than rewrite the body's pointer
// arithmetic to a local index, this pass shifts the channel's *base pointer* back by
// `start_tile_id * sizeof(element)` once, in the prologue, so indexing it with the unmodified
// global byte offset lands at the correct local address — one line per pointer parameter, and
// the entire shared `lower_inst`/`lower_term` body from `emit.rs` carries over completely
// unchanged.
//
// # What this pass does not emit
//
// Kernel-side C++ only, matching `emit.rs`'s own single-core scope: no host `Program`/
// `CreateKernel`/`CircularBufferConfig`/`SetRuntimeArgs` C++ is generated. The region/core grid,
// channel capacities, and per-region runtime-argument values this pass's own dump describes are
// therefore a documented *contract* a future host-side generator (Phase 13 territory, per
// `PLAN.md`) must satisfy, not something this pass can wire up and verify end to end itself —
// exactly the same honest boundary `emit.rs` already draws for the single-core case.

use basalt_bir::{Function, Module};
use basalt_diag::{Diag, ECode};
use basalt_passes::construct_ssa;

use crate::emit::{
    analyze_params, build_phi_copies, check_module, cpp_scalar_ty, goto_targets, CodeGen, ParamKind,
};
use crate::tdf::{Channel, CoreCoord, NocArc, NocDir, Region, Role, TdfProgram};

/// The fixed core grid every fissioned kernel targets. Not derived from a real device query
/// (`compute_with_storage_grid_size()` needs a live `MeshDevice` this crate never opens — see
/// the module header on host-side scope), and not derived from `nthreads` either, since that is
/// a runtime value this backend has never had compile-time access to (`emit.rs`'s own `bdim.x`
/// handling). Four cores, laid out 2x2, mirrors `vecadd_multi_core.cpp`'s own real "the program
/// will use 4 cores" convention.
const GRID_X: u32 = 2;
const GRID_Y: u32 = 2;
const NUM_REGIONS: u32 = GRID_X * GRID_Y;

fn conflicting_load_and_store(i: u32) -> Diag {
    Diag::new(ECode::UnsupportedOp).with_arg(format!(
        "p{i}: the TDF fission pass has no channel story yet for a pointer parameter that is \
         both read and written by the same kernel (would need a read-modify-write channel \
         spanning the reader and compute_writer kernels); refusing rather than guessing"
    ))
}

/// Everything `check_module` already validates for the single-core backend, plus one extra
/// constraint this fission pass's own two-kernel channel design cannot yet express: a pointer
/// parameter read by `reader` and written by `compute_writer` has no channel carrying it back
/// the other way. `vector_add.cu`'s own shape (two load-only pointers, one store-only pointer)
/// never hits this; see the module header for why it is a real, additional restriction rather
/// than a copy of `emit.rs`'s own refusal surface.
pub(crate) fn check_fissionable(module: &Module) -> Result<(), Diag> {
    check_module(module)?;
    let f = &module.funcs[0];
    let params = analyze_params(f)?;
    for (i, kind) in params.iter().enumerate() {
        if let ParamKind::Ptr(access) = kind {
            if access.has_load && access.has_store {
                return Err(conflicting_load_and_store(i as u32));
            }
        }
    }
    Ok(())
}

/// One pointer parameter's real per-channel facts, gathered once and threaded through every
/// codegen pass below rather than re-deriving them from `params`/`channels` at each call site.
struct PtrChan {
    idx: u32,
    cb_index: u32,
    cty: &'static str,
}

fn load_specs(params: &[ParamKind], channels: &[Channel]) -> Vec<PtrChan> {
    params
        .iter()
        .enumerate()
        .filter_map(|(i, kind)| match kind {
            ParamKind::Ptr(access) if access.has_load => {
                let cty = cpp_scalar_ty(access.scalar.expect("checked by analyze_params"))
                    .expect("f16 already refused by check_fissionable's check_module call");
                let cb_index = channels
                    .iter()
                    .find(|c| c.param == i as u32 && c.producer == Role::Reader)
                    .expect("build_channels emits one reader-produced channel per load param")
                    .cb_index;
                Some(PtrChan {
                    idx: i as u32,
                    cb_index,
                    cty,
                })
            }
            _ => None,
        })
        .collect()
}

fn store_specs(params: &[ParamKind], channels: &[Channel]) -> Vec<PtrChan> {
    params
        .iter()
        .enumerate()
        .filter_map(|(i, kind)| match kind {
            ParamKind::Ptr(access) if access.has_store => {
                let cty = cpp_scalar_ty(access.scalar.expect("checked by analyze_params"))
                    .expect("f16 already refused by check_fissionable's check_module call");
                let cb_index = channels
                    .iter()
                    .find(|c| c.param == i as u32 && c.producer == Role::ComputeWriter)
                    .expect("build_channels emits one compute_writer-owned channel per store param")
                    .cb_index;
                Some(PtrChan {
                    idx: i as u32,
                    cb_index,
                    cty,
                })
            }
            _ => None,
        })
        .collect()
}

/// One real circular buffer per load parameter (`reader` produces, `compute_writer` consumes)
/// and one per store parameter (`compute_writer` both produces and consumes it itself — see
/// `Channel::is_internal_scratch` and the module header on why that never crosses a kernel
/// boundary). Input indices start at 0, output indices at 16, mirroring
/// `add_2_integers_in_compute`'s own real `c_0`/`c_1` in, `c_16` out convention — an arbitrary
/// but real split, not a hardware requirement.
fn build_channels(params: &[ParamKind]) -> Vec<Channel> {
    let mut channels = Vec::new();
    let mut next_in = 0u32;
    let mut next_out = 16u32;
    for (i, kind) in params.iter().enumerate() {
        let ParamKind::Ptr(access) = kind else {
            continue;
        };
        let cty = cpp_scalar_ty(access.scalar.expect("checked by analyze_params"))
            .expect("f16 already refused by check_fissionable's check_module call");
        if access.has_load {
            channels.push(Channel {
                cb_index: next_in,
                name: format!("cb_in{next_in}"),
                param: i as u32,
                scalar_ty: cty.to_string(),
                producer: Role::Reader,
                consumer: Role::ComputeWriter,
            });
            next_in += 1;
        }
        if access.has_store {
            channels.push(Channel {
                cb_index: next_out,
                name: format!("cb_out{}", next_out - 16),
                param: i as u32,
                scalar_ty: cty.to_string(),
                producer: Role::ComputeWriter,
                consumer: Role::ComputeWriter,
            });
            next_out += 1;
        }
    }
    channels
}

fn build_noc_arcs(channels: &[Channel]) -> Vec<NocArc> {
    channels
        .iter()
        .map(|c| {
            if c.producer == Role::Reader {
                NocArc {
                    dir: NocDir::Read,
                    param: c.param,
                    channel_name: c.name.clone(),
                    role: Role::Reader,
                }
            } else {
                NocArc {
                    dir: NocDir::Write,
                    param: c.param,
                    channel_name: c.name.clone(),
                    role: Role::ComputeWriter,
                }
            }
        })
        .collect()
}

/// `reader` (BRISC/`RISCV_0`): moves every load parameter's region-assigned slice DRAM->L1,
/// one real `noc_async_read` per parameter, landing in that parameter's own channel. Real
/// runtime-arg layout: `start_tile_id`, `n_tiles`, then one DRAM address per load parameter in
/// parameter order — the host-side contract this kernel's own `SetRuntimeArgs` call (not
/// emitted by this crate — see the module header) must honor.
fn emit_reader_kernel(params: &[ParamKind], channels: &[Channel]) -> String {
    let loads = load_specs(params, channels);
    let mut out = String::new();
    out.push_str("void kernel_main() {\n");
    out.push_str("    uint32_t start_tile_id = get_arg_val<uint32_t>(0);\n");
    out.push_str("    uint32_t n_tiles = get_arg_val<uint32_t>(1);\n");
    for (k, spec) in loads.iter().enumerate() {
        out.push_str(&format!(
            "    uint32_t p{}_dram = get_arg_val<uint32_t>({});\n",
            spec.idx,
            2 + k
        ));
    }
    if !loads.is_empty() {
        out.push('\n');
    }
    for spec in &loads {
        out.push_str(&format!(
            "    InterleavedAddrGen<true> p{}_gen = {{.bank_base_address = p{}_dram, \
             .page_size = sizeof({})}};\n",
            spec.idx, spec.idx, spec.cty
        ));
    }
    for spec in &loads {
        out.push_str(&format!("    cb_reserve_back({}, 1);\n", spec.cb_index));
    }
    for spec in &loads {
        out.push_str(&format!(
            "    uint32_t cb_in{}_addr = get_write_ptr({});\n",
            spec.cb_index, spec.cb_index
        ));
    }
    for spec in &loads {
        out.push_str(&format!(
            "    noc_async_read(p{}_gen.get_noc_addr(start_tile_id), cb_in{}_addr, \
             (uint32_t)(n_tiles * sizeof({})));\n",
            spec.idx, spec.cb_index, spec.cty
        ));
    }
    if !loads.is_empty() {
        out.push_str("    noc_async_read_barrier();\n");
    }
    for spec in &loads {
        out.push_str(&format!("    cb_push_back({}, 1);\n", spec.cb_index));
    }
    out.push_str("}\n");
    out
}

/// `compute_writer` (NCRISC/`RISCV_1`): waits for every load channel, runs the exact same
/// per-thread body `emit.rs`'s single-core backend would (`lower_inst`/`lower_term`, unchanged —
/// see the module header on the pointer-base-shift technique that makes this possible), then
/// writes every store parameter's region-assigned slice L1->DRAM. Real runtime-arg layout:
/// `start_tile_id`, `n_tiles`, `nthreads` (the *global* count — still needed for `bdim.x`
/// parity with the single-core backend's own semantics), then one DRAM address per store
/// parameter, then one value per scalar parameter, all in parameter order.
fn emit_compute_writer_kernel(
    f: &Function,
    params: &[ParamKind],
    channels: &[Channel],
) -> Result<String, Diag> {
    let loads = load_specs(params, channels);
    let stores = store_specs(params, channels);

    let mut cg = CodeGen {
        f,
        params,
        phi_copies: build_phi_copies(f),
        out: String::new(),
    };
    cg.out.push_str("void kernel_main() {\n");
    cg.line("uint32_t start_tile_id = get_arg_val<uint32_t>(0);");
    cg.line("uint32_t n_tiles = get_arg_val<uint32_t>(1);");
    cg.line("uint32_t nthreads = get_arg_val<uint32_t>(2);");
    let mut arg_idx = 3u32;
    for spec in &stores {
        cg.line(&format!(
            "uint32_t p{}_dram = get_arg_val<uint32_t>({arg_idx});",
            spec.idx
        ));
        arg_idx += 1;
    }
    for (i, kind) in params.iter().enumerate() {
        if let ParamKind::Scalar(s) = kind {
            let cty = cpp_scalar_ty(*s)?;
            cg.line(&format!(
                "{cty} p{i} = ({cty})get_arg_val<uint32_t>({arg_idx});"
            ));
            arg_idx += 1;
        }
    }
    cg.out.push('\n');

    for spec in &loads {
        cg.line(&format!("cb_wait_front({}, 1);", spec.cb_index));
        cg.line(&format!(
            "uint32_t cb_in{}_addr = get_read_ptr({});",
            spec.cb_index, spec.cb_index
        ));
        cg.line(&format!(
            "uint8_t* p{} = (uint8_t*)(uintptr_t)cb_in{}_addr - (uintptr_t)start_tile_id * \
             sizeof({});",
            spec.idx, spec.cb_index, spec.cty
        ));
    }
    for spec in &stores {
        // See the module header: this channel is reserved and consumed within this same kernel,
        // never crossing to `reader` — it exists purely as a real, host-configured L1 scratch
        // region, the same technique `vecadd_multi_core`'s own reader kernel documents ("we can
        // actually access [circular buffer SRAM] by casting the address to a pointer").
        cg.line(&format!("cb_reserve_back({}, 1);", spec.cb_index));
        cg.line(&format!(
            "uint32_t cb_out{}_addr = get_write_ptr({});",
            spec.cb_index - 16,
            spec.cb_index
        ));
        cg.line(&format!(
            "uint8_t* p{} = (uint8_t*)(uintptr_t)cb_out{}_addr - (uintptr_t)start_tile_id * \
             sizeof({});",
            spec.idx,
            spec.cb_index - 16,
            spec.cty
        ));
        cg.line(&format!(
            "InterleavedAddrGen<true> p{}_gen = {{.bank_base_address = p{}_dram, .page_size = \
             sizeof({})}};",
            spec.idx, spec.idx, spec.cty
        ));
    }

    cg.emit_ssa_decls()?;
    cg.out.push('\n');
    cg.line("for (uint32_t __tid = start_tile_id; __tid < start_tile_id + n_tiles; ++__tid) {");
    let targets = goto_targets(f);
    for (bidx, block) in f.blocks.iter().enumerate() {
        if targets.contains(&(bidx as u32)) {
            cg.out.push_str(&format!("    L{bidx}:\n"));
        }
        for &inst_id in &block.insts {
            cg.lower_inst(inst_id)?;
        }
        cg.lower_term(bidx as u32, &block.term);
    }
    cg.line("}");
    cg.out.push('\n');

    for spec in &stores {
        cg.line(&format!(
            "noc_async_write(cb_out{}_addr, p{}_gen.get_noc_addr(start_tile_id), (uint32_t)(n_tiles \
             * sizeof({})));",
            spec.cb_index - 16,
            spec.idx,
            spec.cty
        ));
    }
    if !stores.is_empty() {
        cg.line("noc_async_write_barrier();");
    }
    for spec in &loads {
        cg.line(&format!("cb_pop_front({}, 1);", spec.cb_index));
    }
    cg.out.push_str("}\n");
    Ok(cg.out)
}

/// Runs the fission pass over `module`'s one function (`check_fissionable` requires exactly
/// one, same as `emit.rs`), producing the full `TdfProgram` layout: a fixed 4-region grid,
/// the real channels/NoC arcs that grid's reader/compute_writer kernel pair uses, and the two
/// kernel bodies themselves.
pub(crate) fn build_tdf(module: &Module) -> Result<TdfProgram, Diag> {
    check_fissionable(module)?;
    let ssa_module = construct_ssa(module);
    let f = &ssa_module.funcs[0];
    let params = analyze_params(f)?;

    let channels = build_channels(&params);
    let noc_arcs = build_noc_arcs(&channels);
    let reader_kernel = emit_reader_kernel(&params, &channels);
    let compute_writer_kernel = emit_compute_writer_kernel(f, &params, &channels)?;

    let regions = (0..NUM_REGIONS)
        .map(|id| Region {
            id,
            core: CoreCoord {
                x: id % GRID_X,
                y: id / GRID_X,
            },
        })
        .collect();

    Ok(TdfProgram {
        func_name: f.name.clone(),
        grid_x: GRID_X,
        grid_y: GRID_Y,
        regions,
        channels,
        noc_arcs,
        reader_kernel,
        compute_writer_kernel,
    })
}

#[cfg(test)]
mod tests;
