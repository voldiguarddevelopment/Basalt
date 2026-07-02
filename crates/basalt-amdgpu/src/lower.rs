// BIR-to-AMDGCN (gfx1100) lowering: the `Backend` impl (`Amdgcn`) that turns a BIR `Module`
// into a real HSACO object, built directly on `enc`'s instruction encoders and `hsaco`'s
// container writer. This is the project's first hand-rolled backend that targets genuine
// concurrent SIMT hardware (shared property with `basalt-ptx`, see that crate's own header) —
// there is no synthesized per-thread loop anywhere in this file.
//
// # Scope: one validated kernel first, not universal BIR coverage
//
// This lowering pass covers exactly the scalar/pointer slice `tests/kernels/stress.cu` (this
// phase's proof kernel) and its immediate neighbors actually need, plus a few op families that
// are cheap to add correctly once the core machinery exists (all of `TidX/Y/Z`, `Phi`,
// `Select`, integer/float compare, `Atomic` on `Global`). Anything outside that slice is a
// clean `Support::Unsupported` refusal with a stable E-code (`check_module` below), never a
// guess:
//   - `i8`/`i16`/`f16`/`f64` and every `Ty::Vec` are refused (`E091`): only `i1`/`i32`/`i64`/
//     `f32`/pointers are lowered.
//   - Integer `div`/`rem` and float `div`/`rem` are refused (`E093`): AMDGCN has no native
//     integer divide (true of every real GPU ISA), and an IEEE-correct float divide needs a
//     verified reciprocal-plus-Newton-Raphson sequence this task's time budget did not reach —
//     guessing at either would be exactly the silently-wrong codegen this project refuses to
//     ship.
//   - `Shuffle`/`Ballot`/`VoteAny`/`VoteAll` are refused (`E093`). AMDGCN's real mechanisms
//     (`ds_permute`/`ds_bpermute` for shuffle, exec-mask popcount for ballot/vote) are a
//     different encoding family this task did not reach; a later task can add them the same
//     way `enc.rs`'s own history reads — one verified encoder at a time.
//   - `Op::BdimX/Y/Z`/`GdimX/Y/Z` are refused (`E093`), matching `basalt-llvm`'s own documented
//     gap for the identical reason: block/grid dimensions are not simply available in a
//     register the way `tid`/`bid` are — they live in the dispatch packet, reachable only
//     through the "implicit kernarg" hidden-argument mechanism. `Op::TidX/Y/Z` and
//     `Op::BidX/Y/Z` *are* implemented for real (see "thread/block index" below) since both are
//     plain preloaded-register reads with no dispatch-packet plumbing required.
//   - `Op::AtomicCas` is refused (`E093`): the real FLAT/GLOBAL `cmpswap` opcode packs its
//     compare and new-value operands into one *adjacent* VGPR pair, a constraint this backend's
//     one-dedicated-register-per-SSA-value scheme does not arrange for; wiring that up
//     correctly is future work, not a guess.
//   - `AddrSpace::Param`/`Constant` `Load`/`Store` are refused (`E092`): these address spaces
//     are `basalt-sema`'s synthetic parameter/constant-slot bookkeeping (see that crate's
//     `lower.rs` header), which `basalt_passes::construct_ssa` is expected to eliminate
//     entirely before a backend ever sees it — every kernel this backend has been validated
//     against confirms this. A slot access that somehow survives is refused rather than handed
//     a made-up memory model.
//   - `Shared`/`Local` `Load`/`Store` of a 64-bit-wide value (`i64`, a `Global`-width pointer)
//     are refused (`E093`): LDS addressing in this backend is a single 32-bit VGPR offset, and
//     a 64-bit LDS payload needs a two-word DS form this task did not add.
//   - `Term::Switch` is refused (`E093`): only `Br`/`CondBr`/`Ret` are lowered.
//   - More than one function per module is refused (`E093`): `hsaco::HsacoSpec` is one kernel
//     per object (see that module's header), matching this same simplification.
//
// # Register model
//
// Every SSA value gets a VGPR home picked by a simple liveness-based linear scan (see
// `RegAlloc`/`Pools`/`compute_last_use`'s own header just above their definitions for the full
// reasoning) — *not* the divergence-aware allocator a later, dedicated task will build, and
// *not* a real spiller: it has no notion of a value being uniform vs. divergent and no
// exec-mask interaction, it just frees a register the instant its value's last real use has
// passed and hands it to the next value needing the same width. A register is still never
// shared by two *simultaneously live* values, so (matching `basalt-ptx`'s own documented
// reasoning for its own, permanent-only, register scheme) phi resolution still needs no staging
// dance: an unconditional register-to-register copy per incoming edge is always correct.
//
// VGPRs: `v0` is permanently reserved as the untouched hardware-preloaded packed
// thread-index register (see "thread/block index" below) and is never assigned to an SSA
// value; every SSA value's VGPR home is numbered starting at `v1`. A value's width in VGPRs
// follows its `Ty`: `i1`/`i32`/`f32`/a `Shared`/`Local` pointer (a 32-bit LDS offset, not a
// full address) take one VGPR; `i64`/a `Global`/`Constant`/`Param` pointer (a full 64-bit
// address) take two *consecutive* VGPRs, low word first — required by `enc.rs`'s FLAT/GLOBAL
// forms, which address a 64-bit pointer as one VGPR pair. Running past `v255` (with every
// already-dead register recycled) is a clean `E093` refusal (see `Pools::alloc`), not silent
// wraparound. Every value — including `i1` — lives in the vector file; this backend does not
// attempt the scalar/vector uniformity split a real divergence-aware allocator would use, so
// `Op::BidX` deliberately reads its hardware SGPR straight into a VGPR rather than keeping it
// scalar — simpler, always correct, just not using the scalar unit optimally.
//
// SGPRs: `s[0:1]` is the kernarg segment base pointer (hardware-preloaded whenever the kernel
// takes at least one parameter — see `HsacoSpec::with_kernarg_segment`); `s2`, then `s3`, then
// `s4` are the workgroup (block) id x/y/z components, preloaded only for the axes this
// function's body actually reads, packed contiguously starting at `s2` in x-then-y-then-z
// order and skipping any unused axis — a real hardware/kernel-descriptor packing rule, not a
// convention this backend invented (see `hsaco::HsacoSpec::with_workgroup_ids`). `s16`/`s17`
// are a fixed scratch pair reused, one parameter at a time, to stage each kernarg value between
// its SMEM load and the `v_mov_b32` that broadcasts it into the parameter's own VGPR home —
// safe to reuse because parameter loads happen strictly in sequence at kernel entry, before
// anything else runs.
//
// # `i1` representation
//
// Every `i1`-typed VGPR is a deliberately maintained invariant: it holds exactly `0` or `1`,
// never any other bit pattern, enforced at every production site (`icmp`/`fcmp` materialize
// their vector-compare result via `v_cndmask_b32 dst, 0, 1, vcc_lo`; a `trunc` to `i1` masks
// with `v_and_b32 dst, src, 1`; everything else that produces an `i1` — `phi`, `select`,
// `zext`/`sext` *from* `i1` — just copies or arithmetically extends an already-clean value).
// This is a narrower, cheaper invariant than `basalt-ptx`'s general "operate at declared width,
// extend on demand" convention (this backend does not need that generality, since `i8`/`i16`
// are refused outright — `i1` is the only sub-32-bit type in scope), and it means every `i1`
// consumer (`zext`, `select`, a branch condition) can read the VGPR directly with no on-demand
// canonicalization.
//
// # Thread/block index
//
// `Op::TidX/Y/Z`: real AMDGCN hardware packs all three local thread-index components into one
// preloaded VGPR (`v0`) as three 10-bit fields — `tid.x` in bits `[9:0]`, `tid.y` in `[19:10]`,
// `tid.z` in `[29:20]` — this is not a simulator convention, it is the same packed-workitem-id
// layout a real `TargetMachine` assumes for the common (non-`ENABLE_VGPR_WORKITEM_ID`-expanded)
// case, and it is what this backend's kernel descriptor requests (no VGPR workitem-id expansion
// bits are ever set). `TidX` is therefore `v0 & 0x3FF`; `TidY`/`TidZ` shift right by 10/20
// first.
//
// `Op::BidX/Y/Z`: real hardware preloads the workgroup id into an SGPR only when the kernel
// descriptor's `ENABLE_SGPR_WORKGROUP_ID_{X,Y,Z}` bit asks for it (`hsaco::HsacoSpec`'s
// `with_workgroup_ids`); this backend scans a function once for which axes its body actually
// reads and requests exactly those, then reads the resulting SGPR straight into the value's
// own VGPR with a single `v_mov_b32` (see "Register model" above for why this lands in a VGPR,
// not a scalar-file home).
//
// # Control flow and divergence — a documented, deliberate scope limit
//
// `Term::CondBr` lowers as a genuine data-dependent branch: the `i1` condition is re-derived
// into `vcc_lo` (`v_cmp_ne_u32 vcc_lo, 0, cond`) and then `s_cbranch_vccnz`/an unconditional
// fallback `s_branch` pick the taken block, with a small trampoline sequence on each edge that
// carries that edge's own phi copies before jumping to the real target block (always emitted,
// whether or not that edge actually has any phi to copy — trading a handful of trivial bytes
// for having exactly one code shape to reason about). This is **correct whenever every active
// lane in the wave agrees on the branch outcome** — true by construction for a single-lane wave
// (the shape this phase's own proof kernel is driven with) and true for any genuinely uniform
// branch, but **not** a general divergent-control-flow implementation: `vcc_lo`'s zero-ness
// only reflects "does some lane disagree," not "which," and this pass never saves/masks/
// restores `exec` the way real divergent control flow requires. That is real hardware SIMT's
// hardest remaining problem and is explicitly left to a later, dedicated task — this pass's job
// is a correct-first slice, not full generality, exactly like the CPU oracle's own register
// allocator started narrow before a later phase added real allocation.
//
// # Memory and synchronization
//
// `Load`/`Store`/`Atomic` on `AddrSpace::Global` go through `enc::flat_load`/`flat_store`/
// `flat_atomic` with `Seg::Global`, a full 64-bit VGPR-pair address (no `saddr`, matching that
// module's own "the form this crate always uses" note) and no synthesized loop — the load
// genuinely reads whatever lane's own address is in its own VGPR pair, real per-lane hardware
// addressing. `Shared`/`Local` go through `enc::ds_load`/`ds_store` (a 32-bit LDS offset in one
// VGPR). Every load and every atomic is followed by a blanket `s_waitcnt(0, 0, 0)` before its
// result is used — always waiting on every counter (vector-memory, export, LDS/constant/scalar)
// rather than tracking exactly which one applies is a deliberately conservative, always-correct
// choice over a cleverer, riskier one. `Barrier` always emits a real `s_waitcnt(0, 0, 0)` +
// `s_barrier` — genuine hardware wavefront synchronization, unlike the CPU oracle's no-op
// (`basalt-x86/src/oracle.rs`'s own reasoning: there, "concurrent threads" is a fiction created
// by one sequential loop, so nothing needs to actually wait; here, wavefronts genuinely execute
// concurrently in hardware, so the barrier is real). `Term::Ret` likewise drains
// `s_waitcnt(0, 0, 0)` before `s_endpgm`, so a kernel's final store is guaranteed complete
// before the wave reports itself done.
//
// # Kernarg segment layout
//
// Parameters are packed in declaration order at each type's natural size/alignment (pointers
// and `i64` at 8 bytes, `i32`/`f32` at 4) with no padding beyond that natural alignment. This
// coincides with `tests/diff/rdna3_sim/run_kernel.py`'s own kernarg-packing convention (buffers
// first at 8 bytes each, then scalars at 4 bytes each) only when every pointer parameter
// precedes every scalar parameter in the function's signature — true of every kernel this
// project's frontend currently produces (see that script's own header) and checked here
// (`E093`) rather than silently mismatching a param a real launcher would place elsewhere.

use std::collections::HashMap;

use basalt_backend::{Artifact, ArtifactKind, Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, BlockId, CastOp, FCmpPred, Function, ICmpPred, InstId, Module, Op,
    Scalar, Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};
use basalt_passes::construct_ssa;

use crate::enc::{
    self, BrCc, DsLoadOp, DsStoreOp, FlatOp, Imm, Seg, SmemOp, VCmpOp, VSrc, Vop1Op, Vop2Op,
    Vop3CarryOp, Vop3Mods, Vop3Op, VCC_LO,
};
use crate::hsaco::{write_hsaco, GfxArch, HsacoSpec};

/// `v0` is the hardware-preloaded packed thread index; SSA values start at `v1`.
const FIRST_FREE_VGPR: u16 = 1;
/// The highest legal VGPR number (`enc.rs`'s own field width: VGPRs are numbered 0-255).
const MAX_VGPR: u16 = 255;
/// `s[0:1]`: the kernarg segment base pointer, whenever the kernel takes any parameter.
const KERNARG_SGPR: u8 = 0;
/// `s2` is the first SGPR a workgroup-id axis can occupy (right after the kernarg pointer
/// pair); see the module header's packing rule.
const BID_SGPR_BASE: u8 = 2;
/// Fixed scratch pair for staging one kernarg value at a time during the prologue.
const STAGE_SGPR: u8 = 16;

fn e_type() -> Diag {
    Diag::new(ECode::UnsupportedType)
}

fn e_space() -> Diag {
    Diag::new(ECode::UnsupportedAddressSpace)
}

fn e_feature() -> Diag {
    Diag::new(ECode::UnsupportedFeature)
}

// ---- register width/class -------------------------------------------------------------------

/// How many consecutive VGPRs a value of this type occupies. `None` for anything out of this
/// backend's declared scope (see the module header) — never a guess at a plausible-looking
/// width.
fn vgpr_width(ty: Ty) -> Option<u8> {
    match ty {
        Ty::Void => Some(0),
        Ty::Scalar(Scalar::I1 | Scalar::I32 | Scalar::F32) => Some(1),
        Ty::Scalar(Scalar::I64) => Some(2),
        Ty::Ptr(AddrSpace::Shared | AddrSpace::Local) => Some(1),
        Ty::Ptr(AddrSpace::Global | AddrSpace::Constant | AddrSpace::Param) => Some(2),
        Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::F16 | Scalar::F64) | Ty::Vec(..) => None,
    }
}

/// Whether `ty` is a "wide" (two-VGPR, 64-bit) value — the pointer-arithmetic/generic-64-bit-add
/// carry-chain path applies to these; everything else takes the plain 32-bit VOP2 path.
fn is_wide(ty: Ty) -> bool {
    matches!(
        ty,
        Ty::Scalar(Scalar::I64)
            | Ty::Ptr(AddrSpace::Global | AddrSpace::Constant | AddrSpace::Param)
    )
}

fn is_ds_space(space: AddrSpace) -> bool {
    matches!(space, AddrSpace::Shared | AddrSpace::Local)
}

fn valref_ty(f: &Function, v: ValRef) -> Ty {
    match v {
        ValRef::Param(i) => f.params[i as usize],
        ValRef::Val(id) => f.insts[id.0 as usize].ty,
    }
}

// ---- register allocation: liveness-based reuse, still no divergence awareness --------------
//
// A pure "one permanent VGPR per SSA value, forever" scheme (this backend's first cut) cannot
// fit `stress.cu` — its own comment explains it is deliberately built to exceed a small fixed
// register file (18 simultaneously-live float temporaries) — into 255 available VGPRs: this
// function alone has over 200 instructions, most producing a value. Since spilling to LDS/
// scratch is the other option the design brief allows, and a value's live range is easy to
// compute exactly in a function with no loops (this backend's whole declared scope has no
// looping construct, see the module header), a simple linear-scan reuse is both simpler to get
// right than real spill-code and enough to fit every kernel in scope: a register becomes
// reusable the instant its value's last real use has been passed, and is handed to the next
// value that needs a register of the same width from then on. This is *not* the
// divergence-aware allocator a later, dedicated task will build (see the module header) — it
// has no notion of a value being uniform vs. divergent, no exec-mask interaction, and no
// spilling; it is exactly the amount of liveness tracking needed to fit a straight-line
// (or simply-branching) kernel's honest register pressure into a real, finite VGPR file.
//
// Correctness of the reuse rests on one fact: `Function::insts`, after `construct_ssa`, is laid
// out in the exact per-block program order it will be lowered in (see `basalt-bir`'s own module
// header — "instructions ... populated in program order" — and `construct_ssa`'s own header on
// building its output "in exactly the order it will be printed"), so `InstId.0` doubles as a
// safe linear timeline: a value's definition point is its own index (params before every
// instruction), and its last-use point is the highest index of any instruction that reads it —
// with terminator operands and `phi` incoming values conservatively pinned to "alive through the
// end of the function" (see `compute_last_use`), since those don't have a single instruction
// index of their own and a wrong answer there would silently corrupt a live value, not just
// waste a register.

/// Every `ValRef`-typed operand an `Op` reads — the "uses" a liveness scan needs. Purely a data
/// query (no interpretation of whether this backend actually lowers the op), so it stays exactly
/// as exhaustive as `Op` itself regardless of `check_module`'s own narrower scope.
fn op_operands(op: &Op) -> Vec<ValRef> {
    match op {
        Op::ConstInt(_) | Op::ConstFloat(_) => vec![],
        Op::Bin(_, a, b) => vec![*a, *b],
        Op::ICmp(_, _, a, b) => vec![*a, *b],
        Op::FCmp(_, _, a, b) => vec![*a, *b],
        Op::Select(c, a, b) => vec![*c, *a, *b],
        Op::Cast(_, _, v) => vec![*v],
        Op::Load { ptr, .. } => vec![*ptr],
        Op::Store { ptr, val, .. } => vec![*ptr, *val],
        Op::Phi(preds) => preds.iter().map(|&(_, v)| v).collect(),
        Op::TidX
        | Op::TidY
        | Op::TidZ
        | Op::BidX
        | Op::BidY
        | Op::BidZ
        | Op::BdimX
        | Op::BdimY
        | Op::BdimZ
        | Op::GdimX
        | Op::GdimY
        | Op::GdimZ
        | Op::Barrier => vec![],
        Op::Shuffle(_, a, b) => vec![*a, *b],
        Op::Ballot(v) | Op::VoteAny(v) | Op::VoteAll(v) => vec![*v],
        Op::Atomic(_, p, v, _) => vec![*p, *v],
        Op::AtomicCas(p, c, n, _) => vec![*p, *c, *n],
        Op::Mma { a, b, c, d, .. } => vec![*a, *b, *c, *d],
    }
}

/// `(param last-use point, inst last-use point)`, one entry each, `-1`/own-index by default for
/// a value with no recorded use (dead code the optimizer should already have removed; still
/// safe to allocate a same-point interval for). See the module header for the conservative
/// "end of function" pin on terminator operands and `phi` incoming values.
fn compute_last_use(f: &Function) -> (Vec<i64>, Vec<i64>) {
    let mut param_last: Vec<i64> = vec![-1; f.params.len()];
    let mut inst_last: Vec<i64> = (0..f.insts.len() as i64).collect();
    let record = |v: ValRef, point: i64, param_last: &mut [i64], inst_last: &mut [i64]| match v {
        ValRef::Param(i) => {
            let i = i as usize;
            if point > param_last[i] {
                param_last[i] = point;
            }
        }
        ValRef::Val(id) => {
            let i = id.0 as usize;
            if point > inst_last[i] {
                inst_last[i] = point;
            }
        }
    };
    for (idx, inst) in f.insts.iter().enumerate() {
        if let Op::Phi(preds) = &inst.op {
            // A phi's incoming value may be defined later in program order than the phi
            // itself appears (a loop-carried value); this backend has no loop in scope, but
            // pinning conservatively to "alive through the end" is always safe regardless.
            for &(_, v) in preds {
                record(v, f.insts.len() as i64, &mut param_last, &mut inst_last);
            }
        } else {
            for v in op_operands(&inst.op) {
                record(v, idx as i64, &mut param_last, &mut inst_last);
            }
        }
    }
    let end_point = f.insts.len() as i64;
    for block in &f.blocks {
        match &block.term {
            Term::Br(_) => {}
            Term::CondBr(c, _, _) => record(*c, end_point, &mut param_last, &mut inst_last),
            Term::Ret(Some(v)) => record(*v, end_point, &mut param_last, &mut inst_last),
            Term::Ret(None) => {}
            Term::Switch(s, _, _) => record(*s, end_point, &mut param_last, &mut inst_last),
        }
    }
    (param_last, inst_last)
}

/// Two independent free-register pools (one per width) sharing one bump pointer into
/// never-yet-touched VGPR numbers: a freed narrow (1-VGPR) slot only ever comes back for
/// another narrow value, a freed wide (2-VGPR) slot only for another wide one, so a reused slot
/// is always the exact width its new owner needs — no fragmentation bookkeeping required.
struct Pools {
    next_free_vgpr: u16,
    narrow_free: Vec<u8>,
    wide_free: Vec<u8>,
}

impl Pools {
    fn new() -> Pools {
        Pools {
            next_free_vgpr: FIRST_FREE_VGPR,
            narrow_free: Vec::new(),
            wide_free: Vec::new(),
        }
    }

    fn alloc(&mut self, width: u8) -> Result<Vec<u8>, Diag> {
        match width {
            0 => Ok(Vec::new()),
            1 => {
                if let Some(r) = self.narrow_free.pop() {
                    return Ok(vec![r]);
                }
                let r = self.next_free_vgpr;
                if r > MAX_VGPR {
                    return Err(e_feature());
                }
                self.next_free_vgpr += 1;
                Ok(vec![r as u8])
            }
            2 => {
                if let Some(base) = self.wide_free.pop() {
                    return Ok(vec![base, base + 1]);
                }
                let base = self.next_free_vgpr;
                if base + 1 > MAX_VGPR {
                    return Err(e_feature());
                }
                self.next_free_vgpr += 2;
                Ok(vec![base as u8, (base + 1) as u8])
            }
            _ => unreachable!("vgpr_width only ever returns 0, 1, or 2"),
        }
    }

    fn free(&mut self, regs: &[u8]) {
        match regs.len() {
            0 => {}
            1 => self.narrow_free.push(regs[0]),
            2 => self.wide_free.push(regs[0]),
            _ => unreachable!("vgpr_width only ever returns 0, 1, or 2"),
        }
    }
}

struct RegAlloc {
    param_vgpr: Vec<Vec<u8>>,
    inst_vgpr: Vec<Vec<u8>>,
}

impl RegAlloc {
    fn build(f: &Function) -> Result<RegAlloc, Diag> {
        let (param_last, inst_last) = compute_last_use(f);
        let mut pools = Pools::new();
        let mut active: Vec<(Vec<u8>, i64)> = Vec::new();

        let mut param_vgpr = Vec::with_capacity(f.params.len());
        for (i, &ty) in f.params.iter().enumerate() {
            let width = vgpr_width(ty).ok_or_else(e_type)?;
            let regs = pools.alloc(width)?;
            if width > 0 {
                active.push((regs.clone(), param_last[i]));
            }
            param_vgpr.push(regs);
        }

        let mut inst_vgpr = Vec::with_capacity(f.insts.len());
        for (idx, inst) in f.insts.iter().enumerate() {
            let point = idx as i64;
            active.retain(|(regs, last_use)| {
                if *last_use < point {
                    pools.free(regs);
                    false
                } else {
                    true
                }
            });
            let width = vgpr_width(inst.ty).ok_or_else(e_type)?;
            let regs = pools.alloc(width)?;
            if width > 0 {
                active.push((regs.clone(), inst_last[idx]));
            }
            inst_vgpr.push(regs);
        }

        Ok(RegAlloc {
            param_vgpr,
            inst_vgpr,
        })
    }

    fn val(&self, v: ValRef) -> &[u8] {
        match v {
            ValRef::Param(i) => &self.param_vgpr[i as usize],
            ValRef::Val(id) => &self.inst_vgpr[id.0 as usize],
        }
    }
}

// ---- kernarg segment layout -----------------------------------------------------------------

fn round_up(x: u32, align: u32) -> u32 {
    x.div_ceil(align) * align
}

/// `(byte offset, size)` for each parameter, natural-size/align packed in declaration order
/// (see the module header for why this matches the diff harness's own kernarg convention only
/// when every pointer precedes every scalar).
fn kernarg_layout(f: &Function) -> Result<(Vec<(u32, u32)>, u32), Diag> {
    let mut seen_scalar = false;
    let mut offsets = Vec::with_capacity(f.params.len());
    let mut cursor: u32 = 0;
    for &ty in &f.params {
        let size = match ty {
            Ty::Ptr(_) | Ty::Scalar(Scalar::I64) => 8,
            Ty::Scalar(Scalar::I32 | Scalar::F32) => 4,
            _ => return Err(e_type()),
        };
        let is_ptr = matches!(ty, Ty::Ptr(_));
        if is_ptr && seen_scalar {
            return Err(e_feature());
        }
        if !is_ptr {
            seen_scalar = true;
        }
        let offset = round_up(cursor, size);
        offsets.push((offset, size));
        cursor = offset + size;
    }
    Ok((offsets, cursor))
}

// ---- workgroup-id axis usage ------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct BidUsage {
    x: bool,
    y: bool,
    z: bool,
}

impl BidUsage {
    fn scan(f: &Function) -> BidUsage {
        let mut u = BidUsage::default();
        for inst in &f.insts {
            match inst.op {
                Op::BidX => u.x = true,
                Op::BidY => u.y = true,
                Op::BidZ => u.z = true,
                _ => {}
            }
        }
        u
    }

    /// The SGPR number for each requested axis, packed contiguously from `BID_SGPR_BASE` in
    /// x-then-y-then-z order, skipping any axis this function never reads — the same packing
    /// real hardware (and this project's own kernel descriptor) applies.
    fn sgprs(self) -> (Option<u8>, Option<u8>, Option<u8>) {
        let mut next = BID_SGPR_BASE;
        let take = |want: bool, next: &mut u8| -> Option<u8> {
            if want {
                let r = *next;
                *next += 1;
                Some(r)
            } else {
                None
            }
        };
        let x = take(self.x, &mut next);
        let y = take(self.y, &mut next);
        let z = take(self.z, &mut next);
        (x, y, z)
    }
}

// ---- phi resolution ---------------------------------------------------------------------------

/// `(from_block, to_block) -> [(phi's own InstId, incoming value)]`. Every SSA value owns its
/// VGPR home permanently (see the module header), so — exactly like `basalt-ptx`'s own phi
/// resolution — no staging is needed: an unconditional copy per incoming edge is always
/// correct.
type PhiCopies = HashMap<(u32, u32), Vec<(InstId, ValRef)>>;

fn build_phi_copies(f: &Function) -> PhiCopies {
    let mut map: PhiCopies = HashMap::new();
    for (bidx, block) in f.blocks.iter().enumerate() {
        for &inst_id in &block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            if let Op::Phi(preds) = &inst.op {
                for &(pred_block, val) in preds {
                    map.entry((pred_block.0, bidx as u32))
                        .or_default()
                        .push((inst_id, val));
                }
            }
        }
    }
    map
}

// ---- module validation ------------------------------------------------------------------------

fn check_bin(op: BinOp, ty: Ty) -> Result<(), Diag> {
    use BinOp::*;
    match op {
        Div | Rem | FDiv | FRem => Err(e_feature()),
        // `Add`/`Mul` have a real 64-bit lowering (the carry chain / cross-term-multiply
        // sequence in `lower_bin`); a 64-bit `Sub`/bitwise-or-shift op would need its own
        // multi-instruction sequence (borrow chain, ...) this task's time budget did not reach.
        Sub | And | Or | Xor | Shl | Lshr | Ashr if is_wide(ty) => Err(e_feature()),
        FAdd | FSub | FMul if ty != Ty::Scalar(Scalar::F32) => Err(e_type()),
        _ => Ok(()),
    }
}

fn check_cast(cop: CastOp, sty: Ty, dty: Ty) -> Result<(), Diag> {
    use CastOp::*;
    use Scalar::*;
    let i1 = Ty::Scalar(I1);
    let i32_ = Ty::Scalar(I32);
    let i64_ = Ty::Scalar(I64);
    let f32_ = Ty::Scalar(F32);
    let ok = match cop {
        Trunc => (sty, dty) == (i32_, i1) || (sty, dty) == (i64_, i32_),
        Zext | Sext => (sty, dty) == (i1, i32_) || (sty, dty) == (i32_, i64_),
        FpToSi | FpToUi => (sty, dty) == (f32_, i32_),
        SiToFp | UiToFp => (sty, dty) == (i32_, f32_),
        Bitcast => sty == dty || (sty, dty) == (i32_, f32_) || (sty, dty) == (f32_, i32_),
        FpTrunc | FpExt => false,
    };
    if ok {
        Ok(())
    } else {
        Err(e_type())
    }
}

fn check_function(f: &Function) -> Result<(), Diag> {
    for &ty in &f.params {
        vgpr_width(ty).ok_or_else(e_type)?;
    }
    if vgpr_width(f.ret).is_none() {
        return Err(e_type());
    }
    for inst in &f.insts {
        vgpr_width(inst.ty).ok_or_else(e_type)?;
        match &inst.op {
            Op::ConstInt(_) => {
                if !matches!(inst.ty, Ty::Scalar(Scalar::I1 | Scalar::I32 | Scalar::I64)) {
                    return Err(e_type());
                }
            }
            Op::ConstFloat(_) => {
                if inst.ty != Ty::Scalar(Scalar::F32) {
                    return Err(e_type());
                }
            }
            Op::Bin(op, _a, b) => {
                check_bin(*op, inst.ty)?;
                if matches!(op, BinOp::Add)
                    && is_wide(inst.ty)
                    && valref_ty(f, *b) != Ty::Scalar(Scalar::I64)
                {
                    return Err(e_type());
                }
            }
            Op::ICmp(_, cty, _, _) => {
                if !matches!(cty, Ty::Scalar(Scalar::I1 | Scalar::I32)) {
                    return Err(e_type());
                }
            }
            Op::FCmp(_, cty, _, _) => {
                if *cty != Ty::Scalar(Scalar::F32) {
                    return Err(e_type());
                }
            }
            Op::Select(..) => {}
            Op::Cast(cop, sty, _v) => {
                check_cast(*cop, *sty, inst.ty)?;
            }
            Op::Load { space, .. } => {
                if matches!(space, AddrSpace::Param | AddrSpace::Constant) {
                    return Err(e_space());
                }
                if is_ds_space(*space) && is_wide(inst.ty) {
                    return Err(e_feature());
                }
            }
            Op::Store { ty, space, .. } => {
                if matches!(space, AddrSpace::Param | AddrSpace::Constant) {
                    return Err(e_space());
                }
                if is_ds_space(*space) && is_wide(*ty) {
                    return Err(e_feature());
                }
            }
            Op::Phi(_) => {}
            Op::TidX | Op::TidY | Op::TidZ | Op::BidX | Op::BidY | Op::BidZ => {}
            Op::BdimX | Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ => {
                return Err(e_feature());
            }
            Op::Barrier => {}
            Op::Shuffle(..) | Op::Ballot(_) | Op::VoteAny(_) | Op::VoteAll(_) => {
                return Err(e_feature());
            }
            Op::Atomic(_, _, _, space) => {
                if *space != AddrSpace::Global {
                    return Err(e_space());
                }
                if inst.ty != Ty::Scalar(Scalar::I32) {
                    return Err(e_type());
                }
            }
            Op::AtomicCas(..) => return Err(e_feature()),
            Op::Mma { .. } => return Err(Diag::new(ECode::MatrixPathUnsupported)),
        }
    }
    for block in &f.blocks {
        match &block.term {
            Term::Br(_) | Term::CondBr(..) | Term::Ret(_) => {}
            Term::Switch(..) => return Err(e_feature()),
        }
    }
    RegAlloc::build(f).map(|_| ())?;
    kernarg_layout(f).map(|_| ())?;
    Ok(())
}

fn check_module(module: &Module) -> Result<(), Diag> {
    if module.funcs.len() != 1 {
        return Err(e_feature());
    }
    check_function(&module.funcs[0])
}

// ---- code generation --------------------------------------------------------------------------

fn icmp_vcmp(pred: ICmpPred) -> VCmpOp {
    use ICmpPred::*;
    match pred {
        Eq => VCmpOp::EqI32,
        Ne => VCmpOp::NeI32,
        Slt => VCmpOp::LtI32,
        Sle => VCmpOp::LeI32,
        Sgt => VCmpOp::GtI32,
        Sge => VCmpOp::GeI32,
        Ult => VCmpOp::LtU32,
        Ule => VCmpOp::LeU32,
        Ugt => VCmpOp::GtU32,
        Uge => VCmpOp::GeU32,
    }
}

fn fcmp_vcmp(pred: FCmpPred) -> VCmpOp {
    use FCmpPred::*;
    match pred {
        Oeq => VCmpOp::EqF32,
        One => VCmpOp::LgF32,
        Olt => VCmpOp::LtF32,
        Ole => VCmpOp::LeF32,
        Ogt => VCmpOp::GtF32,
        Oge => VCmpOp::GeF32,
        Ord => VCmpOp::OF32,
        Uno => VCmpOp::UF32,
    }
}

fn atomic_flatop(op: AtomicOp) -> FlatOp {
    match op {
        AtomicOp::Add => FlatOp::AtomicAddU32,
        AtomicOp::Sub => FlatOp::AtomicSubU32,
        AtomicOp::Exch => FlatOp::AtomicSwapB32,
        AtomicOp::Min => FlatOp::AtomicSminI32,
        AtomicOp::Max => FlatOp::AtomicSmaxI32,
        AtomicOp::And => FlatOp::AtomicAndB32,
        AtomicOp::Or => FlatOp::AtomicOrB32,
        AtomicOp::Xor => FlatOp::AtomicXorB32,
    }
}

enum BranchTarget {
    Block(u32),
}

struct CodeGen<'a> {
    f: &'a Function,
    alloc: RegAlloc,
    phi_copies: PhiCopies,
    bid_sgpr: (Option<u8>, Option<u8>, Option<u8>),
    code: Vec<u8>,
    block_start: HashMap<u32, usize>,
    pending: Vec<(usize, BranchTarget)>,
}

impl<'a> CodeGen<'a> {
    fn push(&mut self, bytes: Vec<u8>) {
        self.code.extend_from_slice(&bytes);
    }

    /// Emits a branch/cbranch with a zero placeholder offset, deferring the real patch until
    /// every block's start address is known.
    fn push_branch_to_block(&mut self, bytes: Vec<u8>, target: BlockId) {
        let pos = self.code.len();
        self.push(bytes);
        self.pending.push((pos, BranchTarget::Block(target.0)));
    }

    /// Patches a placeholder branch/cbranch at `pos` to target the buffer's *current* end —
    /// used for a trampoline whose start address is already known at the point its incoming
    /// branch was emitted.
    fn patch_to_current(&mut self, pos: usize) {
        let target = self.code.len() as i64;
        let off = (target - (pos as i64 + 4)) / 4;
        self.code[pos..pos + 2].copy_from_slice(&(off as i16).to_le_bytes());
    }

    fn resolve_pending(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        for (pos, target) in pending {
            let BranchTarget::Block(bid) = target;
            let addr = self.block_start[&bid];
            let off = (addr as i64 - (pos as i64 + 4)) / 4;
            self.code[pos..pos + 2].copy_from_slice(&(off as i16).to_le_bytes());
        }
    }

    fn dst(&self, id: InstId) -> &[u8] {
        &self.alloc.inst_vgpr[id.0 as usize]
    }

    fn val(&self, v: ValRef) -> &[u8] {
        self.alloc.val(v)
    }

    fn mov(&mut self, dst: u8, src: u8) {
        if dst != src {
            self.push(enc::vop1(Vop1Op::MovB32, dst, VSrc::Vgpr(src)));
        }
    }

    fn mov_imm(&mut self, dst: u8, imm: Imm) {
        self.push(enc::vop1(Vop1Op::MovB32, dst, VSrc::Imm(imm)));
    }

    fn waitcnt_all(&mut self) {
        self.push(enc::s_waitcnt(0, 0, 0));
    }

    fn materialize_bool(&mut self, dst: u8) {
        self.push(enc::vop3(
            Vop3Op::CndmaskB32,
            dst,
            VSrc::Imm(Imm::Int(0)),
            VSrc::Imm(Imm::Int(1)),
            VSrc::Sgpr(VCC_LO),
            Vop3Mods::default(),
        ));
    }

    // ---- prologue -------------------------------------------------------------------------

    fn emit_prologue(&mut self, offsets: &[(u32, u32)], total_size: u32) {
        if total_size == 0 {
            return;
        }
        for (i, &(offset, size)) in offsets.iter().enumerate() {
            let dst_regs = self.alloc.param_vgpr[i].clone();
            let op = if size == 4 {
                SmemOp::LoadB32
            } else {
                SmemOp::LoadB64
            };
            self.push(enc::smem_load(
                op,
                STAGE_SGPR,
                KERNARG_SGPR,
                offset as i32,
                None,
                false,
                false,
            ));
            self.waitcnt_all();
            for (j, &r) in dst_regs.iter().enumerate() {
                self.push(enc::vop1(
                    Vop1Op::MovB32,
                    r,
                    VSrc::Sgpr(STAGE_SGPR + j as u8),
                ));
            }
        }
    }

    // ---- instruction dispatch ---------------------------------------------------------------

    fn lower_inst(&mut self, id: InstId) {
        let f = self.f;
        let inst = &f.insts[id.0 as usize];
        let ty = inst.ty;
        match &inst.op {
            Op::ConstInt(n) => self.lower_const_int(id, *n, ty),
            Op::ConstFloat(v) => self.lower_const_float(id, *v, ty),
            Op::Bin(op, a, b) => self.lower_bin(id, *op, *a, *b, ty),
            Op::ICmp(pred, _cty, a, b) => self.lower_icmp(id, *pred, *a, *b),
            Op::FCmp(pred, _cty, a, b) => self.lower_fcmp(id, *pred, *a, *b),
            Op::Select(c, a, b) => self.lower_select(id, *c, *a, *b),
            Op::Cast(cop, sty, v) => self.lower_cast(id, *cop, *sty, *v, ty),
            Op::Load { ptr, space, .. } => self.lower_load(id, *ptr, *space, ty),
            Op::Store {
                ptr,
                val,
                ty: sty,
                space,
                ..
            } => self.lower_store(*ptr, *val, *space, *sty),
            Op::Phi(_) => {
                // Resolved entirely at each predecessor's edge (`emit_phi_copies`); nothing to
                // do at the definition site itself.
            }
            Op::TidX => self.lower_tid(id, 0),
            Op::TidY => self.lower_tid(id, 10),
            Op::TidZ => self.lower_tid(id, 20),
            Op::BidX => {
                let s = self.bid_sgpr.0;
                self.lower_bid(id, s);
            }
            Op::BidY => {
                let s = self.bid_sgpr.1;
                self.lower_bid(id, s);
            }
            Op::BidZ => {
                let s = self.bid_sgpr.2;
                self.lower_bid(id, s);
            }
            Op::Barrier => {
                self.waitcnt_all();
                self.push(enc::s_barrier());
            }
            Op::Atomic(aop, ptr, val, _space) => self.lower_atomic(id, *aop, *ptr, *val),
            Op::BdimX
            | Op::BdimY
            | Op::BdimZ
            | Op::GdimX
            | Op::GdimY
            | Op::GdimZ
            | Op::Shuffle(..)
            | Op::Ballot(_)
            | Op::VoteAny(_)
            | Op::VoteAll(_)
            | Op::AtomicCas(..)
            | Op::Mma { .. } => {
                unreachable!("check_module refuses this construct before codegen starts")
            }
        }
    }

    fn lower_const_int(&mut self, id: InstId, n: i64, ty: Ty) {
        let regs = self.dst(id).to_vec();
        match ty {
            Ty::Scalar(Scalar::I1) => self.mov_imm(regs[0], Imm::Int((n & 1) as i32)),
            Ty::Scalar(Scalar::I32) => self.mov_imm(regs[0], Imm::Int(n as i32)),
            Ty::Scalar(Scalar::I64) => {
                self.mov_imm(regs[0], Imm::Raw(n as u32));
                self.mov_imm(regs[1], Imm::Raw((n >> 32) as u32));
            }
            _ => unreachable!("check_module restricts ConstInt to i1/i32/i64"),
        }
    }

    fn lower_const_float(&mut self, id: InstId, v: f64, ty: Ty) {
        let regs = self.dst(id).to_vec();
        match ty {
            Ty::Scalar(Scalar::F32) => self.mov_imm(regs[0], Imm::F32(v as f32)),
            _ => unreachable!("check_module restricts ConstFloat to f32"),
        }
    }

    fn lower_bin(&mut self, id: InstId, op: BinOp, a: ValRef, b: ValRef, ty: Ty) {
        let dst = self.dst(id).to_vec();
        let a_regs = self.val(a).to_vec();
        let b_regs = self.val(b).to_vec();
        if is_wide(ty) {
            match op {
                BinOp::Add => {
                    self.push(enc::vop3_carry(
                        Vop3CarryOp::AddCoU32,
                        dst[0],
                        VCC_LO,
                        VSrc::Vgpr(a_regs[0]),
                        VSrc::Vgpr(b_regs[0]),
                        VSrc::Sgpr(0),
                    ));
                    self.push(enc::vop2(
                        Vop2Op::AddCoCiU32,
                        dst[1],
                        VSrc::Vgpr(a_regs[1]),
                        b_regs[1],
                    ));
                }
                BinOp::Mul => self.lower_wide_mul(&dst, &a_regs, &b_regs),
                _ => unreachable!("check_module restricts wide Bin to Add/Mul"),
            }
        } else {
            self.lower_narrow_bin(dst[0], op, a_regs[0], b_regs[0]);
        }
    }

    /// 64-bit `a * b`, truncated to 64 bits (matching `BinOp::Mul`'s wraparound semantics):
    /// the standard cross-term bignum-multiply formula, needing no scratch register beyond
    /// `dst` itself — `dst[1]` accumulates the high word first, and `dst[0]` is used as
    /// throwaway scratch for each cross term until the very last step, when it finally receives
    /// the true low word. `a_hi*b_hi` never contributes to the low 64 bits of the product, so
    /// it is correctly never computed.
    fn lower_wide_mul(&mut self, dst: &[u8], a: &[u8], b: &[u8]) {
        let (a_lo, a_hi) = (a[0], a[1]);
        let (b_lo, b_hi) = (b[0], b[1]);
        let mulhi = |cg: &mut Self, d: u8, x: u8, y: u8| {
            cg.push(enc::vop3(
                Vop3Op::MulHiU32,
                d,
                VSrc::Vgpr(x),
                VSrc::Vgpr(y),
                VSrc::Sgpr(0),
                Vop3Mods::default(),
            ));
        };
        let mullo = |cg: &mut Self, d: u8, x: u8, y: u8| {
            cg.push(enc::vop3(
                Vop3Op::MulLoU32,
                d,
                VSrc::Vgpr(x),
                VSrc::Vgpr(y),
                VSrc::Sgpr(0),
                Vop3Mods::default(),
            ));
        };
        mulhi(self, dst[1], a_lo, b_lo); // dst[1] = high(a_lo * b_lo)
        mullo(self, dst[0], a_lo, b_hi); // dst[0] = a_lo * b_hi (scratch)
        self.push(enc::vop2(
            Vop2Op::AddNcU32,
            dst[1],
            VSrc::Vgpr(dst[1]),
            dst[0],
        ));
        mullo(self, dst[0], a_hi, b_lo); // dst[0] = a_hi * b_lo (scratch)
        self.push(enc::vop2(
            Vop2Op::AddNcU32,
            dst[1],
            VSrc::Vgpr(dst[1]),
            dst[0],
        ));
        mullo(self, dst[0], a_lo, b_lo); // dst[0] = the real low word, computed last
    }

    fn lower_narrow_bin(&mut self, dst: u8, op: BinOp, a: u8, b: u8) {
        use BinOp::*;
        match op {
            Add => self.push(enc::vop2(Vop2Op::AddNcU32, dst, VSrc::Vgpr(a), b)),
            Sub => self.push(enc::vop2(Vop2Op::SubNcU32, dst, VSrc::Vgpr(a), b)),
            And => self.push(enc::vop2(Vop2Op::AndB32, dst, VSrc::Vgpr(a), b)),
            Or => self.push(enc::vop2(Vop2Op::OrB32, dst, VSrc::Vgpr(a), b)),
            Xor => self.push(enc::vop2(Vop2Op::XorB32, dst, VSrc::Vgpr(a), b)),
            Shl => self.push(enc::vop2(Vop2Op::LshlrevB32, dst, VSrc::Vgpr(b), a)),
            Lshr => self.push(enc::vop2(Vop2Op::LshrrevB32, dst, VSrc::Vgpr(b), a)),
            Ashr => self.push(enc::vop2(Vop2Op::AshrrevI32, dst, VSrc::Vgpr(b), a)),
            Mul => self.push(enc::vop3(
                Vop3Op::MulLoU32,
                dst,
                VSrc::Vgpr(a),
                VSrc::Vgpr(b),
                VSrc::Sgpr(0),
                Vop3Mods::default(),
            )),
            FAdd => self.push(enc::vop2(Vop2Op::AddF32, dst, VSrc::Vgpr(a), b)),
            FSub => self.push(enc::vop2(Vop2Op::SubF32, dst, VSrc::Vgpr(a), b)),
            FMul => self.push(enc::vop2(Vop2Op::MulF32, dst, VSrc::Vgpr(a), b)),
            Div | Rem | FDiv | FRem => {
                unreachable!("check_module refuses div/rem before codegen starts")
            }
        }
    }

    fn lower_icmp(&mut self, id: InstId, pred: ICmpPred, a: ValRef, b: ValRef) {
        let dst = self.dst(id)[0];
        let a_r = self.val(a)[0];
        let b_r = self.val(b)[0];
        self.push(enc::vopc_e32(icmp_vcmp(pred), VSrc::Vgpr(a_r), b_r));
        self.materialize_bool(dst);
    }

    fn lower_fcmp(&mut self, id: InstId, pred: FCmpPred, a: ValRef, b: ValRef) {
        let dst = self.dst(id)[0];
        let a_r = self.val(a)[0];
        let b_r = self.val(b)[0];
        self.push(enc::vopc_e32(fcmp_vcmp(pred), VSrc::Vgpr(a_r), b_r));
        self.materialize_bool(dst);
    }

    fn lower_select(&mut self, id: InstId, c: ValRef, a: ValRef, b: ValRef) {
        let cond = self.val(c)[0];
        self.push(enc::vopc_e32(VCmpOp::NeI32, VSrc::Imm(Imm::Int(0)), cond));
        let dst = self.dst(id).to_vec();
        let a_r = self.val(a).to_vec();
        let b_r = self.val(b).to_vec();
        for i in 0..dst.len() {
            self.push(enc::vop3(
                Vop3Op::CndmaskB32,
                dst[i],
                VSrc::Vgpr(b_r[i]),
                VSrc::Vgpr(a_r[i]),
                VSrc::Sgpr(VCC_LO),
                Vop3Mods::default(),
            ));
        }
    }

    fn lower_cast(&mut self, id: InstId, cop: CastOp, sty: Ty, v: ValRef, dty: Ty) {
        let dst = self.dst(id).to_vec();
        let src = self.val(v).to_vec();
        let i1 = Ty::Scalar(Scalar::I1);
        let i32_ = Ty::Scalar(Scalar::I32);
        let i64_ = Ty::Scalar(Scalar::I64);
        let f32_ = Ty::Scalar(Scalar::F32);
        match cop {
            CastOp::Trunc => {
                if (sty, dty) == (i32_, i1) {
                    self.push(enc::vop2(
                        Vop2Op::AndB32,
                        dst[0],
                        VSrc::Imm(Imm::Int(1)),
                        src[0],
                    ));
                } else if (sty, dty) == (i64_, i32_) {
                    self.mov(dst[0], src[0]);
                } else {
                    unreachable!("check_cast restricts Trunc to i32->i1 or i64->i32");
                }
            }
            CastOp::Zext => {
                if (sty, dty) == (i1, i32_) {
                    self.mov(dst[0], src[0]);
                } else if (sty, dty) == (i32_, i64_) {
                    self.mov(dst[0], src[0]);
                    self.mov_imm(dst[1], Imm::Int(0));
                } else {
                    unreachable!("check_cast restricts Zext to i1->i32 or i32->i64");
                }
            }
            CastOp::Sext => {
                if (sty, dty) == (i1, i32_) {
                    self.push(enc::vop2(
                        Vop2Op::LshlrevB32,
                        dst[0],
                        VSrc::Imm(Imm::Int(31)),
                        src[0],
                    ));
                    self.push(enc::vop2(
                        Vop2Op::AshrrevI32,
                        dst[0],
                        VSrc::Imm(Imm::Int(31)),
                        dst[0],
                    ));
                } else if (sty, dty) == (i32_, i64_) {
                    self.mov(dst[0], src[0]);
                    self.push(enc::vop2(
                        Vop2Op::AshrrevI32,
                        dst[1],
                        VSrc::Imm(Imm::Int(31)),
                        dst[0],
                    ));
                } else {
                    unreachable!("check_cast restricts Sext to i1->i32 or i32->i64");
                }
            }
            CastOp::FpToSi => self.push(enc::vop1(Vop1Op::CvtI32F32, dst[0], VSrc::Vgpr(src[0]))),
            CastOp::FpToUi => self.push(enc::vop1(Vop1Op::CvtU32F32, dst[0], VSrc::Vgpr(src[0]))),
            CastOp::SiToFp => self.push(enc::vop1(Vop1Op::CvtF32I32, dst[0], VSrc::Vgpr(src[0]))),
            CastOp::UiToFp => self.push(enc::vop1(Vop1Op::CvtF32U32, dst[0], VSrc::Vgpr(src[0]))),
            CastOp::Bitcast => {
                for i in 0..dst.len() {
                    self.mov(dst[i], src[i]);
                }
            }
            CastOp::FpTrunc | CastOp::FpExt => {
                unreachable!("check_cast refuses FpTrunc/FpExt (f64 is out of scope)")
            }
        }
        let _ = f32_;
    }

    fn width_load_store(ty: Ty) -> (bool, u32) {
        match ty {
            Ty::Scalar(Scalar::I1) => (false, 1),
            Ty::Scalar(Scalar::I32 | Scalar::F32) => (false, 4),
            Ty::Ptr(AddrSpace::Shared | AddrSpace::Local) => (false, 4),
            Ty::Scalar(Scalar::I64) => (true, 8),
            Ty::Ptr(_) => (true, 8),
            _ => unreachable!(
                "check_module restricts Load/Store to this backend's scalar/pointer scope"
            ),
        }
    }

    fn lower_load(&mut self, id: InstId, ptr: ValRef, space: AddrSpace, ty: Ty) {
        let addr = self.val(ptr).to_vec();
        let dst = self.dst(id).to_vec();
        let (wide, bytes) = Self::width_load_store(ty);
        if is_ds_space(space) {
            let op = match bytes {
                1 => DsLoadOp::U8,
                4 => DsLoadOp::B32,
                _ => unreachable!("check_module refuses a wide DS Load"),
            };
            self.push(enc::ds_load(op, dst[0], addr[0], 0));
        } else {
            let op = match (wide, bytes) {
                (false, 1) => FlatOp::LoadU8,
                (false, 4) => FlatOp::LoadB32,
                (true, 8) => FlatOp::LoadB64,
                _ => unreachable!(),
            };
            self.push(enc::flat_load(
                Seg::Global,
                op,
                dst[0],
                addr[0],
                None,
                0,
                false,
            ));
        }
        self.waitcnt_all();
    }

    fn lower_store(&mut self, ptr: ValRef, val: ValRef, space: AddrSpace, ty: Ty) {
        let addr = self.val(ptr).to_vec();
        let data = self.val(val).to_vec();
        let (wide, bytes) = Self::width_load_store(ty);
        if is_ds_space(space) {
            let op = match bytes {
                1 => DsStoreOp::B8,
                4 => DsStoreOp::B32,
                _ => unreachable!("check_module refuses a wide DS Store"),
            };
            self.push(enc::ds_store(op, addr[0], data[0], 0));
        } else {
            let op = match (wide, bytes) {
                (false, 1) => FlatOp::StoreB8,
                (false, 4) => FlatOp::StoreB32,
                (true, 8) => FlatOp::StoreB64,
                _ => unreachable!(),
            };
            self.push(enc::flat_store(
                Seg::Global,
                op,
                addr[0],
                data[0],
                None,
                0,
                false,
            ));
        }
    }

    fn lower_tid(&mut self, id: InstId, shift: u32) {
        let dst = self.dst(id)[0];
        if shift == 0 {
            self.push(enc::vop2(
                Vop2Op::AndB32,
                dst,
                VSrc::Imm(Imm::Int(0x3FF)),
                0,
            ));
        } else {
            self.push(enc::vop2(
                Vop2Op::LshrrevB32,
                dst,
                VSrc::Imm(Imm::Int(shift as i32)),
                0,
            ));
            self.push(enc::vop2(
                Vop2Op::AndB32,
                dst,
                VSrc::Imm(Imm::Int(0x3FF)),
                dst,
            ));
        }
    }

    fn lower_bid(&mut self, id: InstId, sgpr: Option<u8>) {
        let dst = self.dst(id)[0];
        let s = sgpr.expect("BidUsage::scan found this axis used, so its SGPR must be assigned");
        self.push(enc::vop1(Vop1Op::MovB32, dst, VSrc::Sgpr(s)));
    }

    fn lower_atomic(&mut self, id: InstId, aop: AtomicOp, ptr: ValRef, val: ValRef) {
        let dst = self.dst(id)[0];
        let addr = self.val(ptr)[0];
        let data = self.val(val)[0];
        self.push(enc::flat_atomic(
            Seg::Global,
            atomic_flatop(aop),
            Some(dst),
            addr,
            data,
            None,
            0,
        ));
        self.waitcnt_all();
    }

    // ---- phi / terminators ------------------------------------------------------------------

    fn emit_phi_copies(&mut self, from: u32, to: u32) {
        let Some(copies) = self.phi_copies.get(&(from, to)).cloned() else {
            return;
        };
        for (phi_id, val) in copies {
            let dst = self.dst(phi_id).to_vec();
            let src = self.val(val).to_vec();
            for i in 0..dst.len() {
                self.mov(dst[i], src[i]);
            }
        }
    }

    fn lower_term(&mut self, from: u32, term: &Term) {
        match term {
            Term::Br(target) => {
                self.emit_phi_copies(from, target.0);
                self.push_branch_to_block(enc::s_branch(0), *target);
            }
            Term::CondBr(cond, t, f) => {
                let c = self.val(*cond)[0];
                self.push(enc::vopc_e32(VCmpOp::NeI32, VSrc::Imm(Imm::Int(0)), c));
                let cbranch_pos = self.code.len();
                self.push(enc::s_cbranch(BrCc::Vccnz, 0));
                let fallback_pos = self.code.len();
                self.push(enc::s_branch(0));
                self.patch_to_current(cbranch_pos);
                self.emit_phi_copies(from, t.0);
                self.push_branch_to_block(enc::s_branch(0), *t);
                self.patch_to_current(fallback_pos);
                self.emit_phi_copies(from, f.0);
                self.push_branch_to_block(enc::s_branch(0), *f);
            }
            Term::Ret(_) => {
                // A kernel entry point has no way to hand a value back to the host, so a
                // non-void `Ret` simply drops its value — matching `basalt-ptx`'s own
                // documented precedent for the identical reason.
                self.waitcnt_all();
                self.push(enc::s_endpgm());
            }
            Term::Switch(..) => unreachable!("check_module refuses Term::Switch"),
        }
    }
}

/// The hand-rolled RDNA3 (gfx1100) backend. `name()` returns `"amdgpu"` — a stable identifier a
/// later CLI wire-up would register under `--amdgpu` (that wiring is not this task's job; see
/// the module header's scope note).
#[derive(Debug, Default, Clone, Copy)]
pub struct Amdgcn;

impl Backend for Amdgcn {
    fn name(&self) -> &'static str {
        "amdgpu"
    }

    fn supports(&self, module: &Module) -> Support {
        match check_module(module) {
            Ok(()) => Support::Supported,
            Err(diag) => Support::Unsupported(diag.code),
        }
    }

    fn emit(&self, module: &Module, _opts: &EmitOpts) -> Result<Artifact, Diag> {
        check_module(module)?;
        let ssa_module = construct_ssa(module);
        let f = &ssa_module.funcs[0];
        let bytes = lower_function(f)?;
        Ok(Artifact::bytes(ArtifactKind::Object, bytes))
    }
}

fn lower_function(f: &Function) -> Result<Vec<u8>, Diag> {
    let alloc = RegAlloc::build(f)?;
    let (offsets, total_size) = kernarg_layout(f)?;
    let bid = BidUsage::scan(f);
    let bid_sgpr = bid.sgprs();
    let phi_copies = build_phi_copies(f);
    let mut cg = CodeGen {
        f,
        alloc,
        phi_copies,
        bid_sgpr,
        code: Vec::new(),
        block_start: HashMap::new(),
        pending: Vec::new(),
    };
    cg.emit_prologue(&offsets, total_size);
    for (bidx, block) in f.blocks.iter().enumerate() {
        cg.block_start.insert(bidx as u32, cg.code.len());
        for &inst_id in &block.insts {
            cg.lower_inst(inst_id);
        }
        cg.lower_term(bidx as u32, &block.term);
    }
    cg.resolve_pending();

    let vgpr_count = {
        let mut top: u16 = FIRST_FREE_VGPR;
        for regs in cg.alloc.param_vgpr.iter().chain(cg.alloc.inst_vgpr.iter()) {
            if let Some(&last) = regs.last() {
                top = top.max(last as u16 + 1);
            }
        }
        top as u32
    };

    let spec = HsacoSpec::new(GfxArch::Gfx1100, f.name.clone(), cg.code)
        .with_kernarg_segment(total_size, 8)
        .with_workgroup_ids(bid.x, bid.y, bid.z)
        .with_register_counts(vgpr_count, 32, 0, 0);
    write_hsaco(&spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{Block, Inst};

    fn wrap(f: Function) -> Module {
        Module {
            funcs: vec![f],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    /// `store f32 ptr.global %arg0, (const.f 1.0)`: the smallest real kernel entirely inside
    /// this backend's declared scope.
    fn func_store_const() -> Function {
        Function {
            name: "store_const".into(),
            params: vec![Ty::Ptr(AddrSpace::Global)],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: Ty::Scalar(Scalar::F32),
                    op: Op::ConstFloat(1.0),
                },
                Inst {
                    ty: Ty::Void,
                    op: Op::Store {
                        ptr: ValRef::Param(0),
                        val: ValRef::Val(InstId(0)),
                        ty: Ty::Scalar(Scalar::F32),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                },
            ],
            blocks: vec![Block {
                insts: vec![InstId(0), InstId(1)],
                term: Term::Ret(None),
            }],
        }
    }

    /// `%1 = tid.x; %2 = icmp slt %1, %arg1; condbr %2, bb1, bb2` — a real branch with a phi
    /// merging both arms, exercising `CondBr`/`Phi`/`Br` together.
    fn func_branch_with_phi() -> Function {
        Function {
            name: "branch_phi".into(),
            params: vec![Ty::Ptr(AddrSpace::Global), Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::TidX,
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I1),
                    op: Op::ICmp(
                        ICmpPred::Slt,
                        Ty::Scalar(Scalar::I32),
                        ValRef::Val(InstId(0)),
                        ValRef::Param(1),
                    ),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::ConstInt(1),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::ConstInt(2),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::Phi(vec![
                        (BlockId(1), ValRef::Val(InstId(2))),
                        (BlockId(2), ValRef::Val(InstId(3))),
                    ]),
                },
                Inst {
                    ty: Ty::Void,
                    op: Op::Store {
                        ptr: ValRef::Param(0),
                        val: ValRef::Val(InstId(4)),
                        ty: Ty::Scalar(Scalar::I32),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                },
            ],
            blocks: vec![
                Block {
                    insts: vec![InstId(0), InstId(1)],
                    term: Term::CondBr(ValRef::Val(InstId(1)), BlockId(1), BlockId(2)),
                },
                Block {
                    insts: vec![InstId(2)],
                    term: Term::Br(BlockId(3)),
                },
                Block {
                    insts: vec![InstId(3)],
                    term: Term::Br(BlockId(3)),
                },
                Block {
                    insts: vec![InstId(4), InstId(5)],
                    term: Term::Ret(None),
                },
            ],
        }
    }

    #[test]
    fn supports_kernels_using_only_implemented_ops() {
        assert_eq!(
            Amdgcn.supports(&wrap(func_store_const())),
            Support::Supported
        );
        assert_eq!(
            Amdgcn.supports(&wrap(func_branch_with_phi())),
            Support::Supported
        );
    }

    #[test]
    fn emits_a_valid_deterministic_hsaco_for_a_simple_kernel() {
        let module = wrap(func_store_const());
        let a = Amdgcn.emit(&module, &EmitOpts::default()).unwrap();
        let b = Amdgcn.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(a, b, "same module in must produce byte-identical bytes out");
        let bytes = a.as_bytes().unwrap();
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn branch_with_phi_produces_deterministic_bytes() {
        let module = wrap(func_branch_with_phi());
        let a = Amdgcn.emit(&module, &EmitOpts::default()).unwrap();
        let b = Amdgcn.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn refuses_i8_type_with_e091() {
        let f = Function {
            name: "i8_val".into(),
            params: vec![Ty::Scalar(Scalar::I8)],
            ret: Ty::Void,
            insts: vec![],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            Amdgcn.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_integer_div_with_e093() {
        let f = Function {
            name: "idiv".into(),
            params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Bin(BinOp::Div, ValRef::Param(0), ValRef::Param(1)),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        };
        assert_eq!(
            Amdgcn.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_shuffle_and_vote_with_e093() {
        for op in [
            Op::Shuffle(
                basalt_bir::ShuffleKind::Idx,
                ValRef::Param(0),
                ValRef::Param(1),
            ),
            Op::Ballot(ValRef::Param(0)),
            Op::VoteAny(ValRef::Param(0)),
            Op::VoteAll(ValRef::Param(0)),
        ] {
            let f = Function {
                name: "warp_op".into(),
                params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
                ret: Ty::Void,
                insts: vec![Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op,
                }],
                blocks: vec![Block {
                    insts: vec![InstId(0)],
                    term: Term::Ret(None),
                }],
            };
            assert_eq!(
                Amdgcn.supports(&wrap(f)),
                Support::Unsupported(ECode::UnsupportedFeature)
            );
        }
    }

    #[test]
    fn refuses_atomic_cas_with_e093() {
        let f = Function {
            name: "cas".into(),
            params: vec![
                Ty::Ptr(AddrSpace::Global),
                Ty::Scalar(Scalar::I32),
                Ty::Scalar(Scalar::I32),
            ],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::AtomicCas(
                    ValRef::Param(0),
                    ValRef::Param(1),
                    ValRef::Param(2),
                    AddrSpace::Global,
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        };
        assert_eq!(
            Amdgcn.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_switch_terminator_with_e093() {
        let f = Function {
            name: "sw".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![],
            blocks: vec![
                Block {
                    insts: vec![],
                    term: Term::Switch(ValRef::Param(0), BlockId(1), vec![(0, BlockId(1))]),
                },
                Block {
                    insts: vec![],
                    term: Term::Ret(None),
                },
            ],
        };
        assert_eq!(
            Amdgcn.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_param_space_load_with_e092() {
        // A surviving `AddrSpace::Param` pointer value (basalt-sema's synthetic slot pattern
        // construct_ssa is expected to eliminate — see the module header) fed straight in as a
        // parameter, so the refusal under test is `Load`'s own space check, not some earlier,
        // unrelated one.
        let f = Function {
            name: "param_load".into(),
            params: vec![Ty::Ptr(AddrSpace::Param)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Load {
                    ptr: ValRef::Param(0),
                    space: AddrSpace::Param,
                    align: 4,
                    volatile: false,
                },
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        };
        assert_eq!(
            Amdgcn.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedAddressSpace)
        );
    }

    #[test]
    fn refuses_bdim_and_gdim_with_e093() {
        for op in [
            Op::BdimX,
            Op::BdimY,
            Op::BdimZ,
            Op::GdimX,
            Op::GdimY,
            Op::GdimZ,
        ] {
            let f = Function {
                name: "dims".into(),
                params: vec![],
                ret: Ty::Scalar(Scalar::I32),
                insts: vec![Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op,
                }],
                blocks: vec![Block {
                    insts: vec![InstId(0)],
                    term: Term::Ret(Some(ValRef::Val(InstId(0)))),
                }],
            };
            assert_eq!(
                Amdgcn.supports(&wrap(f)),
                Support::Unsupported(ECode::UnsupportedFeature)
            );
        }
    }

    #[test]
    fn refuses_multi_function_module_with_e093() {
        let module = Module {
            funcs: vec![func_store_const(), func_store_const()],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        };
        assert_eq!(
            Amdgcn.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_mma_with_e099() {
        let f = Function {
            name: "mma".into(),
            params: vec![Ty::Ptr(AddrSpace::Global); 4],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Void,
                op: Op::Mma {
                    a: ValRef::Param(0),
                    b: ValRef::Param(1),
                    c: ValRef::Param(2),
                    d: ValRef::Param(3),
                    m: 16,
                    n: 16,
                    k: 16,
                    in_dtype: Scalar::F32,
                    acc_dtype: Scalar::F32,
                    layout_a: basalt_bir::MmaLayout::RowMajor,
                    layout_b: basalt_bir::MmaLayout::RowMajor,
                },
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            Amdgcn.supports(&wrap(f)),
            Support::Unsupported(ECode::MatrixPathUnsupported)
        );
    }

    /// `%2 = mul i64 %0, %1`: a 64-bit multiply of two `i64` params, exactly the shape
    /// `lower_wide_mul` handles (the pointer-index-scaling pattern `stress.cu` itself needs,
    /// generalized to two runtime values rather than one compile-time constant).
    fn func_wide_mul() -> Function {
        Function {
            name: "wide_mul".into(),
            params: vec![Ty::Scalar(Scalar::I64), Ty::Scalar(Scalar::I64)],
            ret: Ty::Scalar(Scalar::I64),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I64),
                op: Op::Bin(BinOp::Mul, ValRef::Param(0), ValRef::Param(1)),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    #[test]
    fn wide_multiply_is_supported_and_lowers_deterministically() {
        let module = wrap(func_wide_mul());
        assert_eq!(Amdgcn.supports(&module), Support::Supported);
        let a = Amdgcn.emit(&module, &EmitOpts::default()).unwrap();
        let b = Amdgcn.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(a, b);
    }
}
