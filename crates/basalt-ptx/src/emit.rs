// Hand-rolled NVIDIA PTX text emitter: the project's first target that runs on genuine SIMT
// hardware rather than emulating one thread at a time on a CPU core (contrast `basalt-x86`'s
// oracle/regalloc backends, which both synthesize a native per-thread loop because a CPU core
// has no real hardware threading to lean on). A BIR function's blocks/instructions translate
// directly, one to one, into a PTX kernel body: `tid.x`/`bid.x`/etc. become reads of PTX's own
// predefined special registers, and there is no synthesized loop or trailing `nthreads`
// parameter anywhere in this file. Grid/block launch dimensions are a host-side, launch-time
// concern (the CUDA Driver API) entirely outside this backend's scope.
//
// # Toolchain target
//
// `.version 8.0`, `.target sm_70`, `.address_size 64`. `sm_70` (Volta) is the oldest
// architecture with full `.sync` warp-intrinsic support (`shfl.sync`/`vote.sync` with an
// explicit member mask, as opposed to the legacy maskless forms PTX has since deprecated) while
// still being broadly deployed; PTX ISA 8.0 is a modern, widely-supported toolchain baseline
// that pairs with it. Every warp-collective instruction this backend emits uses the `.sync`
// form with a full-warp member mask (`0xffffffff`) — BIR carries no narrower mask information,
// so "every lane of the warp participates" is the only mask this backend can honestly claim.
//
// # Register model
//
// PTX is itself a virtual-register assembly language: `ptxas`, invoked inside the driver at
// module-load time, does the real register allocation. This backend does none of its own — it
// declares one virtual PTX register per BIR SSA value (and, for a function parameter, one per
// `ValRef::Param`), grouped into five typed pools matching PTX's own register-declaration
// convention (`.reg .pred %p<N>;`, `.reg .b32 %r<N>;`, `.reg .b64 %rd<N>;`, `.reg .f32 %f<N>;`,
// `.reg .f64 %fd<N>;`), and numbers each SSA value's register within its own pool in the order
// `RegAlloc::build` visits BIR's own append-only arenas (params first, then `Function::insts` in
// `InstId` order) — never a `HashMap`'s iteration order, so the same module always yields
// byte-identical text. A handful of fixed scratch registers (`%rs0`-`%rs2`, `%rds0`-`%rds1`,
// `%fs0`-`%fs1`, `%fds0`-`%fds1`, `%ps0`) are declared under names that can never collide with a
// counted pool's `%r<N>`-style range, used transiently within a single instruction's own
// lowering and never expected to stay live across one.
//
// `i1` gets its own pool (`.pred`) rather than sharing `.b32`: `icmp`/`fcmp` write their result
// straight into a `%p<N>`, and `condbr`/`select`'s condition operand is read straight out of one
// — no synthesis in either direction. `i8`/`i16` share the `.b32` pool with `i32` (PTX has no
// distinct 8/16-bit general-purpose register file — this is the same convention every real PTX
// toolchain uses): arithmetic that only depends on a result's low bits (`add`/`sub`/`mul`/`and`/
// `or`/`xor`/`shl`) runs directly on the raw 32-bit register, upper bits treated as don't-care
// exactly the way `basalt-x86`'s oracle treats its own slots' unused width; anything that reads
// a value's true sign or magnitude (`icmp`, `div`, `rem`, `ashr`, `lshr`, `sitofp`, `uitofp`,
// `switch`) first canonicalizes via `cvt.{s,u}32.{s,u}{8,16}` into a scratch register. `st`
// truncates its source register to the declared width automatically, so a narrow store never
// needs canonicalizing. This "operate at declared width, extend on demand, never assume a
// stored extension invariant" discipline is the direct PTX analogue of the exact-width
// memory-access discipline `basalt-x86/src/oracle.rs`'s header documents for its own slots.
//
// # Phi resolution
//
// Every predecessor block copies its own incoming value straight into a phi's own register at
// the end of the edge (`emit_phi_copies`), exactly like every prior backend in this tree — but
// with none of `basalt-x86/src/regalloc.rs`'s staging dance. That backend needs a two-phase
// stage-then-place sequence because a *physical* register or stack slot can house different SSA
// values at different points in the function, so writing two phis' destinations naively can
// clobber a value a later phi in the same copy list still needs to read. Here, every SSA value
// owns its PTX virtual register permanently and uniquely — no two distinct values ever share
// one — so that hazard cannot arise: an unconditional `mov` per incoming edge is always correct.
//
// # Signedness and other documented conventions
//
// `div`/`rem` lower as signed (`div.s32`/`rem.s64`/...), matching the convention
// `basalt-x86/src/oracle.rs` documents for the same reason: BIR's `Bin` carries no signed/
// unsigned distinction for these, so this backend picks the one interpretation the rest of the
// project already committed to. `atom.min`/`atom.max` follow the same signed convention for
// integer types. Pointers compare/atomic as unsigned 64-bit addresses uniformly, regardless of
// an `ICmpPred`'s own signed/unsigned flavor — address ordering has no sign to begin with.
//
// Floating-point `min`/`max` atomics have no native `atom.min.f32`/`atom.max.f32`/`.f64` form
// this backend is confident is available on every target this crate might run against, and
// there is no `ptxas` in this build environment to check against — rather than guess, they
// lower via the standard CAS-retry-loop technique (`lower_atomic_float_minmax`): read the
// current value, compute the candidate `min`/`max` in the native float type, `atom.cas` it in,
// and retry on a mismatch. `atom.add`/`.exch`/`.and`/`.or`/`.xor` are used directly (`add.f32`/
// `.f64` is unquestionably standard since Pascal; `exch`/`and`/`or`/`xor` operate on the raw bit
// pattern via `.b32`/`.b64`, well-defined regardless of what the bits represent). `f16` is
// refused outright (`E091`) for the identical reason: no `ptxas`/hardware access here to
// validate the exact native `.f16` instruction forms (arithmetic gated on `sm_53`+, no native
// `div.f16`, uncertain `setp`/`atom` support) — guessing at an unverifiable encoding is exactly
// the "silently-wrong codegen" this project refuses to ship; the CPU oracle draws an analogous
// line at `f16` for its own (different, F16C-shaped) reason.
//
// # Vector types
//
// A `Ty::Vec` `Load`/`Store` uses PTX's native `.v2`/`.v4` vector memory-access form for 2- or
// 4-lane vectors (matching real CUDA `float2`/`float4`-style accesses); any other lane count
// falls back to independent per-lane scalar accesses at consecutive byte offsets, which is
// always correct if not maximally efficient. Every other op BIR's type system allows to carry a
// `Ty::Vec` result (`Bin`, `Select`, `Cast`, `Phi` — the only shapes this backend has a
// consistent per-lane meaning for) decomposes into N independent scalar instructions, one per
// lane, even though no current frontend/lowering path emits one of these; proven here only via
// hand-built BIR in this file's tests, as no `.cu` kernel in this tree's test suite currently
// produces vector-typed BIR. `ConstInt`/`ConstFloat`, the atomics, the warp-collective ops, and
// the GPU index ops are refused on a `Ty::Vec` result (`E091`): BIR gives no defined
// lane-broadcast or lane-reduction semantics for any of them, and inventing one would be exactly
// the kind of guess this project's "no silently-wrong codegen" rule forbids. A vector-*operand*
// `icmp`/`fcmp` (as opposed to a vector-*result*) is refused for the same reason — BIR's own
// text grammar prints an `icmp`/`fcmp` instruction's result type as a bare `i1`, never a vector,
// so there is no defined way to fold a per-lane vector comparison down to one.
//
// # Synthetic local/param/shared/constant slot addresses
//
// `emit`/`supports` are handed the module post-`construct_ssa` (this backend runs the pass
// itself, exactly like `basalt-x86/src/regalloc.rs` — see that module's header for why a
// `Backend` is expected to be self-sufficient this way). Most of `basalt-sema/src/lower.rs`'s
// synthetic `const.i ptr.<space> (slot * SLOT_STRIDE)` addresses (see that crate's header) are
// promoted away by the time this backend sees them; whatever survives is treated, the same way
// every prior backend treats it, as an opaque `(space, raw offset)` identifier and backed by a
// real, named, function-scoped PTX variable declaration (`LocalSlots`) — `AddrSpace::Shared`
// gets a genuine `.shared` declaration (CTA-shared memory is real, distinct hardware state on a
// GPU, unlike the CPU oracle where it collapses into ordinary stack memory); `Local` and the
// rare/expected-empty `Param`/`Constant` fallbacks all get a `.local` declaration, `.local`
// being the closest real PTX state space to "this thread's own private, non-address-taken
// storage." Every such declaration is a plain 8-byte cell — this backend never learns a
// synthesized slot's true size (see the `basalt-sema` header on `SLOT_STRIDE`), so it backs
// every one uniformly, exactly like `basalt-x86/src/oracle.rs`'s own 8-byte-per-slot policy.
//
// A `Ret(Some(_))` reached inside a `.visible .entry` kernel drops the value and emits a plain
// `ret;`: a PTX kernel entry point has no way to hand a value back to the host at all, so
// carrying one through would have nowhere honest to go. Module-level `launch_bounds`/
// `shared_mem_bytes` metadata is intentionally left untouched here (dynamic shared-memory
// sizing and `.maxntid` launch-bounds directives are a real but separate concern) — the same
// scope line every backend in this tree already draws; see `basalt-x86/src/oracle.rs`'s own
// `ignores_launch_bounds_metadata` test for the established precedent.

use std::collections::HashMap;

use basalt_backend::{Artifact, ArtifactKind, Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId, Module, Op,
    Scalar, ShuffleKind, Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};
use basalt_passes::construct_ssa;

// ---- scratch registers -------------------------------------------------------------------
//
// Fixed, always-declared, never assigned to an SSA value; used transiently within one
// instruction's own lowering. Named with prefixes (`rs`/`rds`/`fs`/`fds`/`ps`) that can never
// collide with a counted pool's `%r<N>`/`%rd<N>`/`%f<N>`/`%fd<N>`/`%p<N>` range.
const SCRATCH0: &str = "%rs0";
const SCRATCH1: &str = "%rs1";
const SCRATCH2: &str = "%rs2";
const SCRATCH_D0: &str = "%rds0";
const SCRATCH_D1: &str = "%rds1";
const SCRATCH_F0: &str = "%fs0";
const SCRATCH_F1: &str = "%fs1";
const SCRATCH_FD0: &str = "%fds0";
const SCRATCH_FD1: &str = "%fds1";
const PRED_SCRATCH: &str = "%ps0";

// ---- register classes and per-value virtual registers ------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegClass {
    Pred = 0,
    B32 = 1,
    B64 = 2,
    F32 = 3,
    F64 = 4,
}

impl RegClass {
    fn bit_width(self) -> u32 {
        match self {
            RegClass::B32 | RegClass::F32 => 32,
            RegClass::B64 | RegClass::F64 => 64,
            RegClass::Pred => unreachable!("bit_width: check_module refuses predicate bitcasts"),
        }
    }
}

fn reg_type_word(class: RegClass) -> &'static str {
    match class {
        RegClass::Pred => "pred",
        RegClass::B32 => "b32",
        RegClass::B64 => "b64",
        RegClass::F32 => "f32",
        RegClass::F64 => "f64",
    }
}

fn reg_prefix(class: RegClass) -> &'static str {
    match class {
        RegClass::Pred => "p",
        RegClass::B32 => "r",
        RegClass::B64 => "rd",
        RegClass::F32 => "f",
        RegClass::F64 => "fd",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reg {
    class: RegClass,
    idx: u32,
}

impl Reg {
    fn text(self) -> String {
        format!("%{}{}", reg_prefix(self.class), self.idx)
    }
}

fn scalar_class(s: Scalar) -> Option<RegClass> {
    match s {
        Scalar::I1 => Some(RegClass::Pred),
        Scalar::I8 | Scalar::I16 | Scalar::I32 => Some(RegClass::B32),
        Scalar::I64 => Some(RegClass::B64),
        Scalar::F32 => Some(RegClass::F32),
        Scalar::F64 => Some(RegClass::F64),
        Scalar::F16 => None,
    }
}

fn reg_class_of(ty: Ty) -> RegClass {
    match ty {
        Ty::Scalar(s) => scalar_class(s).expect("f16 refused by check_module"),
        Ty::Ptr(_) => RegClass::B64,
        Ty::Vec(..) | Ty::Void => unreachable!("reg_class_of called on a non-scalar type"),
    }
}

fn scalar_of(ty: Ty) -> Scalar {
    match ty {
        Ty::Scalar(s) => s,
        _ => unreachable!("scalar_of called on a non-scalar type"),
    }
}

// ---- type-suffix tables --------------------------------------------------------------------

/// Unsigned/float memory-access suffix (`ld`/`st`, and the source-type half of a widening
/// `cvt`): a load never needs to pick sign vs. zero extension since every consumer that cares
/// canonicalizes on demand (see the module header).
fn mem_suffix(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::I1 | Scalar::I8 => "u8",
        Scalar::I16 => "u16",
        Scalar::I32 => "u32",
        Scalar::I64 => "u64",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
        Scalar::F16 => unreachable!("mem_suffix: f16 refused by check_module"),
    }
}

fn int_signed_suffix(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::I1 | Scalar::I8 => "s8",
        Scalar::I16 => "s16",
        Scalar::I32 => "s32",
        Scalar::I64 => "s64",
        _ => unreachable!("int_signed_suffix called on a non-integer type"),
    }
}

fn scalar_bytes(scalar: Scalar) -> u32 {
    match scalar {
        Scalar::I1 | Scalar::I8 => 1,
        Scalar::I16 => 2,
        Scalar::I32 | Scalar::F32 => 4,
        Scalar::I64 | Scalar::F64 => 8,
        Scalar::F16 => unreachable!("scalar_bytes: f16 refused by check_module"),
    }
}

fn space_word(space: AddrSpace) -> &'static str {
    match space {
        AddrSpace::Global => "global",
        AddrSpace::Shared => "shared",
        AddrSpace::Constant => "const",
        // `Param` should not survive `construct_ssa` in practice; see the module header.
        AddrSpace::Local | AddrSpace::Param => "local",
    }
}

fn local_like(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Shared | AddrSpace::Constant
    )
}

fn local_decl_space_word(space: AddrSpace) -> &'static str {
    match space {
        AddrSpace::Shared => "shared",
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Constant => "local",
        AddrSpace::Global => unreachable!("local_decl_space_word: local_like excludes Global"),
    }
}

fn space_tag(space: AddrSpace) -> u8 {
    match space {
        AddrSpace::Global => 0,
        AddrSpace::Shared => 1,
        AddrSpace::Constant => 2,
        AddrSpace::Local => 3,
        AddrSpace::Param => 4,
    }
}

fn icmp_cmp_suffix(pred: ICmpPred) -> &'static str {
    match pred {
        ICmpPred::Eq => "eq",
        ICmpPred::Ne => "ne",
        ICmpPred::Slt | ICmpPred::Ult => "lt",
        ICmpPred::Sle | ICmpPred::Ule => "le",
        ICmpPred::Sgt | ICmpPred::Ugt => "gt",
        ICmpPred::Sge | ICmpPred::Uge => "ge",
    }
}

fn icmp_signed(pred: ICmpPred) -> bool {
    matches!(
        pred,
        ICmpPred::Eq | ICmpPred::Ne | ICmpPred::Slt | ICmpPred::Sle | ICmpPred::Sgt | ICmpPred::Sge
    )
}

/// BIR's `FCmpPred` set (`Oeq/One/Olt/Ole/Ogt/Oge/Ord/Uno`) maps 1:1 onto PTX's ordered `setp`
/// suffixes plus `.num`/`.nan` — no unordered eq/lt/... variant is needed since BIR has none.
fn fcmp_suffix(pred: FCmpPred) -> &'static str {
    match pred {
        FCmpPred::Oeq => "eq",
        FCmpPred::One => "ne",
        FCmpPred::Olt => "lt",
        FCmpPred::Ole => "le",
        FCmpPred::Ogt => "gt",
        FCmpPred::Oge => "ge",
        FCmpPred::Ord => "num",
        FCmpPred::Uno => "nan",
    }
}

fn shuffle_mode(kind: ShuffleKind) -> &'static str {
    match kind {
        ShuffleKind::Idx => "idx",
        ShuffleKind::Up => "up",
        ShuffleKind::Down => "down",
        ShuffleKind::Xor => "bfly",
    }
}

/// The `c` (segment-mask/clamp) operand of `shfl.sync`: `0x1f` selects "whole warp, no
/// segmentation" for `idx`/`down`/`bfly`; `up`'s clamp conventionally names the lowest source
/// lane (`0`) rather than the highest, matching the documented direction of that shuffle.
fn shuffle_clamp(kind: ShuffleKind) -> &'static str {
    match kind {
        ShuffleKind::Up => "0x0",
        _ => "0x1f",
    }
}

// ---- refusal surface ------------------------------------------------------------------------

fn ty_has_f16(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F16) | Ty::Vec(Scalar::F16, _))
}

fn f16_refusal() -> Diag {
    Diag::new(ECode::UnsupportedType).with_arg(
        "f16: no ptxas-validated native instruction encoding available in this build environment",
    )
}

fn check_no_f16(inst: &Inst) -> Result<(), Diag> {
    if ty_has_f16(inst.ty) {
        return Err(f16_refusal());
    }
    match &inst.op {
        Op::Cast(_, sty, _) | Op::FCmp(_, sty, _, _) if ty_has_f16(*sty) => Err(f16_refusal()),
        Op::Store { ty, .. } if ty_has_f16(*ty) => Err(f16_refusal()),
        _ => Ok(()),
    }
}

/// The only op shapes this backend has a defined per-lane meaning for; see the module header.
fn vec_decomposable(op: &Op) -> bool {
    matches!(
        op,
        Op::Bin(..)
            | Op::Select(..)
            | Op::Cast(..)
            | Op::Phi(..)
            | Op::Load { .. }
            | Op::Store { .. }
    )
}

fn check_vec_decomposable(inst: &Inst) -> Result<(), Diag> {
    if let Ty::Vec(_, lanes) = inst.ty {
        if lanes == 0 {
            return Err(Diag::new(ECode::UnsupportedType).with_arg("zero-lane vector type"));
        }
        if !vec_decomposable(&inst.op) {
            return Err(Diag::new(ECode::UnsupportedType).with_arg(
                "vector-typed result has no defined per-lane decomposition for this op",
            ));
        }
    }
    Ok(())
}

fn check_no_vec_compare_operand(inst: &Inst) -> Result<(), Diag> {
    let oty = match &inst.op {
        Op::ICmp(_, oty, _, _) | Op::FCmp(_, oty, _, _) => *oty,
        _ => return Ok(()),
    };
    if matches!(oty, Ty::Vec(..)) {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("vector-operand compare has no scalar-reduction semantics in this backend"));
    }
    Ok(())
}

fn check_no_pred_bitcast(inst: &Inst) -> Result<(), Diag> {
    if let Op::Cast(CastOp::Bitcast, sty, _) = &inst.op {
        if matches!(sty, Ty::Scalar(Scalar::I1)) || matches!(inst.ty, Ty::Scalar(Scalar::I1)) {
            return Err(Diag::new(ECode::UnsupportedType)
                .with_arg("bitcast on a predicate-typed value has no defined bit pattern"));
        }
    }
    Ok(())
}

fn check_pred_bin_is_logical(inst: &Inst) -> Result<(), Diag> {
    if let Op::Bin(op, ..) = &inst.op {
        if matches!(inst.ty, Ty::Scalar(Scalar::I1))
            && !matches!(op, BinOp::And | BinOp::Or | BinOp::Xor)
        {
            return Err(Diag::new(ECode::UnsupportedOp)
                .with_arg("only and/or/xor are defined on a predicate-typed Bin"));
        }
    }
    Ok(())
}

fn check_atomic_width(inst: &Inst) -> Result<(), Diag> {
    if !matches!(&inst.op, Op::Atomic(..) | Op::AtomicCas(..)) {
        return Ok(());
    }
    if matches!(
        inst.ty,
        Ty::Scalar(Scalar::I1) | Ty::Scalar(Scalar::I8) | Ty::Scalar(Scalar::I16)
    ) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg(
            "sub-32-bit atomic RMW/CAS: no distinct 8/16-bit register class in this backend",
        ));
    }
    if matches!(inst.ty, Ty::Vec(..)) {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("vector-typed atomic RMW/CAS has no meaningful single-instruction form"));
    }
    Ok(())
}

fn check_no_pred_shuffle(inst: &Inst) -> Result<(), Diag> {
    if matches!(&inst.op, Op::Shuffle(..)) && matches!(inst.ty, Ty::Scalar(Scalar::I1)) {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("shuffling a predicate-typed value directly is not implemented"));
    }
    Ok(())
}

/// `mma` has no `mma.sync`/tensor-core lowering in this backend yet — that is separate, later
/// work (see the module header). Refuse cleanly rather than falling through to the scalar
/// per-op emitters below, which have no case for it.
fn check_no_mma(inst: &Inst) -> Result<(), Diag> {
    if matches!(&inst.op, Op::Mma { .. }) {
        return Err(Diag::new(ECode::UnsupportedOp)
            .with_arg("mma has no mma.sync lowering in this backend yet"));
    }
    Ok(())
}

/// Kernel-launch and CUDA Runtime API ops (`Op::KernelLaunch`/`Op::CudaMalloc`/
/// `Op::CudaMemcpy`/`Op::CudaFree`/`Op::CudaDeviceSynchronize`) are sema-only today — see
/// `Op::KernelLaunch`'s own doc comment in `basalt-bir/src/ir.rs`. A real host-side dispatch
/// story for this backend is separate, later work; refuse cleanly rather than falling through
/// to the scalar per-op emitters below, which have no case for any of them.
fn check_no_host_ops(inst: &Inst) -> Result<(), Diag> {
    if matches!(
        &inst.op,
        Op::KernelLaunch { .. }
            | Op::CudaMalloc { .. }
            | Op::CudaMemcpy { .. }
            | Op::CudaFree { .. }
            | Op::CudaDeviceSynchronize
    ) {
        return Err(Diag::new(ECode::UnsupportedOp).with_arg(
            "kernel launch / CUDA Runtime API calls have no lowering in this backend yet",
        ));
    }
    Ok(())
}

/// Single source of truth for what this backend refuses, shared verbatim by `supports()` and
/// `emit()`. Run once, on the module as originally handed in — `construct_ssa` never introduces
/// an op/type shape this pass didn't already see, so re-checking after it would be redundant.
///
/// Every function in the module is emitted as its own `.visible .entry` kernel (see
/// `emit_module`), so a non-kernel function (`is_kernel == false` — a plain/`__host__`/
/// `__device__` function) has no honest lowering here yet: nothing distinguishes a real
/// `__global__` kernel from a host-side helper function that happened to be lowered into the
/// same module. Refuse rather than silently emitting the latter as if it were launchable.
fn check_module(module: &Module) -> Result<(), Diag> {
    for f in &module.funcs {
        if !f.is_kernel {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("host/non-kernel function compilation is not yet implemented"));
        }
        if ty_has_f16(f.ret) {
            return Err(f16_refusal());
        }
        if f.params.iter().any(|&t| ty_has_f16(t)) {
            return Err(f16_refusal());
        }
        for inst in &f.insts {
            check_no_f16(inst)?;
            check_vec_decomposable(inst)?;
            check_no_vec_compare_operand(inst)?;
            check_no_pred_bitcast(inst)?;
            check_pred_bin_is_logical(inst)?;
            check_atomic_width(inst)?;
            check_no_pred_shuffle(inst)?;
            check_no_mma(inst)?;
            check_no_host_ops(inst)?;
        }
    }
    Ok(())
}

// ---- register allocation (naming, not allocation: see the module header) -------------------

#[derive(Clone)]
struct RegAlloc {
    param_regs: Vec<Vec<Reg>>,
    inst_regs: Vec<Vec<Reg>>,
    counts: [u32; 5],
}

fn alloc_reg(class: RegClass, counts: &mut [u32; 5]) -> Reg {
    let idx = counts[class as usize];
    counts[class as usize] += 1;
    Reg { class, idx }
}

fn alloc_for_ty(ty: Ty, counts: &mut [u32; 5]) -> Vec<Reg> {
    match ty {
        Ty::Void => vec![],
        Ty::Scalar(s) => vec![alloc_reg(
            scalar_class(s).expect("f16 refused by check_module"),
            counts,
        )],
        Ty::Ptr(_) => vec![alloc_reg(RegClass::B64, counts)],
        Ty::Vec(s, n) => {
            let class = scalar_class(s).expect("f16 refused by check_module");
            (0..n).map(|_| alloc_reg(class, counts)).collect()
        }
    }
}

impl RegAlloc {
    /// Params first (in declaration order), then every instruction in `InstId` order — BIR's
    /// own append-only construction order, never a `HashMap`'s iteration order (determinism).
    fn build(f: &Function) -> RegAlloc {
        let mut counts = [0u32; 5];
        let param_regs = f
            .params
            .iter()
            .map(|&ty| alloc_for_ty(ty, &mut counts))
            .collect();
        let inst_regs = f
            .insts
            .iter()
            .map(|inst| alloc_for_ty(inst.ty, &mut counts))
            .collect();
        RegAlloc {
            param_regs,
            inst_regs,
            counts,
        }
    }

    fn val(&self, v: ValRef) -> &[Reg] {
        match v {
            ValRef::Param(i) => &self.param_regs[i as usize],
            ValRef::Val(id) => &self.inst_regs[id.0 as usize],
        }
    }

    fn counted_classes(&self) -> Vec<(RegClass, u32)> {
        [
            RegClass::Pred,
            RegClass::B32,
            RegClass::B64,
            RegClass::F32,
            RegClass::F64,
        ]
        .into_iter()
        .filter_map(|c| {
            let n = self.counts[c as usize];
            if n > 0 {
                Some((c, n))
            } else {
                None
            }
        })
        .collect()
    }
}

// ---- synthetic local/param/shared/constant slot addresses ----------------------------------

struct LocalSlots {
    /// Insertion order = first-seen order over `Function::insts` (deterministic); `lookup`
    /// exists purely for O(1) codegen-time resolution, never iterated for output.
    order: Vec<(AddrSpace, i64, String)>,
    lookup: HashMap<(u8, i64), usize>,
}

impl LocalSlots {
    fn build(f: &Function) -> LocalSlots {
        let mut order = Vec::new();
        let mut lookup: HashMap<(u8, i64), usize> = HashMap::new();
        for inst in &f.insts {
            if let (Op::ConstInt(n), Ty::Ptr(space)) = (&inst.op, inst.ty) {
                if local_like(space) {
                    let key = (space_tag(space), *n);
                    lookup.entry(key).or_insert_with(|| {
                        let idx = order.len();
                        order.push((space, *n, format!("__local{idx}")));
                        idx
                    });
                }
            }
        }
        LocalSlots { order, lookup }
    }

    fn symbol(&self, space: AddrSpace, n: i64) -> &str {
        let idx = self.lookup[&(space_tag(space), n)];
        &self.order[idx].2
    }
}

// ---- phi resolution --------------------------------------------------------------------------

/// `(from_block, to_block) -> [(phi's own InstId, incoming value)]`. See the module header for
/// why this needs no staging, unlike every prior backend's version of this table.
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

// ---- parameter declarations ------------------------------------------------------------------

fn param_symbol(fname: &str, i: usize) -> String {
    format!("{fname}_param_{i}")
}

fn param_scalar_word(s: Scalar) -> &'static str {
    match s {
        Scalar::I1 | Scalar::I8 => "u8",
        Scalar::I16 => "u16",
        Scalar::I32 => "u32",
        Scalar::I64 => "u64",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
        Scalar::F16 => unreachable!("param_scalar_word: f16 refused by check_module"),
    }
}

/// Every param, scalar or vector, declares a plainly byte-addressed `.param` slot; a vector
/// param always uses the generic byte-array form (`.align <elem> .b8 name[bytes]`) rather than
/// PTX's `.v2`/`.v4` *load* syntax, since that native form only exists for `ld`/`st`, not for a
/// `.param` declaration itself — see `emit_param_loads` for how each lane is then read back out.
fn param_decl(fname: &str, i: usize, ty: Ty) -> String {
    let name = param_symbol(fname, i);
    match ty {
        Ty::Scalar(s) => format!(".param .{} {name}", param_scalar_word(s)),
        Ty::Ptr(_) => format!(".param .u64 {name}"),
        Ty::Vec(s, lanes) => {
            let bytes = scalar_bytes(s);
            let size = bytes * lanes as u32;
            format!(".param .align {bytes} .b8 {name}[{size}]")
        }
        Ty::Void => unreachable!("param_decl: a function parameter is never void"),
    }
}

// ---- code generation ---------------------------------------------------------------------

struct CodeGen<'a> {
    f: &'a Function,
    alloc: RegAlloc,
    locals: LocalSlots,
    phi_copies: PhiCopies,
    out: String,
    label_counter: u32,
}

impl<'a> CodeGen<'a> {
    fn line(&mut self, text: &str) {
        self.out.push('\t');
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn label(&mut self, text: &str) {
        self.out.push_str(text);
        self.out.push_str(":\n");
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("${prefix}_{}", self.label_counter)
    }

    fn dst_reg(&self, id: InstId) -> Reg {
        self.alloc.inst_regs[id.0 as usize][0]
    }

    fn dst_regs(&self, id: InstId) -> Vec<Reg> {
        self.alloc.inst_regs[id.0 as usize].clone()
    }

    fn val_reg(&self, v: ValRef) -> Reg {
        self.alloc.val(v)[0]
    }

    fn val_regs(&self, v: ValRef) -> &[Reg] {
        self.alloc.val(v)
    }

    fn valref_ty(&self, v: ValRef) -> Ty {
        match v {
            ValRef::Param(i) => self.f.params[i as usize],
            ValRef::Val(id) => self.f.insts[id.0 as usize].ty,
        }
    }

    fn emit_param_loads(&mut self) {
        for i in 0..self.f.params.len() {
            let ty = self.f.params[i];
            let sym = param_symbol(&self.f.name, i);
            let regs = self.alloc.param_regs[i].clone();
            match ty {
                Ty::Scalar(Scalar::I1) => {
                    self.line(&format!("ld.param.u8 {SCRATCH0}, [{sym}];"));
                    self.line(&format!("setp.ne.s32 {}, {SCRATCH0}, 0;", regs[0].text()));
                }
                Ty::Scalar(s) => {
                    self.line(&format!(
                        "ld.param.{} {}, [{sym}];",
                        mem_suffix(s),
                        regs[0].text()
                    ));
                }
                Ty::Ptr(_) => {
                    self.line(&format!("ld.param.u64 {}, [{sym}];", regs[0].text()));
                }
                Ty::Vec(s, _) => {
                    let bytes = scalar_bytes(s);
                    for (i2, r) in regs.iter().enumerate() {
                        self.line(&format!(
                            "ld.param.{} {}, [{sym}+{}];",
                            mem_suffix(s),
                            r.text(),
                            i2 as u32 * bytes
                        ));
                    }
                }
                Ty::Void => unreachable!("emit_param_loads: a function parameter is never void"),
            }
        }
    }

    // ---- dispatch ---------------------------------------------------------------------

    fn lower_inst(&mut self, id: InstId) {
        let f = self.f;
        let inst = &f.insts[id.0 as usize];
        let ty = inst.ty;
        match &inst.op {
            Op::ConstInt(n) => {
                let n = *n;
                self.lower_const_int(id, n, ty);
            }
            Op::ConstFloat(v) => {
                let v = *v;
                self.lower_const_float(id, v, ty);
            }
            Op::Bin(op, a, b) => {
                let (op, a, b) = (*op, *a, *b);
                self.lower_bin(id, op, a, b, ty);
            }
            Op::ICmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                self.lower_icmp(id, pred, cty, a, b);
            }
            Op::FCmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                self.lower_fcmp(id, pred, cty, a, b);
            }
            Op::Select(c, a, b) => {
                let (c, a, b) = (*c, *a, *b);
                self.lower_select(id, c, a, b, ty);
            }
            Op::Cast(cop, sty, v) => {
                let (cop, sty, v) = (*cop, *sty, *v);
                self.lower_cast(id, cop, sty, v, ty);
            }
            Op::Load { ptr, space, .. } => {
                let (ptr, space) = (*ptr, *space);
                self.lower_load(id, ptr, space, ty);
            }
            Op::Store {
                ptr,
                val,
                ty: sty,
                space,
                ..
            } => {
                let (ptr, val, sty, space) = (*ptr, *val, *sty, *space);
                self.lower_store(ptr, val, space, sty);
            }
            // Every predecessor writes this phi's own register before jumping here — see
            // `emit_phi_copies`.
            Op::Phi(_) => {}
            Op::TidX => self.lower_index(id, "%tid.x"),
            Op::TidY => self.lower_index(id, "%tid.y"),
            Op::TidZ => self.lower_index(id, "%tid.z"),
            Op::BidX => self.lower_index(id, "%ctaid.x"),
            Op::BidY => self.lower_index(id, "%ctaid.y"),
            Op::BidZ => self.lower_index(id, "%ctaid.z"),
            Op::BdimX => self.lower_index(id, "%ntid.x"),
            Op::BdimY => self.lower_index(id, "%ntid.y"),
            Op::BdimZ => self.lower_index(id, "%ntid.z"),
            Op::GdimX => self.lower_index(id, "%nctaid.x"),
            Op::GdimY => self.lower_index(id, "%nctaid.y"),
            Op::GdimZ => self.lower_index(id, "%nctaid.z"),
            // Real hardware CTA-wide synchronization: unlike the CPU oracle's `nop` (correct
            // there only because threads run strictly one at a time), PTX threads genuinely
            // run concurrently, so this barrier is load-bearing.
            Op::Barrier => self.line("bar.sync 0;"),
            Op::Shuffle(kind, val, amt) => {
                let (kind, val, amt) = (*kind, *val, *amt);
                self.lower_shuffle(id, kind, val, amt, ty);
            }
            Op::Ballot(v) => {
                let v = *v;
                self.lower_ballot(id, v);
            }
            Op::VoteAny(v) => {
                let v = *v;
                self.lower_vote(id, v, "any");
            }
            Op::VoteAll(v) => {
                let v = *v;
                self.lower_vote(id, v, "all");
            }
            Op::Atomic(op, ptr, val, space) => {
                let (op, ptr, val, space) = (*op, *ptr, *val, *space);
                self.lower_atomic(id, op, ptr, val, space, ty);
            }
            Op::AtomicCas(ptr, cmp, newv, space) => {
                let (ptr, cmp, newv, space) = (*ptr, *cmp, *newv, *space);
                self.lower_atomic_cas(id, ptr, cmp, newv, space, ty);
            }
            Op::Mma { .. } => {
                unreachable!("check_module refuses mma before codegen starts")
            }
            Op::KernelLaunch { .. }
            | Op::CudaMalloc { .. }
            | Op::CudaMemcpy { .. }
            | Op::CudaFree { .. }
            | Op::CudaDeviceSynchronize => {
                unreachable!("check_module refuses kernel launch / CUDA Runtime API ops before codegen starts")
            }
        }
    }

    // ---- constants ----------------------------------------------------------------------

    fn lower_const_int(&mut self, id: InstId, n: i64, ty: Ty) {
        if let Ty::Ptr(space) = ty {
            let dst = self.dst_reg(id);
            if local_like(space) {
                let sym = self.locals.symbol(space, n).to_string();
                self.line(&format!("mov.u64 {}, {sym};", dst.text()));
            } else {
                self.line(&format!("mov.u64 {}, {n};", dst.text()));
            }
            return;
        }
        let dst = self.dst_reg(id);
        match ty {
            Ty::Scalar(Scalar::I1) => {
                let v = if n != 0 { 1 } else { 0 };
                self.line(&format!("mov.u32 {SCRATCH0}, {v};"));
                self.line(&format!("setp.ne.s32 {}, {SCRATCH0}, 0;", dst.text()));
            }
            Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::I32) => {
                self.line(&format!("mov.u32 {}, {};", dst.text(), n as i32));
            }
            Ty::Scalar(Scalar::I64) => {
                self.line(&format!("mov.u64 {}, {n};", dst.text()));
            }
            _ => unreachable!("check_module refused f16/vec/float ConstInt"),
        }
    }

    fn lower_const_float(&mut self, id: InstId, v: f64, ty: Ty) {
        let dst = self.dst_reg(id);
        match ty {
            Ty::Scalar(Scalar::F32) => {
                let bits = (v as f32).to_bits();
                self.line(&format!("mov.f32 {}, 0f{bits:08X};", dst.text()));
            }
            Ty::Scalar(Scalar::F64) => {
                let bits = v.to_bits();
                self.line(&format!("mov.f64 {}, 0d{bits:016X};", dst.text()));
            }
            _ => unreachable!("check_module refused f16/vec/int ConstFloat"),
        }
    }

    // ---- narrow-int canonicalization ------------------------------------------------------

    /// Extends an `i8`/`i16` value from its true declared width into a scratch `.b32` register
    /// (signed or unsigned per `signed`); returns that value's own register text unchanged for
    /// `i32`/`i64`, which are always already exact. See the module header.
    fn canon_int_reg(
        &mut self,
        r: Reg,
        scalar: Scalar,
        signed: bool,
        scratch: &'static str,
    ) -> String {
        match scalar {
            Scalar::I8 | Scalar::I16 => {
                let ssuf = if signed {
                    int_signed_suffix(scalar)
                } else {
                    mem_suffix(scalar)
                };
                let dsuf = if signed { "s32" } else { "u32" };
                self.line(&format!("cvt.{dsuf}.{ssuf} {scratch}, {};", r.text()));
                scratch.to_string()
            }
            _ => r.text(),
        }
    }

    // ---- Bin --------------------------------------------------------------------------

    fn lower_bin(&mut self, id: InstId, op: BinOp, a: ValRef, b: ValRef, ty: Ty) {
        if let Ty::Vec(scalar, lanes) = ty {
            let dst_regs = self.dst_regs(id);
            let a_regs = self.val_regs(a).to_vec();
            let b_regs = self.val_regs(b).to_vec();
            for i in 0..lanes as usize {
                self.lower_bin_reg(dst_regs[i], op, a_regs[i], b_regs[i], Ty::Scalar(scalar));
            }
            return;
        }
        let dst = self.dst_reg(id);
        let ra = self.val_reg(a);
        let rb = self.val_reg(b);
        self.lower_bin_reg(dst, op, ra, rb, ty);
    }

    fn lower_bin_reg(&mut self, dst: Reg, op: BinOp, a: Reg, b: Reg, ty: Ty) {
        match ty {
            Ty::Scalar(Scalar::F32) => self.lower_fbin_reg(dst, op, a, b, false),
            Ty::Scalar(Scalar::F64) => self.lower_fbin_reg(dst, op, a, b, true),
            Ty::Scalar(Scalar::I1) => self.lower_bin_pred_reg(dst, op, a, b),
            Ty::Scalar(s @ (Scalar::I8 | Scalar::I16 | Scalar::I32)) => {
                self.lower_bin_int32_reg(dst, op, a, b, s)
            }
            // `i64`-typed pointer arithmetic (`basalt-sema/src/lower.rs`'s `lower_ptr_offset`)
            // reaches here too: a pointer operand and an `i64` offset operand are both already
            // `.b64`-class registers, needing no per-operand type dispatch to add correctly.
            Ty::Scalar(Scalar::I64) | Ty::Ptr(_) => self.lower_bin_int64_reg(dst, op, a, b),
            _ => unreachable!("check_module refused f16 Bin"),
        }
    }

    fn lower_bin_int32_reg(&mut self, dst: Reg, op: BinOp, a: Reg, b: Reg, scalar: Scalar) {
        let ra = a.text();
        let rb = b.text();
        match op {
            BinOp::Add => self.line(&format!("add.s32 {}, {ra}, {rb};", dst.text())),
            BinOp::Sub => self.line(&format!("sub.s32 {}, {ra}, {rb};", dst.text())),
            BinOp::Mul => self.line(&format!("mul.lo.s32 {}, {ra}, {rb};", dst.text())),
            BinOp::And => self.line(&format!("and.b32 {}, {ra}, {rb};", dst.text())),
            BinOp::Or => self.line(&format!("or.b32 {}, {ra}, {rb};", dst.text())),
            BinOp::Xor => self.line(&format!("xor.b32 {}, {ra}, {rb};", dst.text())),
            BinOp::Shl => self.line(&format!("shl.b32 {}, {ra}, {rb};", dst.text())),
            BinOp::Div => {
                let ea = self.canon_int_reg(a, scalar, true, SCRATCH0);
                let eb = self.canon_int_reg(b, scalar, true, SCRATCH1);
                self.line(&format!("div.s32 {}, {ea}, {eb};", dst.text()));
            }
            BinOp::Rem => {
                let ea = self.canon_int_reg(a, scalar, true, SCRATCH0);
                let eb = self.canon_int_reg(b, scalar, true, SCRATCH1);
                self.line(&format!("rem.s32 {}, {ea}, {eb};", dst.text()));
            }
            BinOp::Ashr => {
                let ea = self.canon_int_reg(a, scalar, true, SCRATCH0);
                self.line(&format!("shr.s32 {}, {ea}, {rb};", dst.text()));
            }
            BinOp::Lshr => {
                let ea = self.canon_int_reg(a, scalar, false, SCRATCH0);
                self.line(&format!("shr.u32 {}, {ea}, {rb};", dst.text()));
            }
            _ => unreachable!("float BinOp on integer Bin"),
        }
    }

    fn lower_bin_int64_reg(&mut self, dst: Reg, op: BinOp, a: Reg, b: Reg) {
        let ra = a.text();
        let rb = b.text();
        match op {
            BinOp::Add => self.line(&format!("add.s64 {}, {ra}, {rb};", dst.text())),
            BinOp::Sub => self.line(&format!("sub.s64 {}, {ra}, {rb};", dst.text())),
            BinOp::Mul => self.line(&format!("mul.lo.s64 {}, {ra}, {rb};", dst.text())),
            BinOp::And => self.line(&format!("and.b64 {}, {ra}, {rb};", dst.text())),
            BinOp::Or => self.line(&format!("or.b64 {}, {ra}, {rb};", dst.text())),
            BinOp::Xor => self.line(&format!("xor.b64 {}, {ra}, {rb};", dst.text())),
            // The shift-count operand of a 64-bit shift is always `.u32`, unlike the shifted
            // value itself; narrow it down from its `.b64` register first.
            BinOp::Shl => {
                self.line(&format!("cvt.u32.u64 {SCRATCH0}, {rb};"));
                self.line(&format!("shl.b64 {}, {ra}, {SCRATCH0};", dst.text()));
            }
            BinOp::Ashr => {
                self.line(&format!("cvt.u32.u64 {SCRATCH0}, {rb};"));
                self.line(&format!("shr.s64 {}, {ra}, {SCRATCH0};", dst.text()));
            }
            BinOp::Lshr => {
                self.line(&format!("cvt.u32.u64 {SCRATCH0}, {rb};"));
                self.line(&format!("shr.u64 {}, {ra}, {SCRATCH0};", dst.text()));
            }
            BinOp::Div => self.line(&format!("div.s64 {}, {ra}, {rb};", dst.text())),
            BinOp::Rem => self.line(&format!("rem.s64 {}, {ra}, {rb};", dst.text())),
            _ => unreachable!("float BinOp on integer Bin"),
        }
    }

    fn lower_bin_pred_reg(&mut self, dst: Reg, op: BinOp, a: Reg, b: Reg) {
        let mnem = match op {
            BinOp::And => "and.pred",
            BinOp::Or => "or.pred",
            BinOp::Xor => "xor.pred",
            _ => unreachable!("check_module allows only and/or/xor on predicate-typed Bin"),
        };
        self.line(&format!(
            "{mnem} {}, {}, {};",
            dst.text(),
            a.text(),
            b.text()
        ));
    }

    fn lower_fbin_reg(&mut self, dst: Reg, op: BinOp, a: Reg, b: Reg, f64_: bool) {
        let ra = a.text();
        let rb = b.text();
        let t = if f64_ { "f64" } else { "f32" };
        match op {
            BinOp::FAdd => self.line(&format!("add.{t} {}, {ra}, {rb};", dst.text())),
            BinOp::FSub => self.line(&format!("sub.{t} {}, {ra}, {rb};", dst.text())),
            BinOp::FMul => self.line(&format!("mul.{t} {}, {ra}, {rb};", dst.text())),
            BinOp::FDiv => self.line(&format!("div.rn.{t} {}, {ra}, {rb};", dst.text())),
            BinOp::FRem => self.lower_frem(dst, &ra, &rb, f64_),
            _ => unreachable!("integer BinOp on float Bin"),
        }
    }

    /// PTX has no native `frem`; software emulation identical in shape to
    /// `basalt-x86/src/oracle.rs`'s own `lower_frem`: `q = trunc(a/b); result = a - q*b`.
    fn lower_frem(&mut self, dst: Reg, ra: &str, rb: &str, f64_: bool) {
        if f64_ {
            self.line(&format!("div.rn.f64 {SCRATCH_FD0}, {ra}, {rb};"));
            self.line(&format!("cvt.rzi.s64.f64 {SCRATCH_D0}, {SCRATCH_FD0};"));
            self.line(&format!("cvt.rn.f64.s64 {SCRATCH_FD0}, {SCRATCH_D0};"));
            self.line(&format!("mul.f64 {SCRATCH_FD0}, {SCRATCH_FD0}, {rb};"));
            self.line(&format!("sub.f64 {}, {ra}, {SCRATCH_FD0};", dst.text()));
        } else {
            self.line(&format!("div.rn.f32 {SCRATCH_F0}, {ra}, {rb};"));
            self.line(&format!("cvt.rzi.s32.f32 {SCRATCH2}, {SCRATCH_F0};"));
            self.line(&format!("cvt.rn.f32.s32 {SCRATCH_F0}, {SCRATCH2};"));
            self.line(&format!("mul.f32 {SCRATCH_F0}, {SCRATCH_F0}, {rb};"));
            self.line(&format!("sub.f32 {}, {ra}, {SCRATCH_F0};", dst.text()));
        }
    }

    // ---- ICmp / FCmp ------------------------------------------------------------------

    fn lower_icmp(&mut self, id: InstId, pred: ICmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let dst = self.dst_reg(id);
        let signed = icmp_signed(pred);
        let ra_reg = self.val_reg(a);
        let rb_reg = self.val_reg(b);
        let (ra, rb, tsuf) = match cty {
            Ty::Scalar(Scalar::I1) => {
                self.line(&format!("selp.u32 {SCRATCH0}, 1, 0, {};", ra_reg.text()));
                self.line(&format!("selp.u32 {SCRATCH1}, 1, 0, {};", rb_reg.text()));
                (
                    SCRATCH0.to_string(),
                    SCRATCH1.to_string(),
                    if signed { "s32" } else { "u32" },
                )
            }
            Ty::Scalar(s @ (Scalar::I8 | Scalar::I16)) => {
                let ra = self.canon_int_reg(ra_reg, s, signed, SCRATCH0);
                let rb = self.canon_int_reg(rb_reg, s, signed, SCRATCH1);
                (ra, rb, if signed { "s32" } else { "u32" })
            }
            Ty::Scalar(Scalar::I32) => (
                ra_reg.text(),
                rb_reg.text(),
                if signed { "s32" } else { "u32" },
            ),
            Ty::Scalar(Scalar::I64) => (
                ra_reg.text(),
                rb_reg.text(),
                if signed { "s64" } else { "u64" },
            ),
            Ty::Ptr(_) => (ra_reg.text(), rb_reg.text(), "u64"),
            _ => unreachable!("check_module refused vector/f16 icmp operands"),
        };
        self.line(&format!(
            "setp.{}.{} {}, {}, {};",
            icmp_cmp_suffix(pred),
            tsuf,
            dst.text(),
            ra,
            rb
        ));
    }

    fn lower_fcmp(&mut self, id: InstId, pred: FCmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let dst = self.dst_reg(id);
        let f = if matches!(cty, Ty::Scalar(Scalar::F64)) {
            "f64"
        } else {
            "f32"
        };
        let ra = self.val_reg(a).text();
        let rb = self.val_reg(b).text();
        self.line(&format!(
            "setp.{}.{f} {}, {ra}, {rb};",
            fcmp_suffix(pred),
            dst.text()
        ));
    }

    // ---- Select -------------------------------------------------------------------------

    fn lower_select(&mut self, id: InstId, c: ValRef, a: ValRef, b: ValRef, ty: Ty) {
        let cond = self.val_reg(c);
        if let Ty::Vec(scalar, lanes) = ty {
            let dst_regs = self.dst_regs(id);
            let a_regs = self.val_regs(a).to_vec();
            let b_regs = self.val_regs(b).to_vec();
            for i in 0..lanes as usize {
                self.lower_select_reg(dst_regs[i], cond, a_regs[i], b_regs[i], Ty::Scalar(scalar));
            }
            return;
        }
        let dst = self.dst_reg(id);
        let ra = self.val_reg(a);
        let rb = self.val_reg(b);
        self.lower_select_reg(dst, cond, ra, rb, ty);
    }

    fn lower_select_reg(&mut self, dst: Reg, cond: Reg, a: Reg, b: Reg, ty: Ty) {
        let p = cond.text();
        match ty {
            Ty::Scalar(Scalar::I1) => {
                self.line(&format!("selp.u32 {SCRATCH0}, 1, 0, {};", a.text()));
                self.line(&format!("selp.u32 {SCRATCH1}, 1, 0, {};", b.text()));
                self.line(&format!(
                    "selp.b32 {SCRATCH2}, {SCRATCH0}, {SCRATCH1}, {p};"
                ));
                self.line(&format!("setp.ne.s32 {}, {SCRATCH2}, 0;", dst.text()));
            }
            Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::I32) => self.line(&format!(
                "selp.b32 {}, {}, {}, {p};",
                dst.text(),
                a.text(),
                b.text()
            )),
            Ty::Scalar(Scalar::I64) | Ty::Ptr(_) => self.line(&format!(
                "selp.b64 {}, {}, {}, {p};",
                dst.text(),
                a.text(),
                b.text()
            )),
            Ty::Scalar(Scalar::F32) => self.line(&format!(
                "selp.f32 {}, {}, {}, {p};",
                dst.text(),
                a.text(),
                b.text()
            )),
            Ty::Scalar(Scalar::F64) => self.line(&format!(
                "selp.f64 {}, {}, {}, {p};",
                dst.text(),
                a.text(),
                b.text()
            )),
            _ => unreachable!("check_module refused f16/vec Select arms"),
        }
    }

    // ---- Cast -------------------------------------------------------------------------

    fn lower_cast(&mut self, id: InstId, cop: CastOp, sty: Ty, v: ValRef, dty: Ty) {
        if let Ty::Vec(dscalar, lanes) = dty {
            let sscalar = match sty {
                Ty::Vec(s, _) => s,
                _ => unreachable!("check_module keeps cast lane counts matched"),
            };
            let dst_regs = self.dst_regs(id);
            let src_regs = self.val_regs(v).to_vec();
            for i in 0..lanes as usize {
                self.lower_cast_scalar(
                    dst_regs[i],
                    cop,
                    Ty::Scalar(sscalar),
                    src_regs[i],
                    Ty::Scalar(dscalar),
                );
            }
            return;
        }
        let dst = self.dst_reg(id);
        let src = self.val_reg(v);
        self.lower_cast_scalar(dst, cop, sty, src, dty);
    }

    fn lower_cast_scalar(&mut self, dst: Reg, cop: CastOp, sty: Ty, src: Reg, dty: Ty) {
        match cop {
            CastOp::Trunc => self.lower_trunc(dst, sty, src, dty),
            CastOp::Zext => self.lower_zext(dst, sty, src, dty),
            CastOp::Sext => self.lower_sext(dst, sty, src, dty),
            CastOp::FpTrunc => {
                self.line(&format!("cvt.rn.f32.f64 {}, {};", dst.text(), src.text()))
            }
            CastOp::FpExt => self.line(&format!("cvt.f64.f32 {}, {};", dst.text(), src.text())),
            CastOp::FpToSi => self.lower_fp_to_int(dst, sty, src, dty, true),
            CastOp::FpToUi => self.lower_fp_to_int(dst, sty, src, dty, false),
            CastOp::SiToFp => self.lower_int_to_fp(dst, sty, src, dty, true),
            CastOp::UiToFp => self.lower_int_to_fp(dst, sty, src, dty, false),
            CastOp::Bitcast => self.lower_bitcast(dst, sty, src),
        }
    }

    fn lower_trunc(&mut self, dst: Reg, sty: Ty, src: Reg, dty: Ty) {
        if matches!(dty, Ty::Scalar(Scalar::I1)) {
            self.line(&format!("and.b32 {SCRATCH0}, {}, 1;", src.text()));
            self.line(&format!("setp.ne.s32 {}, {SCRATCH0}, 0;", dst.text()));
            return;
        }
        if matches!(sty, Ty::Scalar(Scalar::I64)) {
            self.line(&format!("cvt.u32.u64 {}, {};", dst.text(), src.text()));
            return;
        }
        // Both source and destination are `.b32`-class already (narrowing within i32/i16/i8):
        // the same physical register width, just a fresh SSA register number.
        self.line(&format!("mov.b32 {}, {};", dst.text(), src.text()));
    }

    fn lower_zext(&mut self, dst: Reg, sty: Ty, src: Reg, dty: Ty) {
        if matches!(sty, Ty::Scalar(Scalar::I1)) {
            if matches!(dty, Ty::Scalar(Scalar::I64)) {
                self.line(&format!("selp.u32 {SCRATCH0}, 1, 0, {};", src.text()));
                self.line(&format!("cvt.u64.u32 {}, {SCRATCH0};", dst.text()));
            } else {
                self.line(&format!("selp.u32 {}, 1, 0, {};", dst.text(), src.text()));
            }
            return;
        }
        let ssuf = mem_suffix(scalar_of(sty));
        if matches!(dty, Ty::Scalar(Scalar::I64)) {
            self.line(&format!("cvt.u64.{ssuf} {}, {};", dst.text(), src.text()));
        } else {
            self.line(&format!("cvt.u32.{ssuf} {}, {};", dst.text(), src.text()));
        }
    }

    fn lower_sext(&mut self, dst: Reg, sty: Ty, src: Reg, dty: Ty) {
        if matches!(sty, Ty::Scalar(Scalar::I1)) {
            if matches!(dty, Ty::Scalar(Scalar::I64)) {
                self.line(&format!("selp.s32 {SCRATCH0}, -1, 0, {};", src.text()));
                self.line(&format!("cvt.s64.s32 {}, {SCRATCH0};", dst.text()));
            } else {
                self.line(&format!("selp.s32 {}, -1, 0, {};", dst.text(), src.text()));
            }
            return;
        }
        let ssuf = int_signed_suffix(scalar_of(sty));
        if matches!(dty, Ty::Scalar(Scalar::I64)) {
            self.line(&format!("cvt.s64.{ssuf} {}, {};", dst.text(), src.text()));
        } else {
            self.line(&format!("cvt.s32.{ssuf} {}, {};", dst.text(), src.text()));
        }
    }

    fn lower_fp_to_int(&mut self, dst: Reg, sty: Ty, src: Reg, dty: Ty, signed: bool) {
        let f = if matches!(sty, Ty::Scalar(Scalar::F64)) {
            "f64"
        } else {
            "f32"
        };
        let dscalar = scalar_of(dty);
        let dsuf = if signed {
            int_signed_suffix(dscalar)
        } else {
            mem_suffix(dscalar)
        };
        self.line(&format!(
            "cvt.rzi.{dsuf}.{f} {}, {};",
            dst.text(),
            src.text()
        ));
    }

    fn lower_int_to_fp(&mut self, dst: Reg, sty: Ty, src: Reg, dty: Ty, signed: bool) {
        let f = if matches!(dty, Ty::Scalar(Scalar::F64)) {
            "f64"
        } else {
            "f32"
        };
        if matches!(sty, Ty::Scalar(Scalar::I1)) {
            if signed {
                self.line(&format!("selp.s32 {SCRATCH0}, -1, 0, {};", src.text()));
            } else {
                self.line(&format!("selp.u32 {SCRATCH0}, 1, 0, {};", src.text()));
            }
            self.line(&format!("cvt.rn.{f}.s32 {}, {SCRATCH0};", dst.text()));
            return;
        }
        let sscalar = scalar_of(sty);
        let ssuf = if signed {
            int_signed_suffix(sscalar)
        } else {
            mem_suffix(sscalar)
        };
        self.line(&format!(
            "cvt.rn.{f}.{ssuf} {}, {};",
            dst.text(),
            src.text()
        ));
    }

    fn lower_bitcast(&mut self, dst: Reg, sty: Ty, src: Reg) {
        let width = reg_class_of(sty).bit_width();
        let word = if width == 32 { "b32" } else { "b64" };
        self.line(&format!("mov.{word} {}, {};", dst.text(), src.text()));
    }

    // ---- Load / Store -------------------------------------------------------------------

    fn lower_load(&mut self, id: InstId, ptr: ValRef, space: AddrSpace, ty: Ty) {
        let addr = self.val_reg(ptr).text();
        let sw = space_word(space);
        if let Ty::Vec(scalar, lanes) = ty {
            let regs = self.dst_regs(id);
            let suffix = mem_suffix(scalar);
            if lanes == 2 || lanes == 4 {
                let list = regs.iter().map(|r| r.text()).collect::<Vec<_>>().join(", ");
                self.line(&format!("ld.{sw}.v{lanes}.{suffix} {{{list}}}, [{addr}];"));
            } else {
                let bytes = scalar_bytes(scalar);
                for (i, r) in regs.iter().enumerate() {
                    self.line(&format!(
                        "ld.{sw}.{suffix} {}, [{addr}+{}];",
                        r.text(),
                        i as u32 * bytes
                    ));
                }
            }
            return;
        }
        let dst = self.dst_reg(id);
        match ty {
            Ty::Scalar(Scalar::I1) => {
                self.line(&format!("ld.{sw}.u8 {SCRATCH0}, [{addr}];"));
                self.line(&format!("setp.ne.s32 {}, {SCRATCH0}, 0;", dst.text()));
            }
            Ty::Scalar(s) => self.line(&format!(
                "ld.{sw}.{} {}, [{addr}];",
                mem_suffix(s),
                dst.text()
            )),
            Ty::Ptr(_) => self.line(&format!("ld.{sw}.u64 {}, [{addr}];", dst.text())),
            _ => unreachable!("check_module refused f16 Load"),
        }
    }

    fn lower_store(&mut self, ptr: ValRef, val: ValRef, space: AddrSpace, ty: Ty) {
        let addr = self.val_reg(ptr).text();
        let sw = space_word(space);
        if let Ty::Vec(scalar, lanes) = ty {
            let regs = self.val_regs(val).to_vec();
            let suffix = mem_suffix(scalar);
            if lanes == 2 || lanes == 4 {
                let list = regs.iter().map(|r| r.text()).collect::<Vec<_>>().join(", ");
                self.line(&format!("st.{sw}.v{lanes}.{suffix} [{addr}], {{{list}}};"));
            } else {
                let bytes = scalar_bytes(scalar);
                for (i, r) in regs.iter().enumerate() {
                    self.line(&format!(
                        "st.{sw}.{suffix} [{addr}+{}], {};",
                        i as u32 * bytes,
                        r.text()
                    ));
                }
            }
            return;
        }
        match ty {
            Ty::Scalar(Scalar::I1) => {
                let v = self.val_reg(val).text();
                self.line(&format!("selp.u32 {SCRATCH0}, 1, 0, {v};"));
                self.line(&format!("st.{sw}.u8 [{addr}], {SCRATCH0};"));
            }
            Ty::Scalar(s) => {
                let v = self.val_reg(val).text();
                self.line(&format!("st.{sw}.{} [{addr}], {v};", mem_suffix(s)));
            }
            Ty::Ptr(_) => {
                let v = self.val_reg(val).text();
                self.line(&format!("st.{sw}.u64 [{addr}], {v};"));
            }
            _ => unreachable!("check_module refused f16 Store"),
        }
    }

    // ---- GPU index ops --------------------------------------------------------------------

    /// Every `.sreg` special register PTX exposes here (`%tid.x`, `%ctaid.x`, ...) is natively
    /// `.u32` — CUDA-C's own lowering (`basalt-sema`'s `lower.rs`) always gives these ops a
    /// 32-bit BIR type, matching that native width, but a frontend is free to ask for a wider
    /// result (Triton's own lowering uniformly types every index/arithmetic value `i64` — see
    /// `triton_lower.rs`'s module header). A 64-bit destination therefore reads the special
    /// register into a 32-bit scratch first and widens it, rather than assuming the caller's
    /// result width always matches the hardware register's own.
    fn lower_index(&mut self, id: InstId, special: &str) {
        let dst = self.dst_reg(id);
        match dst.class {
            RegClass::B32 => self.line(&format!("mov.u32 {}, {special};", dst.text())),
            RegClass::B64 => {
                self.line(&format!("mov.u32 {SCRATCH0}, {special};"));
                self.line(&format!("cvt.u64.u32 {}, {SCRATCH0};", dst.text()));
            }
            RegClass::Pred | RegClass::F32 | RegClass::F64 => {
                unreachable!("a GPU index op's result is always an unsigned integer type")
            }
        }
    }

    // ---- warp-collective ops ---------------------------------------------------------------
    //
    // These are genuinely, natively meaningful here — a real capability advantage over the CPU
    // oracle/regalloc backends, which must refuse them (no concurrent hardware threads to shuffle
    // values between). PTX runs on real SIMT hardware, so this is direct instruction selection,
    // not emulation.

    fn lower_shuffle(&mut self, id: InstId, kind: ShuffleKind, val: ValRef, amt: ValRef, ty: Ty) {
        let dst = self.dst_reg(id);
        let mode = shuffle_mode(kind);
        let clamp = shuffle_clamp(kind);
        let a = self.val_reg(amt).text();
        match reg_class_of(ty) {
            RegClass::B32 => {
                let v = self.val_reg(val).text();
                self.line(&format!(
                    "shfl.sync.{mode}.b32 {}, {v}, {a}, {clamp}, 0xffffffff;",
                    dst.text()
                ));
            }
            RegClass::F32 => {
                let v = self.val_reg(val).text();
                self.line(&format!("mov.b32 {SCRATCH0}, {v};"));
                self.line(&format!(
                    "shfl.sync.{mode}.b32 {SCRATCH1}, {SCRATCH0}, {a}, {clamp}, 0xffffffff;"
                ));
                self.line(&format!("mov.b32 {}, {SCRATCH1};", dst.text()));
            }
            RegClass::B64 | RegClass::F64 => {
                let v = self.val_reg(val).text();
                self.line(&format!("mov.b64 {{{SCRATCH0}, {SCRATCH1}}}, {v};"));
                self.line(&format!(
                    "shfl.sync.{mode}.b32 {SCRATCH0}, {SCRATCH0}, {a}, {clamp}, 0xffffffff;"
                ));
                self.line(&format!(
                    "shfl.sync.{mode}.b32 {SCRATCH1}, {SCRATCH1}, {a}, {clamp}, 0xffffffff;"
                ));
                self.line(&format!(
                    "mov.b64 {}, {{{SCRATCH0}, {SCRATCH1}}};",
                    dst.text()
                ));
            }
            RegClass::Pred => unreachable!("check_module refuses predicate-typed Shuffle"),
        }
    }

    fn lower_ballot(&mut self, id: InstId, v: ValRef) {
        let dst = self.dst_reg(id);
        let p = self.val_reg(v).text();
        self.line(&format!(
            "vote.sync.ballot.b32 {}, {p}, 0xffffffff;",
            dst.text()
        ));
    }

    fn lower_vote(&mut self, id: InstId, v: ValRef, mode: &str) {
        let dst = self.dst_reg(id);
        let p = self.val_reg(v).text();
        self.line(&format!(
            "vote.sync.{mode}.pred {}, {p}, 0xffffffff;",
            dst.text()
        ));
    }

    // ---- atomics ------------------------------------------------------------------------

    fn lower_atomic(
        &mut self,
        id: InstId,
        op: AtomicOp,
        ptr: ValRef,
        val: ValRef,
        space: AddrSpace,
        ty: Ty,
    ) {
        let dst = self.dst_reg(id);
        let addr = self.val_reg(ptr).text();
        let v = self.val_reg(val).text();
        let sw = space_word(space);
        match ty {
            Ty::Scalar(Scalar::F32) | Ty::Scalar(Scalar::F64) => {
                self.lower_atomic_float(dst, op, &addr, &v, sw, ty)
            }
            Ty::Scalar(Scalar::I32) | Ty::Scalar(Scalar::I64) | Ty::Ptr(_) => {
                self.lower_atomic_int(dst, op, &addr, &v, sw, ty)
            }
            _ => unreachable!("check_module refuses sub-32-bit/f16/predicate/vector atomics"),
        }
    }

    /// PTX has `atom.add`/`.exch`/`.min`/`.max`/`.and`/`.or`/`.xor`/`.cas` but no `atom.sub`;
    /// `Sub` lowers as "negate, then add". `Exch`/`And`/`Or`/`Xor` use the bit-generic `.b32`/
    /// `.b64` type (correct regardless of int/pointer content); `Add`/`Min`/`Max` use the
    /// signed type per this backend's uniform div/rem/atomics signedness convention.
    fn lower_atomic_int(&mut self, dst: Reg, op: AtomicOp, addr: &str, v: &str, sw: &str, ty: Ty) {
        let (ssuf, bword) = match ty {
            Ty::Scalar(Scalar::I32) => ("s32", "b32"),
            Ty::Scalar(Scalar::I64) | Ty::Ptr(_) => ("s64", "b64"),
            _ => unreachable!(),
        };
        match op {
            AtomicOp::Add => self.line(&format!(
                "atom.{sw}.add.{ssuf} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Sub => {
                let neg = if bword == "b32" { SCRATCH0 } else { SCRATCH_D0 };
                self.line(&format!("neg.{ssuf} {neg}, {v};"));
                self.line(&format!(
                    "atom.{sw}.add.{ssuf} {}, [{addr}], {neg};",
                    dst.text()
                ));
            }
            AtomicOp::Exch => self.line(&format!(
                "atom.{sw}.exch.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Min => self.line(&format!(
                "atom.{sw}.min.{ssuf} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Max => self.line(&format!(
                "atom.{sw}.max.{ssuf} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::And => self.line(&format!(
                "atom.{sw}.and.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Or => self.line(&format!(
                "atom.{sw}.or.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Xor => self.line(&format!(
                "atom.{sw}.xor.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
        }
    }

    /// `Add` is natively supported (`atom.add.f32`/`.f64`, unconditionally since Pascal);
    /// `Exch`/`And`/`Or`/`Xor` operate on the raw bit pattern via `.b32`/`.b64` (well-defined
    /// regardless of what the bits represent, matching `basalt-x86/src/oracle.rs`'s own note on
    /// the same operations); `Sub` negates then adds; `Min`/`Max` have no float `atom` form this
    /// backend trusts without `ptxas` to check against, so they go through a CAS-retry loop —
    /// see the module header.
    fn lower_atomic_float(
        &mut self,
        dst: Reg,
        op: AtomicOp,
        addr: &str,
        v: &str,
        sw: &str,
        ty: Ty,
    ) {
        let f64_ = matches!(ty, Ty::Scalar(Scalar::F64));
        let f = if f64_ { "f64" } else { "f32" };
        let bword = if f64_ { "b64" } else { "b32" };
        match op {
            AtomicOp::Add => {
                self.line(&format!("atom.{sw}.add.{f} {}, [{addr}], {v};", dst.text()))
            }
            AtomicOp::Exch => self.line(&format!(
                "atom.{sw}.exch.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::And => self.line(&format!(
                "atom.{sw}.and.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Or => self.line(&format!(
                "atom.{sw}.or.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Xor => self.line(&format!(
                "atom.{sw}.xor.{bword} {}, [{addr}], {v};",
                dst.text()
            )),
            AtomicOp::Sub => {
                let neg = if f64_ { SCRATCH_FD0 } else { SCRATCH_F0 };
                self.line(&format!("neg.{f} {neg}, {v};"));
                self.line(&format!(
                    "atom.{sw}.add.{f} {}, [{addr}], {neg};",
                    dst.text()
                ));
            }
            AtomicOp::Min | AtomicOp::Max => {
                self.lower_atomic_float_minmax(dst, op, addr, v, sw, f64_)
            }
        }
    }

    /// CAS-retry loop: read the current value, compute the candidate `min`/`max` in the native
    /// float type, `atom.cas` it in against the value just read, and retry on a mismatch (the
    /// memory location changed under us). `atom.cas` always returns the value actually found in
    /// memory, so on success that is exactly the pre-modification value CUDA's atomic-RMW
    /// semantics call for; on failure it becomes the next iteration's new "current value".
    fn lower_atomic_float_minmax(
        &mut self,
        dst: Reg,
        op: AtomicOp,
        addr: &str,
        v: &str,
        sw: &str,
        f64_: bool,
    ) {
        let loop_label = self.fresh_label("atomic_fminmax_loop");
        let cmp = if matches!(op, AtomicOp::Min) {
            "min"
        } else {
            "max"
        };
        if f64_ {
            self.line(&format!("ld.{sw}.f64 {SCRATCH_FD0}, [{addr}];"));
            self.label(&loop_label);
            self.line(&format!("mov.b64 {SCRATCH_D0}, {SCRATCH_FD0};"));
            self.line(&format!("{cmp}.f64 {SCRATCH_FD1}, {SCRATCH_FD0}, {v};"));
            self.line(&format!("mov.b64 {SCRATCH_D1}, {SCRATCH_FD1};"));
            self.line(&format!(
                "atom.{sw}.cas.b64 {SCRATCH_D1}, [{addr}], {SCRATCH_D0}, {SCRATCH_D1};"
            ));
            self.line(&format!("mov.b64 {SCRATCH_FD0}, {SCRATCH_D1};"));
            self.line(&format!(
                "setp.ne.b64 {PRED_SCRATCH}, {SCRATCH_D1}, {SCRATCH_D0};"
            ));
            self.line(&format!("@{PRED_SCRATCH} bra {loop_label};"));
            self.line(&format!("mov.f64 {}, {SCRATCH_FD0};", dst.text()));
        } else {
            self.line(&format!("ld.{sw}.f32 {SCRATCH_F0}, [{addr}];"));
            self.label(&loop_label);
            self.line(&format!("mov.b32 {SCRATCH0}, {SCRATCH_F0};"));
            self.line(&format!("{cmp}.f32 {SCRATCH_F1}, {SCRATCH_F0}, {v};"));
            self.line(&format!("mov.b32 {SCRATCH1}, {SCRATCH_F1};"));
            self.line(&format!(
                "atom.{sw}.cas.b32 {SCRATCH1}, [{addr}], {SCRATCH0}, {SCRATCH1};"
            ));
            self.line(&format!("mov.b32 {SCRATCH_F0}, {SCRATCH1};"));
            self.line(&format!(
                "setp.ne.b32 {PRED_SCRATCH}, {SCRATCH1}, {SCRATCH0};"
            ));
            self.line(&format!("@{PRED_SCRATCH} bra {loop_label};"));
            self.line(&format!("mov.f32 {}, {SCRATCH_F0};", dst.text()));
        }
    }

    /// `atomicCAS` compares and swaps the raw bit pattern regardless of `ty` (matching
    /// `basalt-x86/src/oracle.rs`'s identically-named function's own note); a float operand is
    /// explicitly reinterpreted into a `.b32`/`.b64` scratch first rather than assumed
    /// cross-type-compatible as a bit-generic instruction operand.
    fn lower_atomic_cas(
        &mut self,
        id: InstId,
        ptr: ValRef,
        cmp: ValRef,
        newv: ValRef,
        space: AddrSpace,
        ty: Ty,
    ) {
        let dst = self.dst_reg(id);
        let addr = self.val_reg(ptr).text();
        let sw = space_word(space);
        match ty {
            Ty::Scalar(Scalar::I32) | Ty::Scalar(Scalar::I64) | Ty::Ptr(_) => {
                let bword = if reg_class_of(ty) == RegClass::B32 {
                    "b32"
                } else {
                    "b64"
                };
                let c = self.val_reg(cmp).text();
                let n = self.val_reg(newv).text();
                self.line(&format!(
                    "atom.{sw}.cas.{bword} {}, [{addr}], {c}, {n};",
                    dst.text()
                ));
            }
            Ty::Scalar(Scalar::F32) => {
                let c = self.val_reg(cmp).text();
                let n = self.val_reg(newv).text();
                self.line(&format!("mov.b32 {SCRATCH0}, {c};"));
                self.line(&format!("mov.b32 {SCRATCH1}, {n};"));
                self.line(&format!(
                    "atom.{sw}.cas.b32 {SCRATCH1}, [{addr}], {SCRATCH0}, {SCRATCH1};"
                ));
                self.line(&format!("mov.f32 {}, {SCRATCH1};", dst.text()));
            }
            Ty::Scalar(Scalar::F64) => {
                let c = self.val_reg(cmp).text();
                let n = self.val_reg(newv).text();
                self.line(&format!("mov.b64 {SCRATCH_D0}, {c};"));
                self.line(&format!("mov.b64 {SCRATCH_D1}, {n};"));
                self.line(&format!(
                    "atom.{sw}.cas.b64 {SCRATCH_D1}, [{addr}], {SCRATCH_D0}, {SCRATCH_D1};"
                ));
                self.line(&format!("mov.f64 {}, {SCRATCH_D1};", dst.text()));
            }
            _ => unreachable!("check_module refuses sub-32-bit/predicate/f16/vector atomicCas"),
        }
    }

    // ---- phi resolution -------------------------------------------------------------------

    fn emit_mov(&mut self, dst: Reg, src: Reg) {
        self.line(&format!(
            "mov.{} {}, {};",
            reg_type_word(dst.class),
            dst.text(),
            src.text()
        ));
    }

    fn copy_into(&mut self, dst_id: InstId, val: ValRef) {
        let dst_regs = self.dst_regs(dst_id);
        let src_regs = self.val_regs(val).to_vec();
        for (d, s) in dst_regs.iter().zip(src_regs.iter()) {
            self.emit_mov(*d, *s);
        }
    }

    fn emit_phi_copies(&mut self, from: u32, to: u32) {
        let Some(copies) = self.phi_copies.get(&(from, to)).cloned() else {
            return;
        };
        for (phi_id, val) in copies {
            self.copy_into(phi_id, val);
        }
    }

    // ---- terminators --------------------------------------------------------------------

    fn lower_term(&mut self, from_block: u32, term: &Term) {
        match term {
            Term::Br(target) => {
                self.emit_phi_copies(from_block, target.0);
                self.line(&format!("bra $L{};", target.0));
            }
            Term::CondBr(cond, t, f) => {
                let p = self.val_reg(*cond).text();
                let needs_t = self.phi_copies.contains_key(&(from_block, t.0));
                let needs_f = self.phi_copies.contains_key(&(from_block, f.0));
                if !needs_t && !needs_f {
                    self.line(&format!("@{p} bra $L{};", t.0));
                    self.line(&format!("bra $L{};", f.0));
                    return;
                }
                let true_prep = self.fresh_label("condbr_true");
                let false_prep = self.fresh_label("condbr_false");
                self.line(&format!("@{p} bra {true_prep};"));
                self.line(&format!("bra {false_prep};"));
                self.label(&true_prep);
                self.emit_phi_copies(from_block, t.0);
                self.line(&format!("bra $L{};", t.0));
                self.label(&false_prep);
                self.emit_phi_copies(from_block, f.0);
                self.line(&format!("bra $L{};", f.0));
            }
            Term::Switch(scrut, default, cases) => {
                let sty = self.valref_ty(*scrut);
                let scrut_reg = self.val_reg(*scrut);
                let (sv, tsuf) = match sty {
                    Ty::Scalar(s @ (Scalar::I8 | Scalar::I16)) => {
                        (self.canon_int_reg(scrut_reg, s, true, SCRATCH0), "s32")
                    }
                    Ty::Scalar(Scalar::I32) => (scrut_reg.text(), "s32"),
                    Ty::Scalar(Scalar::I64) => (scrut_reg.text(), "s64"),
                    Ty::Ptr(_) => (scrut_reg.text(), "u64"),
                    _ => unreachable!("a switch scrutinee is always integer/pointer typed"),
                };
                for &(case_val, target) in cases {
                    self.line(&format!("setp.eq.{tsuf} {PRED_SCRATCH}, {sv}, {case_val};"));
                    if self.phi_copies.contains_key(&(from_block, target.0)) {
                        let take = self.fresh_label("switch_take");
                        let skip = self.fresh_label("switch_skip");
                        self.line(&format!("@{PRED_SCRATCH} bra {take};"));
                        self.line(&format!("bra {skip};"));
                        self.label(&take);
                        self.emit_phi_copies(from_block, target.0);
                        self.line(&format!("bra $L{};", target.0));
                        self.label(&skip);
                    } else {
                        self.line(&format!("@{PRED_SCRATCH} bra $L{};", target.0));
                    }
                }
                self.emit_phi_copies(from_block, default.0);
                self.line(&format!("bra $L{};", default.0));
            }
            Term::Ret(v) => {
                // A `.visible .entry` kernel has no way to hand a value back to its caller (the
                // host); a non-void `Ret` at kernel scope drops the value rather than fail — see
                // the module header.
                let _ = v;
                self.line("ret;");
            }
        }
    }
}

// ---- module/function assembly ------------------------------------------------------------

fn emit_function(f: &Function, out: &mut String) {
    let alloc = RegAlloc::build(f);
    let locals = LocalSlots::build(f);
    let phi_copies = build_phi_copies(f);

    let params: Vec<String> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, &ty)| param_decl(&f.name, i, ty))
        .collect();
    out.push_str(&format!(
        ".visible .entry {}({})\n",
        f.name,
        params.join(", ")
    ));
    out.push_str("{\n");

    for (class, count) in alloc.counted_classes() {
        out.push_str(&format!(
            "\t.reg .{} %{}<{}>;\n",
            reg_type_word(class),
            reg_prefix(class),
            count
        ));
    }
    out.push_str("\t.reg .b32 %rs0, %rs1, %rs2;\n");
    out.push_str("\t.reg .b64 %rds0, %rds1;\n");
    out.push_str("\t.reg .f32 %fs0, %fs1;\n");
    out.push_str("\t.reg .f64 %fds0, %fds1;\n");
    out.push_str("\t.reg .pred %ps0;\n");

    for (space, _, sym) in &locals.order {
        out.push_str(&format!(
            "\t.{} .align 8 .b8 {sym}[8];\n",
            local_decl_space_word(*space)
        ));
    }

    let mut cg = CodeGen {
        f,
        alloc,
        locals,
        phi_copies,
        out: String::new(),
        label_counter: 0,
    };
    cg.emit_param_loads();
    for (bidx, block) in f.blocks.iter().enumerate() {
        cg.label(&format!("$L{bidx}"));
        for &inst_id in &block.insts {
            cg.lower_inst(inst_id);
        }
        cg.lower_term(bidx as u32, &block.term);
    }
    out.push_str(&cg.out);
    out.push_str("}\n");
}

fn emit_module(module: &Module) -> String {
    let mut out = String::new();
    out.push_str(".version 8.0\n");
    out.push_str(".target sm_70\n");
    out.push_str(".address_size 64\n");
    for f in &module.funcs {
        out.push('\n');
        emit_function(f, &mut out);
    }
    out
}

// ---- Backend impl -------------------------------------------------------------------------

/// The NVIDIA PTX text backend. `name()` returns `"nvidia-ptx"`, matching `basalt-cli`'s own
/// `--nvidia-ptx` flag spelling for a later, separate CLI wire-up. See the module header for
/// the full design.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ptx;

impl Backend for Ptx {
    fn name(&self) -> &'static str {
        "nvidia-ptx"
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
        let text = emit_module(&ssa_module);
        Ok(Artifact::text(ArtifactKind::Ptx, text))
    }
}

#[cfg(test)]
mod tests;
