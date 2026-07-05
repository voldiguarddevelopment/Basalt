// The RV32IM `Backend` impl (`Rv32`): lowers one BIR `Function` to real RV32IM machine code.
// Same design stance as `basalt-x86`'s oracle (see that crate's `oracle.rs` module header,
// which this file mirrors deliberately) — stack-everything, zero register allocation, a
// single straightforward codegen pass. This is a first real backend bring-up, not the
// project's oracle (that role stays `basalt-x86` exclusively, per `CLAUDE.md` invariant 4),
// but the same correctness-first, no-cleverness stance applies here for the same reason: get
// a real encoding right before any optimization work is even considered.
//
// # The SIMT-via-a-native-loop threading model
//
// Identical convention to `basalt-x86/src/oracle.rs`'s own `# The SIMT-via-a-native-loop
// threading model` section, so the *same* GPU kernels can eventually be diff-tested against
// the oracle (a later task's job, not this one's) — every BIR `Function` becomes one native
// function whose signature is its own params plus a trailing synthesized `nthreads`
// parameter; `tid.x` reads a native loop counter, `bdim.x` reads `nthreads`,
// `blockDim.{y,z}`/`gridDim.{y,z}` are fixed at 1, `blockIdx.*` at 0. `barrier` is a `nop`
// landmark (no real concurrency to guard against, single-thread-at-a-time execution).
// `shuffle`/`ballot`/`vote.any`/`vote.all` are refused (`E090`) for the identical reason the
// oracle refuses them: warp-collective semantics need several threads' values live at once,
// which a one-thread-at-a-time interpreter cannot express.
//
// # Calling convention: RV32 `ilp32` (soft-float ABI)
//
// Every parameter is integer-class — RV32IM has no hardware float registers at all, so
// there is no separate float argument class to track (unlike SysV x86-64's split integer/SSE
// sequences). A 32-bit-or-narrower scalar or pointer consumes one of `a0..a7` in order; an
// `i64` consumes a *pair* of consecutive argument registers, aligned to an even index if it
// wouldn't otherwise land on one (the real RV32 `ilp32` calling convention's own rule for
// wide scalars) — see `classify_params`. `nthreads` always takes the next argument register
// after the function's own params, under the same alignment rule (though as a plain `i32` it
// never itself needs the pair alignment). More than 8 argument registers' worth (including
// `nthreads`) is refused (`E093`) rather than spilling to the stack, matching the x86 oracle's
// own stance on stack-passed arguments. Returns: `void` -> nothing; a 32-bit-or-narrower
// scalar/pointer -> `a0`; `i64`/`f64` -> `a0` (low word) / `a1` (high word).
//
// `div`/`rem` are always lowered as signed (matching the x86 oracle's own documented stance:
// BIR's `Bin` op carries no signed/unsigned distinction for these, so this backend picks one
// interpretation and documents it rather than inventing a `udiv` BIR has no way to ask for).
//
// # Synthesizing local/param/shared/constant addresses
//
// Identical idiom to `basalt-x86/src/oracle.rs`'s own `# Synthesizing local/param/shared/
// constant addresses` section: `basalt-sema`'s lowering hands every local/parameter/shared/
// constant storage location a synthesized `const.i ptr.<space> (slot_id * 65536)` address:
// an opaque per-`(space, value)` slot id, never a real address. This backend treats each
// distinct `(space, const-value)` pair as an opaque slot identifier and assigns it a real
// stack cell the first time it is seen (`Frame::const_addr_disp`), folding `Shared`/
// `Constant`/`Local`/`Param` into the same real-stack-memory treatment (no actual
// shared-vs-local distinction matters when exactly one thread ever executes at a time).
// `AddrSpace::Global` values are never synthesized this way; they arrive as real 32-bit
// addresses (an incoming pointer argument, or arithmetic on one) from the start.
//
// # Stack-everything, always via a computed address
//
// Every instruction with a result gets its own fixed-size 8-byte stack slot (uniformly
// 8 bytes regardless of the value's real width, mirroring the x86 oracle's own reasoning: it
// keeps every op's memory access at *exactly* its declared width with no path depending on
// whatever garbage sits in a slot's unused remainder). Unlike the x86 oracle's `[rbp+disp32]`
// addressing (a single instruction, since x86 has a displacement-immediate addressing mode),
// RV32's `lw`/`sw` immediate is only 12 bits (`-2048..=2047`) — nowhere near enough to
// address a frame of any real size directly. Rather than a two-tier "use the short form when
// it fits" scheme (exactly the kind of size-dependent branching the x86 encoder's own
// disp32-always policy rejects), every stack access in this backend materializes its address
// from scratch via `Frame::addr` (`li32` the offset, `add` against `sp`) into a scratch
// register, then issues an ordinary zero-offset load/store — one code path, no
// frame-size-dependent concern ever arises. `sp` itself never moves between the prologue's
// one `sub` and the epilogue's one `add`, so every slot's offset is a fixed compile-time
// constant throughout the function body, exactly like the x86 oracle's fixed `rbp`.
//
// # Soft float
//
// RV32IM has no F/D extension (that hardware doesn't exist in this backend's scope — see
// `TASKS.md`). Every `f32` arithmetic op (`FAdd`/`FSub`/`FMul`/`FDiv`/`FRem`, all eight
// `FCmp` predicates, `fptosi`/`fptoui`/`sitofp`/`uitofp` against `i8`/`i16`/`i32`) lowers to a
// `jal` into one of `softfloat.rs`'s internal routines (`__sf_f32_*`), embedded once per
// emitted object (see `emit_function`'s trailing `softfloat::emit_runtime` call) — never an
// external libm/libgcc call, matching this project's hand-rolled ethos and the fact that
// nothing guarantees a cross-compiled soft-float library is present at this backend's
// eventual load site. `FRem` is not its own routine: like the x86 oracle's `lower_frem`, it
// composes already-existing primitives (`a - trunc(a/b)*b`, via `__sf_f32_div` + the
// `fptosi`/`sitofp` routines + `__sf_f32_mul` + `__sf_f32_add` with a sign-flipped operand).
//
// `f64` is a documented, deliberate scope cut, narrower than the x86 oracle's own float
// coverage: an `f64` value can be stored in a local/global, loaded, `bitcast`, `select`ed,
// `phi`'d, and constructed via `const.f`, but **no `f64` arithmetic is lowered** —
// `FAdd`/`FSub`/`FMul`/`FDiv`/`FRem`/`FCmp` on `f64`, `fpext`/`fptrunc`, and any
// `fptosi`/`fptoui`/`sitofp`/`uitofp` touching `f64` are refused with `E091`. Correct `f64`
// software arithmetic needs a genuinely wider (mantissa doesn't fit one 32-bit register)
// algorithm than `f32`'s — this task's `softfloat.rs` needed real, carefully-derived 48-bit-
// wide alignment logic just for `f32` add (see that file's own header), and this project has
// no way to execution-test any of it yet (no RV32 simulator exists in this tree — a later
// task's job). Shipping an equally intricate, *wider*, and *even less reviewable* `f64`
// algorithm on the same unverified footing is exactly the "silently wrong codegen" risk
// `CLAUDE.md` invariant 3 warns against; refusing cleanly instead is the honest call. `i64`
// arithmetic beyond `add`/`sub`/`mul`/`and`/`or`/`xor`/`icmp` (i.e. variable-amount
// `shl`/`lshr`/`ashr` and `div`/`rem`) is refused for the same reason — a general 64-bit
// variable shift or long division is real, non-trivial, unverified-without-execution-testing
// logic this task deliberately does not gamble on.
//
// # `mma`
//
// Refused outright (`E099`, `MatrixPathUnsupported`): RV32IM has no vector/matrix unit, and a
// software triple-loop `mma` (as the x86 oracle implements) is out of this task's stated
// scope (`TASKS.md`: "no vector units in this task's scope").

use std::collections::HashMap;

use basalt_backend::{
    write_elf_object, Architecture, Artifact, ArtifactKind, Backend, ElfObjectSpec, EmitOpts,
    Endianness, Support,
};
use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, CastOp, FCmpPred, Function, ICmpPred, InstId, Module, Op, Scalar,
    Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};

use crate::enc::AluOp;
use crate::enc::{
    BCond, Enc, MulOp, A0, A1, A2, A3, ARG_REGS, RA, SP, T0, T1, T2, T3, T4, T5, T6, ZERO,
};
use crate::softfloat;

/// The RV32IM hand-rolled backend: correct-first, never clever. See the module header for the
/// full design.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rv32;

impl Backend for Rv32 {
    fn name(&self) -> &'static str {
        "rv32im"
    }

    fn supports(&self, module: &Module) -> Support {
        match check_module(module) {
            Ok(_) => Support::Supported,
            Err(diag) => Support::Unsupported(diag.code),
        }
    }

    fn emit(&self, module: &Module, _opts: &EmitOpts) -> Result<Artifact, Diag> {
        let f = check_module(module)?;
        let text = emit_function(f)?;
        let spec = ElfObjectSpec::new(
            Architecture::Riscv32,
            Endianness::Little,
            f.name.clone(),
            text,
        );
        let bytes = write_elf_object(&spec)?;
        Ok(Artifact::bytes(ArtifactKind::Object, bytes))
    }
}

/// Single source of truth for what this backend refuses, shared by `supports()` and `emit()`
/// so the two can never drift apart. Returns the one function to lower on success.
fn check_module(module: &Module) -> Result<&Function, Diag> {
    if module.funcs.len() != 1 {
        return Err(Diag::new(ECode::UnsupportedFeature)
            .with_arg("multi-function module: ElfObjectSpec names exactly one symbol"));
    }
    let f = &module.funcs[0];
    if !f.is_kernel {
        return Err(Diag::new(ECode::UnsupportedFeature)
            .with_arg("host/non-kernel function compilation is not yet implemented"));
    }

    if matches!(f.ret, Ty::Vec(..)) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed return value"));
    }
    if f.params.iter().any(|t| matches!(t, Ty::Vec(..))) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed parameter"));
    }
    if ty_is_f16(f.ret) {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("f16 has no hardware or soft-float support in this backend"));
    }
    if f.params.iter().any(|&t| ty_is_f16(t)) {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("f16 has no hardware or soft-float support in this backend"));
    }
    if classify_params(&f.params).is_none() {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
            "argument classes overflow the 8 RV32 ilp32 argument registers (incl. trailing nthreads)",
        ));
    }

    for inst in &f.insts {
        if matches!(inst.ty, Ty::Vec(..)) {
            return Err(
                Diag::new(ECode::UnsupportedType).with_arg("vector-typed instruction result")
            );
        }
        if ty_is_f16(inst.ty) {
            return Err(Diag::new(ECode::UnsupportedType)
                .with_arg("f16 has no hardware or soft-float support in this backend"));
        }
        match &inst.op {
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                    "warp-collective op has no meaning under one-thread-at-a-time execution",
                ));
            }
            Op::Mma { .. } => {
                return Err(Diag::new(ECode::MatrixPathUnsupported)
                    .with_arg("no vector/matrix unit in this backend's scope"));
            }
            Op::Cast(cop, sty, _) if ty_is_f16(*sty) => {
                let _ = cop;
                return Err(Diag::new(ECode::UnsupportedType)
                    .with_arg("f16 has no hardware or soft-float support in this backend"));
            }
            Op::FCmp(_, cty, _, _) if ty_is_f16(*cty) => {
                return Err(Diag::new(ECode::UnsupportedType)
                    .with_arg("f16 has no hardware or soft-float support in this backend"));
            }
            Op::Store { ty: sty, .. } if ty_is_f16(*sty) => {
                return Err(Diag::new(ECode::UnsupportedType)
                    .with_arg("f16 has no hardware or soft-float support in this backend"));
            }
            Op::Bin(op, _, _)
                if is_f64(inst.ty)
                    && matches!(
                        op,
                        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem
                    ) =>
            {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "f64 arithmetic is a documented scope cut in this backend (see lower.rs's own header); f32 is fully supported",
                ));
            }
            Op::Bin(op, _, _)
                if matches!(inst.ty, Ty::Scalar(Scalar::I64))
                    && matches!(
                        op,
                        BinOp::Shl | BinOp::Lshr | BinOp::Ashr | BinOp::Div | BinOp::Rem
                    ) =>
            {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "i64 shl/lshr/ashr/div/rem are a documented scope cut in this backend (see lower.rs's own header); i64 add/sub/mul/and/or/xor/icmp are fully supported",
                ));
            }
            Op::FCmp(_, cty, _, _) if is_f64(*cty) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "f64 arithmetic is a documented scope cut in this backend (see lower.rs's own header); f32 is fully supported",
                ));
            }
            Op::Cast(cop, sty, _)
                if matches!(cop, CastOp::FpExt | CastOp::FpTrunc)
                    || (matches!(
                        cop,
                        CastOp::FpToSi | CastOp::FpToUi | CastOp::SiToFp | CastOp::UiToFp
                    ) && (is_f64(*sty) || is_f64(inst.ty))) =>
            {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "f64 conversions are a documented scope cut in this backend (see lower.rs's own header)",
                ));
            }
            Op::Cast(cop, sty, _)
                if matches!(
                    cop,
                    CastOp::FpToSi | CastOp::FpToUi | CastOp::SiToFp | CastOp::UiToFp
                ) && (matches!(sty, Ty::Scalar(Scalar::I64))
                    || matches!(inst.ty, Ty::Scalar(Scalar::I64))) =>
            {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "i64<->float conversions are a documented scope cut in this backend (see lower.rs's own header)",
                ));
            }
            Op::Atomic(_, _, _, _) if is_f64(inst.ty) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "f64 atomics are a documented scope cut in this backend (see lower.rs's own header)",
                ));
            }
            Op::KernelLaunch { .. }
            | Op::CudaMalloc { .. }
            | Op::CudaMemcpy { .. }
            | Op::CudaFree { .. }
            | Op::CudaDeviceSynchronize => {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                    "kernel launch / CUDA Runtime API calls are sema-only today (see \
                     Op::KernelLaunch's own doc comment); this backend has no host-side dispatch \
                     story for them yet",
                ));
            }
            Op::Call { .. } => {
                return Err(Diag::new(ECode::UnsupportedOp)
                    .with_arg("function calls have no lowering in this backend yet"));
            }
            _ => {}
        }
    }

    Ok(f)
}

fn ty_is_f16(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F16))
}

fn is_float(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F32) | Ty::Scalar(Scalar::F64))
}

fn is_f64(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F64))
}

fn is_wide(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::I64) | Ty::Scalar(Scalar::F64))
}

/// Byte width of a single-register-class value. `Ty::Ptr` is 4 bytes on RV32 (`XLEN=32`,
/// unlike `basalt-x86`'s 8-byte pointers) — this is the one width difference from the x86
/// oracle's own `width_of` that matters throughout this file. `i64`/`f64` are handled
/// separately wherever a width matters (they occupy two consecutive 4-byte words within a
/// slot) rather than reported here.
#[derive(Clone, Copy, PartialEq, Eq)]
enum W {
    B1,
    B2,
    B4,
}

fn width_of(ty: Ty) -> W {
    match ty {
        Ty::Scalar(Scalar::I1 | Scalar::I8) => W::B1,
        Ty::Scalar(Scalar::I16) => W::B2,
        Ty::Scalar(Scalar::I32 | Scalar::F32) => W::B4,
        Ty::Ptr(_) => W::B4,
        Ty::Scalar(Scalar::I64 | Scalar::F64)
        | Ty::Scalar(Scalar::F16)
        | Ty::Vec(..)
        | Ty::Void => {
            unreachable!("width_of called on a wide/refused type check_module should have caught")
        }
    }
}

fn local_like(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Shared | AddrSpace::Constant
    )
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

fn block_label(id: u32) -> String {
    format!("bb{id}")
}

#[derive(Clone, Copy)]
enum ArgLoc {
    Single(u8),
    Pair(u8, u8),
}

/// Classifies `params` into the RV32 `ilp32` argument-register sequence (see the module
/// header) and returns the location the trailing `nthreads` parameter lands in. `None` means
/// the signature overflows the 8 argument registers (counting `nthreads`) — this backend does
/// not implement stack-passed arguments, matching the x86 oracle's own stance.
fn classify_params(params: &[Ty]) -> Option<(Vec<ArgLoc>, ArgLoc)> {
    let mut idx: usize = 0;
    let mut locs = Vec::with_capacity(params.len());
    for &ty in params {
        if is_wide(ty) {
            if idx % 2 == 1 {
                idx += 1;
            }
            if idx + 1 >= ARG_REGS.len() {
                return None;
            }
            locs.push(ArgLoc::Pair(ARG_REGS[idx], ARG_REGS[idx + 1]));
            idx += 2;
        } else {
            if idx >= ARG_REGS.len() {
                return None;
            }
            locs.push(ArgLoc::Single(ARG_REGS[idx]));
            idx += 1;
        }
    }
    if idx >= ARG_REGS.len() {
        return None;
    }
    Some((locs, ArgLoc::Single(ARG_REGS[idx])))
}

/// This function's real native stack frame: every fixed home plus one 8-byte slot per BIR
/// instruction result and one per synthesized local/param/shared/constant address. Offsets
/// grow upward from 0 (`sp` after the prologue's `sub` points at the lowest address in the
/// frame), unlike the x86 oracle's negative-from-`rbp` offsets — a stylistic difference with
/// no semantic weight, just the natural direction for a frame addressed purely off `sp`.
struct Frame {
    param_home: Vec<i32>,
    nthreads_home: i32,
    loopctr_home: i32,
    ra_home: i32,
    retval_home: Option<i32>,
    inst_slot: Vec<i32>,
    const_addr_disp: HashMap<(u8, i64), i32>,
    /// Two fixed cross-call scratch homes: any value that must survive a `jal` into the
    /// soft-float runtime has to live in memory, not a register — `softfloat.rs`'s routines
    /// are free to clobber any GPR (see that file's own header), so a register stashed across
    /// a `call` is not actually preserved. `lower_frem32` and the float-atomic path are the
    /// only lowerings that need this (both need at most two live cross-call values at once).
    scratch0: i32,
    scratch1: i32,
    scratch2: i32,
    frame_size: i32,
}

fn next_slot(offset: &mut i32) -> i32 {
    let cur = *offset;
    *offset += 8;
    cur
}

impl Frame {
    fn build(f: &Function) -> Frame {
        let mut offset: i32 = 0;

        // `ra_home` first: every function this backend emits calls into the soft-float
        // runtime unconditionally reachable via `jal` (see `emit_function`), so it is never a
        // true leaf and always needs to preserve its own incoming return address across any
        // internal call it makes.
        let ra_home = next_slot(&mut offset);
        let scratch0 = next_slot(&mut offset);
        let scratch1 = next_slot(&mut offset);
        let scratch2 = next_slot(&mut offset);
        let param_home: Vec<i32> = f.params.iter().map(|_| next_slot(&mut offset)).collect();
        let nthreads_home = next_slot(&mut offset);
        let loopctr_home = next_slot(&mut offset);
        let retval_home = if matches!(f.ret, Ty::Void) {
            None
        } else {
            Some(next_slot(&mut offset))
        };
        let inst_slot: Vec<i32> = f.insts.iter().map(|_| next_slot(&mut offset)).collect();

        let mut const_addr_disp = HashMap::new();
        for inst in &f.insts {
            if let (Op::ConstInt(n), Ty::Ptr(space)) = (&inst.op, inst.ty) {
                if local_like(space) {
                    let key = (space_tag(space), *n);
                    const_addr_disp
                        .entry(key)
                        .or_insert_with(|| next_slot(&mut offset));
                }
            }
        }

        let frame_size = (offset + 15) & !15;
        Frame {
            param_home,
            nthreads_home,
            loopctr_home,
            ra_home,
            retval_home,
            inst_slot,
            const_addr_disp,
            scratch0,
            scratch1,
            scratch2,
            frame_size,
        }
    }
}

/// Per-edge phi lowering: identical in structure to `basalt-x86/src/oracle.rs`'s own
/// `build_phi_copies` (this logic is purely about BIR's own block/value graph, not about any
/// backend's encoding, so the two are expected to look alike).
type PhiCopies = HashMap<(u32, u32), Vec<(i32, ValRef, Ty)>>;

fn build_phi_copies(f: &Function, frame: &Frame) -> PhiCopies {
    let mut map: PhiCopies = HashMap::new();
    for (bidx, block) in f.blocks.iter().enumerate() {
        for &inst_id in &block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            if let Op::Phi(preds) = &inst.op {
                let dest = frame.inst_slot[inst_id.0 as usize];
                for &(pred_block, val) in preds {
                    map.entry((pred_block.0, bidx as u32))
                        .or_default()
                        .push((dest, val, inst.ty));
                }
            }
        }
    }
    map
}

struct CodeGen<'a> {
    f: &'a Function,
    frame: Frame,
    enc: Enc,
    label_counter: u32,
    phi_copies: PhiCopies,
}

fn emit_function(f: &Function) -> Result<Vec<u8>, Diag> {
    let (param_locs, nthreads_loc) =
        classify_params(&f.params).expect("check_module already validated the signature");

    let frame = Frame::build(f);
    let phi_copies = build_phi_copies(f, &frame);
    let mut cg = CodeGen {
        f,
        frame,
        enc: Enc::new(),
        label_counter: 0,
        phi_copies,
    };

    cg.enc.li32(T0, cg.frame.frame_size);
    cg.enc.alu_reg(AluOp::Sub, SP, SP, T0);

    cg.frame_addr(T0, cg.frame.ra_home);
    cg.enc.sw(RA, 0, T0);

    for (i, loc) in param_locs.iter().enumerate() {
        let disp = cg.frame.param_home[i];
        cg.frame_addr(T0, disp);
        match *loc {
            ArgLoc::Single(r) => cg.enc.sw(r, 0, T0),
            ArgLoc::Pair(lo, hi) => {
                cg.enc.sw(lo, 0, T0);
                cg.enc.sw(hi, 4, T0);
            }
        }
    }
    if let ArgLoc::Single(r) = nthreads_loc {
        cg.frame_addr(T0, cg.frame.nthreads_home);
        cg.enc.sw(r, 0, T0);
    }

    cg.frame_addr(T0, cg.frame.loopctr_home);
    cg.enc.sw(ZERO, 0, T0);

    cg.enc.label("__loop_check");
    cg.frame_addr(T0, cg.frame.loopctr_home);
    cg.enc.lw(T1, 0, T0);
    cg.frame_addr(T0, cg.frame.nthreads_home);
    cg.enc.lw(T2, 0, T0);
    cg.enc.branch(BCond::Ge, T1, T2, "__loop_end");

    for (bidx, block) in f.blocks.iter().enumerate() {
        cg.enc.label(&block_label(bidx as u32));
        for &inst_id in &block.insts {
            cg.lower_inst(inst_id);
        }
        cg.lower_term(bidx as u32, &block.term);
    }

    cg.enc.label("__loop_incr");
    cg.frame_addr(T0, cg.frame.loopctr_home);
    cg.enc.lw(T1, 0, T0);
    cg.enc.addi(T1, T1, 1);
    cg.enc.sw(T1, 0, T0);
    cg.enc.jump("__loop_check");

    cg.enc.label("__loop_end");
    if !matches!(f.ret, Ty::Void) {
        let disp = cg
            .frame
            .retval_home
            .expect("non-void return always gets a retval home");
        cg.frame_addr(T0, disp);
        if is_wide(f.ret) {
            cg.enc.lw(A0, 0, T0);
            cg.enc.lw(A1, 4, T0);
        } else {
            cg.enc.lw(A0, 0, T0);
        }
    }
    cg.frame_addr(T0, cg.frame.ra_home);
    cg.enc.lw(RA, 0, T0);
    cg.enc.li32(T0, cg.frame.frame_size);
    cg.enc.alu_reg(AluOp::Add, SP, SP, T0);
    cg.enc.ret();

    softfloat::emit_runtime(&mut cg.enc);

    Ok(cg.enc.finish())
}

impl<'a> CodeGen<'a> {
    fn fresh_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("__{prefix}_{}", self.label_counter)
    }

    /// Materializes `sp + disp` into `dst` from scratch (`li32` + `add`) — see the module
    /// header's `# Stack-everything, always via a computed address` section for why this
    /// backend never uses a direct 12-bit-immediate stack access.
    fn frame_addr(&mut self, dst: u8, disp: i32) {
        self.enc.li32(dst, disp);
        self.enc.alu_reg(AluOp::Add, dst, dst, SP);
    }

    /// Saves `reg` to a fixed cross-call scratch home (`Frame::scratch0`/`scratch1`) — see
    /// `Frame`'s own doc comment for why this exists at all (nothing survives a `call` in a
    /// register). Uses `T6` as its own addressing scratch, so callers must not stash `T6`
    /// itself.
    fn stash(&mut self, slot_disp: i32, reg: u8) {
        self.frame_addr(T6, slot_disp);
        self.enc.sw(reg, 0, T6);
    }

    fn unstash(&mut self, slot_disp: i32, reg: u8) {
        self.frame_addr(T6, slot_disp);
        self.enc.lw(reg, 0, T6);
    }

    fn valref_ty(&self, v: ValRef) -> Ty {
        match v {
            ValRef::Param(i) => self.f.params[i as usize],
            ValRef::Val(id) => self.f.insts[id.0 as usize].ty,
        }
    }

    fn operand_disp(&self, v: ValRef) -> i32 {
        match v {
            ValRef::Param(i) => self.frame.param_home[i as usize],
            ValRef::Val(id) => self.frame.inst_slot[id.0 as usize],
        }
    }

    /// Reloads a single-word (<=32-bit) value at its own exact width, sign- or zero-extending
    /// per `signed` (RV32's `lb`/`lh` vs `lbu`/`lhu` do this in one instruction — no separate
    /// extend step, unlike x86's two-step `movsx`/`movzx` over a narrower `mov`).
    fn reload(&mut self, v: ValRef, dst: u8, w: W, signed: bool) {
        let disp = self.operand_disp(v);
        self.frame_addr(T4, disp);
        match (w, signed) {
            (W::B1, true) => self.enc.lb(dst, 0, T4),
            (W::B1, false) => self.enc.lbu(dst, 0, T4),
            (W::B2, true) => self.enc.lh(dst, 0, T4),
            (W::B2, false) => self.enc.lhu(dst, 0, T4),
            (W::B4, _) => self.enc.lw(dst, 0, T4),
        }
    }

    fn store_result(&mut self, id: InstId, src: u8, w: W) {
        let disp = self.frame.inst_slot[id.0 as usize];
        self.frame_addr(T4, disp);
        match w {
            W::B1 => self.enc.sb(src, 0, T4),
            W::B2 => self.enc.sh(src, 0, T4),
            W::B4 => self.enc.sw(src, 0, T4),
        }
    }

    /// Reloads a wide (`i64`/`f64`) value's two words into `lo`/`hi`.
    fn reload_wide(&mut self, v: ValRef, lo: u8, hi: u8) {
        let disp = self.operand_disp(v);
        self.frame_addr(T4, disp);
        self.enc.lw(lo, 0, T4);
        self.enc.lw(hi, 4, T4);
    }

    fn store_result_wide(&mut self, id: InstId, lo: u8, hi: u8) {
        let disp = self.frame.inst_slot[id.0 as usize];
        self.frame_addr(T4, disp);
        self.enc.sw(lo, 0, T4);
        self.enc.sw(hi, 4, T4);
    }

    /// Copies `v` (of type `ty`) straight into instruction `id`'s own result slot — `select`'s
    /// two arms and any same-type move.
    fn copy_value(&mut self, v: ValRef, id: InstId, ty: Ty) {
        if is_wide(ty) {
            self.reload_wide(v, T0, T1);
            self.store_result_wide(id, T0, T1);
        } else {
            let w = width_of(ty);
            self.reload(v, T0, w, false);
            self.store_result(id, T0, w);
        }
    }

    fn lower_inst(&mut self, id: InstId) {
        let f = self.f;
        let inst = &f.insts[id.0 as usize];
        let ty = inst.ty;
        match &inst.op {
            Op::ConstInt(n) => {
                let n = *n;
                if let Ty::Ptr(space) = ty {
                    if local_like(space) {
                        let key = (space_tag(space), n);
                        let disp = *self
                            .frame
                            .const_addr_disp
                            .get(&key)
                            .expect("Frame::build pre-scans every local-slot constant");
                        self.frame_addr(T0, disp);
                        self.store_result(id, T0, W::B4);
                        return;
                    }
                }
                if is_wide(ty) {
                    self.enc.li32(T0, n as i32);
                    self.enc.li32(T1, (n >> 32) as i32);
                    self.store_result_wide(id, T0, T1);
                } else {
                    self.enc.li32(T0, n as i32);
                    self.store_result(id, T0, width_of(ty));
                }
            }
            Op::ConstFloat(v) => {
                let v = *v;
                if is_f64(ty) {
                    let bits = v.to_bits();
                    self.enc.li32(T0, bits as i32);
                    self.enc.li32(T1, (bits >> 32) as i32);
                    self.store_result_wide(id, T0, T1);
                } else {
                    let bits = (v as f32).to_bits();
                    self.enc.li32(T0, bits as i32);
                    self.store_result(id, T0, W::B4);
                }
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
                self.reload(c, T0, W::B1, false);
                let else_label = self.fresh_label("select_false");
                let end_label = self.fresh_label("select_end");
                self.enc.branch(BCond::Eq, T0, ZERO, &else_label);
                self.copy_value(a, id, ty);
                self.enc.jump(&end_label);
                self.enc.label(&else_label);
                self.copy_value(b, id, ty);
                self.enc.label(&end_label);
            }
            Op::Cast(cop, sty, v) => {
                let (cop, sty, v) = (*cop, *sty, *v);
                self.lower_cast(id, cop, sty, v, ty);
            }
            Op::Load { ptr, .. } => {
                let ptr = *ptr;
                self.lower_load(id, ptr, ty);
            }
            Op::Store {
                ptr, val, ty: sty, ..
            } => {
                let (ptr, val, sty) = (*ptr, *val, *sty);
                self.lower_store(ptr, val, sty);
            }
            Op::Phi(_) => {
                // Nothing at the definition site: every predecessor writes this
                // instruction's own result slot before jumping here (`emit_phi_copies`).
            }
            Op::TidX => {
                let d = self.frame.loopctr_home;
                self.lower_reload_dyn(id, d, ty);
            }
            Op::BdimX => {
                let d = self.frame.nthreads_home;
                self.lower_reload_dyn(id, d, ty);
            }
            Op::TidY | Op::TidZ | Op::BidX | Op::BidY | Op::BidZ => {
                self.enc.li32(T0, 0);
                self.store_result(id, T0, width_of(ty));
            }
            Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ => {
                self.enc.li32(T0, 1);
                self.store_result(id, T0, width_of(ty));
            }
            Op::Barrier => self.enc.nop(),
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
                unreachable!("check_module refuses these before codegen starts")
            }
            Op::Atomic(aop, ptr, val, _space) => {
                let (aop, ptr, val) = (*aop, *ptr, *val);
                self.lower_atomic(id, aop, ptr, val, ty);
            }
            Op::AtomicCas(ptr, cmp, newv, _space) => {
                let (ptr, cmp, newv) = (*ptr, *cmp, *newv);
                self.lower_atomic_cas(id, ptr, cmp, newv, ty);
            }
            Op::Mma { .. } => unreachable!("check_module refuses mma before codegen starts"),
            Op::KernelLaunch { .. }
            | Op::CudaMalloc { .. }
            | Op::CudaMemcpy { .. }
            | Op::CudaFree { .. }
            | Op::CudaDeviceSynchronize => {
                unreachable!("check_module refuses these before codegen starts")
            }
            Op::Call { .. } => {
                unreachable!("check_module refuses function calls before codegen starts")
            }
        }
    }

    /// `tid.x`/`bdim.x`: reload a live scheduling value from its always-4-byte-meaningful home
    /// (both are plain `i32`s), truncating to whatever width this op's result type declares.
    fn lower_reload_dyn(&mut self, id: InstId, src_disp: i32, ty: Ty) {
        self.frame_addr(T0, src_disp);
        self.enc.lw(T0, 0, T0);
        self.store_result(id, T0, width_of(ty));
    }

    fn lower_bin(&mut self, id: InstId, op: BinOp, a: ValRef, b: ValRef, ty: Ty) {
        if is_float(ty) {
            // check_module already refused f64 arithmetic; only f32 reaches here.
            self.reload(a, A0, W::B4, false);
            self.reload(b, A1, W::B4, false);
            match op {
                BinOp::FAdd => self.enc.call("__sf_f32_add"),
                BinOp::FSub => {
                    self.enc.li32(T0, i32::MIN); // 0x80000000: the sign-bit mask
                    self.enc.alu_reg(AluOp::Xor, A1, A1, T0);
                    self.enc.call("__sf_f32_add");
                }
                BinOp::FMul => self.enc.call("__sf_f32_mul"),
                BinOp::FDiv => self.enc.call("__sf_f32_div"),
                BinOp::FRem => self.lower_frem32(),
                _ => unreachable!("float-typed Bin with a non-float BinOp"),
            }
            self.store_result(id, A0, W::B4);
            return;
        }

        if is_wide(ty) {
            self.lower_bin_i64(id, op, a, b);
            return;
        }

        let w = width_of(ty);
        match op {
            BinOp::Add | BinOp::Sub | BinOp::And | BinOp::Or | BinOp::Xor => {
                self.reload(a, T0, w, false);
                self.reload(b, T1, w, false);
                let aop = match op {
                    BinOp::Add => AluOp::Add,
                    BinOp::Sub => AluOp::Sub,
                    BinOp::And => AluOp::And,
                    BinOp::Or => AluOp::Or,
                    BinOp::Xor => AluOp::Xor,
                    _ => unreachable!(),
                };
                self.enc.alu_reg(aop, T0, T0, T1);
                self.store_result(id, T0, w);
            }
            BinOp::Mul => {
                self.reload(a, T0, w, false);
                self.reload(b, T1, w, false);
                self.enc.mul_reg(MulOp::Mul, T0, T0, T1);
                self.store_result(id, T0, w);
            }
            BinOp::Div | BinOp::Rem => {
                // Signed, always — matches the x86 oracle's own documented stance (BIR's
                // `Bin` carries no signed/unsigned distinction for these).
                self.reload(a, T0, w, true);
                self.reload(b, T1, w, true);
                let mop = if matches!(op, BinOp::Div) {
                    MulOp::Div
                } else {
                    MulOp::Rem
                };
                self.enc.mul_reg(mop, T0, T0, T1);
                self.store_result(id, T0, w);
            }
            BinOp::Shl | BinOp::Lshr | BinOp::Ashr => {
                let signed = matches!(op, BinOp::Ashr);
                self.reload(a, T0, w, signed);
                self.reload(b, T1, w, false);
                let aop = match op {
                    BinOp::Shl => AluOp::Sll,
                    BinOp::Lshr => AluOp::Srl,
                    BinOp::Ashr => AluOp::Sra,
                    _ => unreachable!(),
                };
                self.enc.alu_reg(aop, T0, T0, T1);
                self.store_result(id, T0, w);
            }
            _ => unreachable!("integer-typed Bin with a float BinOp"),
        }
    }

    /// `i64` add/sub/mul/and/or/xor via two 32-bit words per operand (see the module
    /// header — `shl`/`lshr`/`ashr`/`div`/`rem` on `i64` are refused before codegen starts).
    fn lower_bin_i64(&mut self, id: InstId, op: BinOp, a: ValRef, b: ValRef) {
        self.reload_wide(a, T0, T1); // T0=a_lo, T1=a_hi
        self.reload_wide(b, T2, T3); // T2=b_lo, T3=b_hi
        match op {
            BinOp::Add => {
                // carry = (sum_lo <u a_lo)
                self.enc.alu_reg(AluOp::Add, T4, T0, T2); // T4 = a_lo+b_lo
                self.enc.alu_reg(AluOp::Sltu, T5, T4, T0); // T5 = carry
                self.enc.alu_reg(AluOp::Add, T1, T1, T3); // hi = a_hi+b_hi
                self.enc.alu_reg(AluOp::Add, T1, T1, T5);
                self.store_result_wide(id, T4, T1);
            }
            BinOp::Sub => {
                // borrow = (a_lo <u b_lo)
                self.enc.alu_reg(AluOp::Sltu, T5, T0, T2); // T5 = borrow
                self.enc.alu_reg(AluOp::Sub, T4, T0, T2); // T4 = a_lo-b_lo
                self.enc.alu_reg(AluOp::Sub, T1, T1, T3);
                self.enc.alu_reg(AluOp::Sub, T1, T1, T5);
                self.store_result_wide(id, T4, T1);
            }
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let aop = match op {
                    BinOp::And => AluOp::And,
                    BinOp::Or => AluOp::Or,
                    BinOp::Xor => AluOp::Xor,
                    _ => unreachable!(),
                };
                self.enc.alu_reg(aop, T4, T0, T2);
                self.enc.alu_reg(aop, T5, T1, T3);
                self.store_result_wide(id, T4, T5);
            }
            BinOp::Mul => {
                // Keep only the low 64 bits of the full 128-bit product (this backend's own
                // documented stance for `i64` multiply, matching the x86 oracle's own
                // truncating-multiply convention): lo = a_lo*b_lo; hi = mulhu(a_lo,b_lo) +
                // a_lo*b_hi + a_hi*b_lo, every term mod 2^32 (plain 32-bit `add`/`mul`
                // wraparound is exactly this).
                self.enc.mul_reg(MulOp::Mulhu, T4, T0, T2); // T4 = mulhu(a_lo,b_lo)
                self.enc.mul_reg(MulOp::Mul, T5, T0, T3); // T5 = a_lo*b_hi (low 32)
                self.enc.alu_reg(AluOp::Add, T4, T4, T5);
                self.enc.mul_reg(MulOp::Mul, T5, T1, T2); // T5 = a_hi*b_lo (low 32)
                self.enc.alu_reg(AluOp::Add, T4, T4, T5); // T4 = hi
                self.enc.mul_reg(MulOp::Mul, T5, T0, T2); // T5 = lo = a_lo*b_lo
                self.store_result_wide(id, T5, T4);
            }
            _ => unreachable!("lower_bin_i64 called with an op check_module should have refused"),
        }
    }

    /// Software float remainder: `q = trunc(a/b); result = a - q*b` — identical structure to
    /// the x86 oracle's own `lower_frem`, composed from already-existing soft-float
    /// primitives rather than a dedicated routine. Entered with `a0=a`, `a1=b` (already
    /// reloaded by `lower_bin`); leaves the result in `a0`. Every value that must survive a
    /// `call` here goes through `Frame::scratch0`/`scratch1` (see `Frame`'s own doc comment),
    /// never a register.
    fn lower_frem32(&mut self) {
        let (s0, s1) = (self.frame.scratch0, self.frame.scratch1);
        self.stash(s0, A0); // scratch0 = a
        self.stash(s1, A1); // scratch1 = b
        self.enc.call("__sf_f32_div"); // a0 = a/b
        self.enc.call("__sf_f32_to_i32"); // a0 = trunc(a/b) as i32
        self.enc.call("__sf_i32_to_f32"); // a0 = float(trunc(a/b)), i.e. q
        self.unstash(s1, A1); // a1 = b
        self.enc.call("__sf_f32_mul"); // a0 = q*b
        self.stash(s1, A0); // scratch1 = q*b (b no longer needed, reuse its slot)
        self.unstash(s0, A0); // a0 = a
        self.unstash(s1, A1); // a1 = q*b
        self.enc.li32(T0, i32::MIN);
        self.enc.alu_reg(AluOp::Xor, A1, A1, T0); // a1 = -(q*b)
        self.enc.call("__sf_f32_add"); // a0 = a - q*b
    }

    fn lower_icmp(&mut self, id: InstId, pred: ICmpPred, cty: Ty, a: ValRef, b: ValRef) {
        if is_wide(cty) {
            self.lower_icmp_i64(id, pred, a, b);
            return;
        }
        let signed = matches!(
            pred,
            ICmpPred::Slt | ICmpPred::Sle | ICmpPred::Sgt | ICmpPred::Sge
        );
        let w = width_of(cty);
        self.reload(a, T0, w, signed);
        self.reload(b, T1, w, signed);
        match pred {
            ICmpPred::Eq => {
                self.enc.alu_reg(AluOp::Xor, T0, T0, T1);
                self.enc.sltiu(T0, T0, 1);
            }
            ICmpPred::Ne => {
                self.enc.alu_reg(AluOp::Xor, T0, T0, T1);
                self.enc.alu_reg(AluOp::Sltu, T0, ZERO, T0);
            }
            ICmpPred::Slt | ICmpPred::Ult => {
                let aop = if signed { AluOp::Slt } else { AluOp::Sltu };
                self.enc.alu_reg(aop, T0, T0, T1);
            }
            ICmpPred::Sgt | ICmpPred::Ugt => {
                let aop = if signed { AluOp::Slt } else { AluOp::Sltu };
                self.enc.alu_reg(aop, T0, T1, T0);
            }
            ICmpPred::Sle | ICmpPred::Ule => {
                let aop = if signed { AluOp::Slt } else { AluOp::Sltu };
                self.enc.alu_reg(aop, T0, T1, T0);
                self.enc.xori(T0, T0, 1);
            }
            ICmpPred::Sge | ICmpPred::Uge => {
                let aop = if signed { AluOp::Slt } else { AluOp::Sltu };
                self.enc.alu_reg(aop, T0, T0, T1);
                self.enc.xori(T0, T0, 1);
            }
        }
        self.store_result(id, T0, W::B1);
    }

    /// `i64` compare: high words first (signed iff the predicate is signed; low words are
    /// always compared unsigned, since they carry no sign of their own).
    /// `i64` compare: derive every predicate from two base 0/1 facts — `eq` (`a==b`, both
    /// words) and `lt` (`a<b` at the predicate's own signedness: high words compared first,
    /// low words unsigned, only when the high words are equal — the standard multi-word
    /// lexicographic compare). Every predicate is then a fixed function of `(lt, eq)`:
    /// `slt/ult = lt`; `sle/ule = lt || eq`; `sgt/ugt = !lt && !eq`; `sge/uge = !lt` (if not
    /// less, `a` is either equal or greater, which is exactly `>=`); `eq`/`ne` directly.
    fn lower_icmp_i64(&mut self, id: InstId, pred: ICmpPred, a: ValRef, b: ValRef) {
        let signed = matches!(
            pred,
            ICmpPred::Slt | ICmpPred::Sle | ICmpPred::Sgt | ICmpPred::Sge
        );
        self.reload_wide(a, T0, T1); // T0=a_lo, T1=a_hi
        self.reload_wide(b, T2, T3); // T2=b_lo, T3=b_hi

        // eq = (a_hi==b_hi) && (a_lo==b_lo), via a combined xor-or-zero test.
        self.enc.alu_reg(AluOp::Xor, T4, T0, T2);
        self.enc.alu_reg(AluOp::Xor, T5, T1, T3);
        self.enc.alu_reg(AluOp::Or, T4, T4, T5);
        self.enc.sltiu(T4, T4, 1); // T4 = eq

        if matches!(pred, ICmpPred::Eq) {
            self.store_result(id, T4, W::B1);
            return;
        }
        if matches!(pred, ICmpPred::Ne) {
            self.enc.xori(T4, T4, 1);
            self.store_result(id, T4, W::B1);
            return;
        }

        // lt = hi_eq ? (a_lo <u b_lo) : (a_hi < b_hi); T6 = hi_eq (recomputed, T4 already
        // holds the *full* eq above and must not be clobbered before the final combine).
        self.enc.alu_reg(AluOp::Xor, T6, T1, T3);
        self.enc.sltiu(T6, T6, 1); // T6 = hi_eq
        let hi_lt_op = if signed { AluOp::Slt } else { AluOp::Sltu };
        self.enc.alu_reg(hi_lt_op, T5, T1, T3); // T5 = a_hi < b_hi
        let lo_lt_label = self.fresh_label("icmp64_lo_lt");
        let combine_label = self.fresh_label("icmp64_combine");
        self.enc.branch(BCond::Ne, T6, ZERO, &lo_lt_label);
        self.enc.jump(&combine_label); // hi differs: T5 already holds `lt`
        self.enc.label(&lo_lt_label);
        self.enc.alu_reg(AluOp::Sltu, T5, T0, T2); // hi equal: lt = a_lo <u b_lo
        self.enc.label(&combine_label);
        // T5 = lt, T4 = eq
        match pred {
            ICmpPred::Slt | ICmpPred::Ult => self.store_result(id, T5, W::B1),
            ICmpPred::Sle | ICmpPred::Ule => {
                self.enc.alu_reg(AluOp::Or, T5, T5, T4);
                self.store_result(id, T5, W::B1);
            }
            ICmpPred::Sgt | ICmpPred::Ugt => {
                self.enc.alu_reg(AluOp::Or, T5, T5, T4);
                self.enc.xori(T5, T5, 1);
                self.store_result(id, T5, W::B1);
            }
            ICmpPred::Sge | ICmpPred::Uge => {
                self.enc.xori(T5, T5, 1);
                self.store_result(id, T5, W::B1);
            }
            ICmpPred::Eq | ICmpPred::Ne => unreachable!("handled above"),
        }
    }

    /// `a0`/`a1` reload, `__sf_f32_cmp` call, then the ordering code (`-2`/`-1`/`0`/`1`) is
    /// combined per predicate exactly as `basalt-x86/src/oracle.rs`'s own `lower_fcmp` combines
    /// `ucomiss`'s flag bits — a fixed small integer comparison against the returned code
    /// rather than a flags-register trick, since RV32 has no flags register at all.
    fn lower_fcmp(&mut self, id: InstId, pred: FCmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let _ = cty; // check_module already refused f64; only f32 reaches here.
        self.reload(a, A0, W::B4, false);
        self.reload(b, A1, W::B4, false);
        self.enc.call("__sf_f32_cmp");
        match pred {
            FCmpPred::Oeq => self.eq_const(T0, A0, 0),
            FCmpPred::Olt => self.eq_const(T0, A0, -1),
            FCmpPred::Ogt => self.eq_const(T0, A0, 1),
            FCmpPred::Uno => self.eq_const(T0, A0, -2),
            FCmpPred::Ord => {
                self.eq_const(T0, A0, -2);
                self.enc.xori(T0, T0, 1);
            }
            FCmpPred::One => {
                self.eq_const(T0, A0, 0);
                self.eq_const(T1, A0, -2);
                self.enc.alu_reg(AluOp::Or, T0, T0, T1);
                self.enc.xori(T0, T0, 1);
            }
            FCmpPred::Ole => {
                self.eq_const(T0, A0, -1);
                self.eq_const(T1, A0, 0);
                self.enc.alu_reg(AluOp::Or, T0, T0, T1);
            }
            FCmpPred::Oge => {
                self.eq_const(T0, A0, 1);
                self.eq_const(T1, A0, 0);
                self.enc.alu_reg(AluOp::Or, T0, T0, T1);
            }
        }
        self.store_result(id, T0, W::B1);
    }

    /// `dst = (src == target)` as a 0/1 value, for a small compile-time `target` (every call
    /// site here passes one of `-2,-1,0,1`, well within `addi`'s 12-bit range): `src - target
    /// == 0` iff equal, and `sltiu dst, x, 1` is `1` iff `x == 0` (treating `x` as unsigned —
    /// exactly what is wanted here, since `0` is the only value `<u 1` regardless of `x`'s own
    /// sign).
    fn eq_const(&mut self, dst: u8, src: u8, target: i32) {
        self.enc.addi(dst, src, -target);
        self.enc.sltiu(dst, dst, 1);
    }

    fn lower_cast(&mut self, id: InstId, cop: CastOp, sty: Ty, v: ValRef, dty: Ty) {
        match cop {
            CastOp::Trunc => {
                // Reading only the low `width_of(dty)` bytes of the (wider) source's own slot
                // is exactly truncation, mirroring the x86 oracle's identical reasoning.
                let dw = width_of(dty);
                self.reload(v, T0, dw, false);
                self.store_result(id, T0, dw);
            }
            CastOp::Zext => {
                if is_wide(dty) {
                    let sw = width_of(sty);
                    self.reload(v, T0, sw, false);
                    self.store_result_wide(id, T0, ZERO);
                } else {
                    let sw = width_of(sty);
                    let dw = width_of(dty);
                    self.reload(v, T0, sw, false);
                    self.store_result(id, T0, dw);
                }
            }
            CastOp::Sext => {
                if is_wide(dty) {
                    let sw = width_of(sty);
                    self.reload(v, T0, sw, true);
                    self.enc.srai(T1, T0, 31); // T1 = all-1s or all-0s per T0's sign
                    self.store_result_wide(id, T0, T1);
                } else {
                    let sw = width_of(sty);
                    let dw = width_of(dty);
                    self.reload(v, T0, sw, true);
                    self.store_result(id, T0, dw);
                }
            }
            CastOp::Bitcast => {
                // No separate float register class on RV32 (see the module header): a
                // same-width reinterpret is always a plain copy, never a real bit
                // manipulation.
                if is_wide(sty) || is_wide(dty) {
                    self.reload_wide(v, T0, T1);
                    self.store_result_wide(id, T0, T1);
                } else {
                    let w = width_of(dty);
                    self.reload(v, T0, w, false);
                    self.store_result(id, T0, w);
                }
            }
            CastOp::FpToSi => {
                self.reload(v, A0, W::B4, false);
                self.enc.call("__sf_f32_to_i32");
                self.store_result(id, A0, width_of(dty));
            }
            CastOp::FpToUi => {
                self.reload(v, A0, W::B4, false);
                self.enc.call("__sf_f32_to_u32");
                self.store_result(id, A0, width_of(dty));
            }
            CastOp::SiToFp => {
                let sw = width_of(sty);
                self.reload(v, A0, sw, true);
                self.enc.call("__sf_i32_to_f32");
                self.store_result(id, A0, W::B4);
            }
            CastOp::UiToFp => {
                let sw = width_of(sty);
                self.reload(v, A0, sw, false);
                self.enc.call("__sf_u32_to_f32");
                self.store_result(id, A0, W::B4);
            }
            CastOp::FpExt | CastOp::FpTrunc => {
                unreachable!("check_module refuses f64 conversions before codegen starts")
            }
        }
    }

    /// Every address space this backend touches is, by the time a value reaches here, a
    /// genuine 32-bit address (see the module header) — one path handles `load` for every
    /// space alike.
    fn lower_load(&mut self, id: InstId, ptr: ValRef, ty: Ty) {
        self.reload(ptr, T4, W::B4, false);
        if is_wide(ty) {
            self.enc.lw(T0, 0, T4);
            self.enc.lw(T1, 4, T4);
            self.store_result_wide(id, T0, T1);
        } else {
            let w = width_of(ty);
            match w {
                W::B1 => self.enc.lbu(T0, 0, T4),
                W::B2 => self.enc.lhu(T0, 0, T4),
                W::B4 => self.enc.lw(T0, 0, T4),
            }
            self.store_result(id, T0, w);
        }
    }

    /// The value operand is reloaded *before* the address is computed into `T4` (`reload`/
    /// `reload_wide` use `T4` as their own internal addressing scratch — computing the target
    /// address first and only then reloading the value would silently clobber it). This
    /// ordering rule matters everywhere in this file a computed address must survive a
    /// subsequent `reload` call; `lower_load` above has no such concern since nothing follows
    /// its own address computation except the immediate use of `T4`.
    fn lower_store(&mut self, ptr: ValRef, val: ValRef, ty: Ty) {
        if is_wide(ty) {
            self.reload_wide(val, T0, T1);
            self.reload(ptr, T4, W::B4, false);
            self.enc.sw(T0, 0, T4);
            self.enc.sw(T1, 4, T4);
        } else {
            let w = width_of(ty);
            self.reload(val, T0, w, false);
            self.reload(ptr, T4, W::B4, false);
            match w {
                W::B1 => self.enc.sb(T0, 0, T4),
                W::B2 => self.enc.sh(T0, 0, T4),
                W::B4 => self.enc.sw(T0, 0, T4),
            }
        }
    }

    /// Ordinary (non-atomic-instruction) load-compute-store: correct here because exactly one
    /// thread ever executes at a time (see the module header), returning the pre-modification
    /// value to match CUDA's atomic-RMW-returns-old semantics — identical reasoning to the x86
    /// oracle's own `lower_atomic`. `f64` atomics are refused before codegen starts (see
    /// `check_module`); `i64` atomics reuse `lower_bin_i64`'s primitives directly since BIR's
    /// `AtomicOp` has no shift/div/rem variant to worry about.
    fn lower_atomic(&mut self, id: InstId, op: AtomicOp, ptr: ValRef, val: ValRef, ty: Ty) {
        if is_float(ty) {
            self.lower_atomic_f32(id, op, ptr, val);
            return;
        }
        if is_wide(ty) {
            self.lower_atomic_i64(id, op, ptr, val);
            return;
        }
        let w = width_of(ty);
        let signed = matches!(op, AtomicOp::Min | AtomicOp::Max);
        self.reload(val, T0, w, signed); // T0 = val
        self.reload(ptr, T4, W::B4, false); // T4 = address (val already safe in T0)
        match w {
            W::B1 => self.enc.lb(T1, 0, T4),
            W::B2 => self.enc.lh(T1, 0, T4),
            W::B4 => self.enc.lw(T1, 0, T4),
        }; // T1 = old (sign per this op's own signedness, harmless for the bitwise ops)
        self.enc.mv(T5, T1); // T5 = old, kept for the return value
        match op {
            AtomicOp::Add => self.enc.alu_reg(AluOp::Add, T1, T1, T0),
            AtomicOp::Sub => self.enc.alu_reg(AluOp::Sub, T1, T1, T0),
            AtomicOp::And => self.enc.alu_reg(AluOp::And, T1, T1, T0),
            AtomicOp::Or => self.enc.alu_reg(AluOp::Or, T1, T1, T0),
            AtomicOp::Xor => self.enc.alu_reg(AluOp::Xor, T1, T1, T0),
            AtomicOp::Exch => self.enc.mv(T1, T0),
            AtomicOp::Min | AtomicOp::Max => {
                let aop = if signed { AluOp::Slt } else { AluOp::Sltu };
                // skip_cc: keep `old` (T1) when old<=val (Min) or old>=val (Max).
                let skip = self.fresh_label("atomic_minmax_skip");
                if matches!(op, AtomicOp::Min) {
                    self.enc.alu_reg(aop, T2, T0, T1); // T2 = val<old
                    self.enc.branch(BCond::Eq, T2, ZERO, &skip); // !(val<old) => old<=val: keep old
                } else {
                    self.enc.alu_reg(aop, T2, T1, T0); // T2 = old<val
                    self.enc.branch(BCond::Eq, T2, ZERO, &skip); // !(old<val) => old>=val: keep old
                }
                self.enc.mv(T1, T0);
                self.enc.label(&skip);
            }
        }
        match w {
            W::B1 => self.enc.sb(T1, 0, T4),
            W::B2 => self.enc.sh(T1, 0, T4),
            W::B4 => self.enc.sw(T1, 0, T4),
        }
        self.store_result(id, T5, w);
    }

    fn lower_atomic_i64(&mut self, id: InstId, op: AtomicOp, ptr: ValRef, val: ValRef) {
        self.reload_wide(val, T0, T1); // T0=val_lo, T1=val_hi
        self.reload(ptr, T4, W::B4, false); // T4 = address
        self.enc.lw(T2, 0, T4);
        self.enc.lw(T3, 4, T4); // T2:T3 = old
                                // No soft-float call happens on this path, so nothing here strictly needs to survive
                                // a `call` — but `T2`/`T3` themselves get overwritten by the arithmetic below (they
                                // become the *new* value), so `old` is parked here simply to have a place to retrieve
                                // it back from for the return value once `T2`/`T3` no longer hold it.
        self.stash(self.frame.scratch0, T2);
        self.stash(self.frame.scratch1, T3);
        match op {
            AtomicOp::Add => {
                self.enc.alu_reg(AluOp::Add, T5, T2, T0);
                self.enc.alu_reg(AluOp::Sltu, T6, T5, T2);
                self.enc.alu_reg(AluOp::Add, T3, T3, T1);
                self.enc.alu_reg(AluOp::Add, T3, T3, T6);
                self.enc.mv(T2, T5);
            }
            AtomicOp::Sub => {
                self.enc.alu_reg(AluOp::Sltu, T6, T2, T0);
                self.enc.alu_reg(AluOp::Sub, T5, T2, T0);
                self.enc.alu_reg(AluOp::Sub, T3, T3, T1);
                self.enc.alu_reg(AluOp::Sub, T3, T3, T6);
                self.enc.mv(T2, T5);
            }
            AtomicOp::And => {
                self.enc.alu_reg(AluOp::And, T2, T2, T0);
                self.enc.alu_reg(AluOp::And, T3, T3, T1);
            }
            AtomicOp::Or => {
                self.enc.alu_reg(AluOp::Or, T2, T2, T0);
                self.enc.alu_reg(AluOp::Or, T3, T3, T1);
            }
            AtomicOp::Xor => {
                self.enc.alu_reg(AluOp::Xor, T2, T2, T0);
                self.enc.alu_reg(AluOp::Xor, T3, T3, T1);
            }
            AtomicOp::Exch => {
                self.enc.mv(T2, T0);
                self.enc.mv(T3, T1);
            }
            AtomicOp::Min | AtomicOp::Max => {
                // Signed 64-bit compare (hi word signed, lo word unsigned), matching this
                // backend's `lower_icmp_i64` structure but inlined for the two registers at
                // hand: old = T2:T3, val = T0:T1.
                let old_lt_val_label = self.fresh_label("atomic64_minmax");
                let done_label = self.fresh_label("atomic64_minmax_done");
                self.enc.alu_reg(AluOp::Slt, T5, T3, T1); // hi_old < hi_val (signed)
                self.enc.alu_reg(AluOp::Xor, T6, T3, T1);
                self.enc.branch(BCond::Ne, T6, ZERO, &old_lt_val_label);
                self.enc.alu_reg(AluOp::Sltu, T5, T2, T0); // hi equal: lo_old <u lo_val
                self.enc.label(&old_lt_val_label);
                // T5 = (old < val). Min keeps old when !(val<old) i.e. old<=val: recompute
                // symmetric "val<old" the same way to decide, per direction.
                if matches!(op, AtomicOp::Min) {
                    self.enc.branch(BCond::Ne, T5, ZERO, &done_label); // old<val: keep old
                } else {
                    self.enc.branch(BCond::Eq, T5, ZERO, &done_label); // !(old<val): keep old
                }
                self.enc.mv(T2, T0);
                self.enc.mv(T3, T1);
                self.enc.label(&done_label);
            }
        }
        self.enc.sw(T2, 0, T4);
        self.enc.sw(T3, 4, T4);
        self.unstash(self.frame.scratch0, T0);
        self.unstash(self.frame.scratch1, T1);
        self.store_result_wide(id, T0, T1);
    }

    /// `f32` atomics: `Add`/`Sub` route through `__sf_f32_add`; `Min`/`Max` route through
    /// `__sf_f32_cmp`; `Exch`/`And`/`Or`/`Xor` operate on the raw bit pattern directly (no
    /// soft-float call needed — well-defined, if unusual for a real kernel, matching the x86
    /// oracle's identical stance on bitwise atomics against a float type). The address, the
    /// pre-modification value, and (for `Add`/`Sub`/`Min`/`Max`) `val` itself all must survive
    /// whichever soft-float call runs, so all three go through `Frame::scratch0/1/2` — no
    /// register is ever assumed to survive a `call` (see `Frame`'s own doc comment).
    fn lower_atomic_f32(&mut self, id: InstId, op: AtomicOp, ptr: ValRef, val: ValRef) {
        self.reload(val, T0, W::B4, false); // T0 = val bits
        self.reload(ptr, T4, W::B4, false); // T4 = address (val already safe in T0)
        self.enc.lw(T5, 0, T4); // T5 = old bits
        let (s_addr, s_old, s_val) = (
            self.frame.scratch0,
            self.frame.scratch1,
            self.frame.scratch2,
        );
        self.stash(s_addr, T4);
        self.stash(s_old, T5);
        self.stash(s_val, T0);
        match op {
            AtomicOp::Add | AtomicOp::Sub => {
                self.enc.mv(A0, T5);
                self.enc.mv(A1, T0);
                if matches!(op, AtomicOp::Sub) {
                    self.enc.li32(T1, i32::MIN);
                    self.enc.alu_reg(AluOp::Xor, A1, A1, T1);
                }
                self.enc.call("__sf_f32_add"); // a0 = new
            }
            AtomicOp::Exch => self.enc.mv(A0, T0),
            AtomicOp::And => self.enc.alu_reg(AluOp::And, A0, T5, T0),
            AtomicOp::Or => self.enc.alu_reg(AluOp::Or, A0, T5, T0),
            AtomicOp::Xor => self.enc.alu_reg(AluOp::Xor, A0, T5, T0),
            AtomicOp::Min | AtomicOp::Max => {
                self.enc.mv(A0, T5);
                self.enc.mv(A1, T0);
                self.enc.call("__sf_f32_cmp"); // a0 = code: old cmp val
                let keep_old = self.fresh_label("atomic_f32_minmax_keep_old");
                let done = self.fresh_label("atomic_f32_minmax_done");
                // Min keeps old when code<=0 (old<=val); Max keeps old when code>=0.
                if matches!(op, AtomicOp::Min) {
                    self.enc.branch(BCond::Ge, ZERO, A0, &keep_old); // 0>=code i.e. code<=0
                } else {
                    self.enc.branch(BCond::Ge, A0, ZERO, &keep_old); // code>=0
                }
                self.unstash(s_val, A0); // new = val
                self.enc.jump(&done);
                self.enc.label(&keep_old);
                self.unstash(s_old, A0); // new = old (unchanged)
                self.enc.label(&done);
            }
        }
        // a0 = the new value to store; every register used above except a0 may have been
        // clobbered by a call, so both the address and the return value are reloaded fresh
        // from memory here rather than trusted from any earlier register copy.
        self.unstash(s_addr, T4);
        self.enc.sw(A0, 0, T4);
        self.unstash(s_old, T5);
        self.store_result(id, T5, W::B4);
    }

    /// `atomicCAS` compares the raw bit pattern regardless of `ty` (real hardware CAS is
    /// always integer, even for a float/double CAS in CUDA) — identical reasoning to the x86
    /// oracle's own `lower_atomic_cas`. Works uniformly for wide (`i64`/`f64`) and narrow
    /// types alike.
    fn lower_atomic_cas(&mut self, id: InstId, ptr: ValRef, cmp: ValRef, newv: ValRef, ty: Ty) {
        if is_wide(ty) {
            self.reload_wide(cmp, T0, T1);
            self.reload_wide(newv, T2, T3);
            self.reload(ptr, T4, W::B4, false);
            self.enc.lw(T5, 0, T4);
            self.enc.lw(T6, 4, T4);
            let mismatch = self.fresh_label("cas64_mismatch");
            self.enc.alu_reg(AluOp::Xor, A2, T5, T0);
            self.enc.alu_reg(AluOp::Xor, A3, T6, T1);
            self.enc.alu_reg(AluOp::Or, A2, A2, A3);
            self.enc.branch(BCond::Ne, A2, ZERO, &mismatch);
            self.enc.sw(T2, 0, T4);
            self.enc.sw(T3, 4, T4);
            self.enc.label(&mismatch);
            self.store_result_wide(id, T5, T6);
        } else {
            let w = width_of(ty);
            self.reload(cmp, T0, w, false);
            self.reload(newv, T1, w, false);
            self.reload(ptr, T4, W::B4, false);
            match w {
                W::B1 => self.enc.lbu(T2, 0, T4),
                W::B2 => self.enc.lhu(T2, 0, T4),
                W::B4 => self.enc.lw(T2, 0, T4),
            }
            let mismatch = self.fresh_label("cas_mismatch");
            self.enc.branch(BCond::Ne, T2, T0, &mismatch);
            match w {
                W::B1 => self.enc.sb(T1, 0, T4),
                W::B2 => self.enc.sh(T1, 0, T4),
                W::B4 => self.enc.sw(T1, 0, T4),
            }
            self.enc.label(&mismatch);
            self.store_result(id, T2, w);
        }
    }

    fn emit_phi_copies(&mut self, from: u32, to: u32) {
        let Some(copies) = self.phi_copies.get(&(from, to)).cloned() else {
            return;
        };
        for (dest_disp, val, ty) in copies {
            if is_wide(ty) {
                self.reload_wide(val, T0, T1);
                self.frame_addr(T4, dest_disp);
                self.enc.sw(T0, 0, T4);
                self.enc.sw(T1, 4, T4);
            } else {
                let w = width_of(ty);
                self.reload(val, T0, w, false);
                self.frame_addr(T4, dest_disp);
                match w {
                    W::B1 => self.enc.sb(T0, 0, T4),
                    W::B2 => self.enc.sh(T0, 0, T4),
                    W::B4 => self.enc.sw(T0, 0, T4),
                }
            }
        }
    }

    fn lower_term(&mut self, from_block: u32, term: &Term) {
        match term {
            Term::Br(target) => {
                self.emit_phi_copies(from_block, target.0);
                self.enc.jump(&block_label(target.0));
            }
            Term::CondBr(cond, t, f) => {
                self.reload(*cond, T0, W::B1, false);
                let false_prep = self.fresh_label("condbr_false");
                self.enc.branch(BCond::Eq, T0, ZERO, &false_prep);
                self.emit_phi_copies(from_block, t.0);
                self.enc.jump(&block_label(t.0));
                self.enc.label(&false_prep);
                self.emit_phi_copies(from_block, f.0);
                self.enc.jump(&block_label(f.0));
            }
            Term::Switch(scrut, default, cases) => {
                let ty = self.valref_ty(*scrut);
                let w = width_of(ty);
                self.reload(*scrut, T0, w, false);
                for &(case_val, target) in cases {
                    self.enc.li32(T1, case_val as i32);
                    let skip = self.fresh_label("switch_skip");
                    self.enc.branch(BCond::Ne, T0, T1, &skip);
                    self.emit_phi_copies(from_block, target.0);
                    self.enc.jump(&block_label(target.0));
                    self.enc.label(&skip);
                }
                self.emit_phi_copies(from_block, default.0);
                self.enc.jump(&block_label(default.0));
            }
            Term::Ret(v) => {
                if let Some(val) = v {
                    let rty = self.f.ret;
                    let disp = self
                        .frame
                        .retval_home
                        .expect("non-void Ret always has a retval home");
                    if is_wide(rty) {
                        self.reload_wide(*val, T0, T1);
                        self.frame_addr(T4, disp);
                        self.enc.sw(T0, 0, T4);
                        self.enc.sw(T1, 4, T4);
                    } else {
                        let w = width_of(rty);
                        self.reload(*val, T0, w, false);
                        self.frame_addr(T4, disp);
                        match w {
                            W::B1 => self.enc.sb(T0, 0, T4),
                            W::B2 => self.enc.sh(T0, 0, T4),
                            W::B4 => self.enc.sw(T0, 0, T4),
                        }
                    }
                }
                // Every thread must still advance the loop, not actually return — jump to the
                // loop's own increment step instead of emitting a real `ret`, matching the x86
                // oracle's identical reasoning.
                self.enc.jump("__loop_incr");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{Block, BlockId, Inst, LaunchBounds};
    use object::read::{Object as ReadObject, ObjectSection, ObjectSymbol};

    fn wrap(f: Function) -> Module {
        Module {
            funcs: vec![f],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    fn parses_as_elf_with_symbol<'a>(bytes: &'a [u8], symbol: &str) -> object::read::File<'a> {
        let file = object::read::File::parse(bytes).expect("parses as an object file");
        assert_eq!(file.format(), object::BinaryFormat::Elf);
        assert_eq!(file.architecture(), object::Architecture::Riscv32);
        let text = file
            .section_by_name(".text")
            .expect(".text section present");
        let data = text.data().unwrap();
        assert!(!data.is_empty(), ".text must not be empty");
        assert_eq!(data.len() % 4, 0, "every RV32 instruction is 4 bytes");
        let sym = file
            .symbols()
            .find(|s| s.name() == Ok(symbol))
            .unwrap_or_else(|| panic!("symbol `{symbol}` present"));
        assert_eq!(sym.size(), data.len() as u64);
        file
    }

    fn func_ret_const() -> Function {
        Function {
            is_kernel: true,
            name: "ret_const".into(),
            params: vec![],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::ConstInt(42),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    fn func_add_i32() -> Function {
        Function {
            is_kernel: true,
            name: "add_i32".into(),
            params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    fn func_max_i32() -> Function {
        Function {
            is_kernel: true,
            name: "max_i32".into(),
            params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I1),
                op: Op::ICmp(
                    ICmpPred::Sgt,
                    Ty::Scalar(Scalar::I32),
                    ValRef::Param(0),
                    ValRef::Param(1),
                ),
            }],
            blocks: vec![
                Block {
                    insts: vec![InstId(0)],
                    term: Term::CondBr(ValRef::Val(InstId(0)), BlockId(1), BlockId(2)),
                },
                Block {
                    insts: vec![],
                    term: Term::Ret(Some(ValRef::Param(0))),
                },
                Block {
                    insts: vec![],
                    term: Term::Ret(Some(ValRef::Param(1))),
                },
            ],
        }
    }

    /// `func @vector_add_like(ptr.global, ptr.global, ptr.global, i32) -> void`, the RV32
    /// analogue of `basalt-x86/src/oracle.rs`'s `func_write_idx`: exercises `tid.x`,
    /// pointer arithmetic, and a `global` store, all inside the per-thread loop.
    fn func_write_idx() -> Function {
        Function {
            is_kernel: true,
            name: "write_idx".into(),
            params: vec![Ty::Ptr(AddrSpace::Global)],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::TidX,
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::ConstInt(4),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::Bin(BinOp::Mul, ValRef::Val(InstId(0)), ValRef::Val(InstId(1))),
                },
                Inst {
                    ty: Ty::Ptr(AddrSpace::Global),
                    op: Op::Cast(
                        CastOp::Bitcast,
                        Ty::Scalar(Scalar::I32),
                        ValRef::Val(InstId(2)),
                    ),
                },
                Inst {
                    ty: Ty::Ptr(AddrSpace::Global),
                    op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Val(InstId(3))),
                },
                Inst {
                    ty: Ty::Void,
                    op: Op::Store {
                        ptr: ValRef::Val(InstId(4)),
                        val: ValRef::Val(InstId(0)),
                        ty: Ty::Scalar(Scalar::I32),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                },
            ],
            blocks: vec![Block {
                insts: (0..6).map(InstId).collect(),
                term: Term::Ret(None),
            }],
        }
    }

    /// `func @vadd_f32(ptr.global x3, i32) -> void`: `c[i] = a[i] + b[i]`, the actual
    /// `tests/kernels/vector_add.cu` shape — exercises `f32` load/add/store end to end
    /// through the soft-float runtime.
    fn func_vector_add() -> Function {
        let ptr_g = Ty::Ptr(AddrSpace::Global);
        let i32t = Ty::Scalar(Scalar::I32);
        let f32t = Ty::Scalar(Scalar::F32);
        Function {
            is_kernel: true,
            name: "vector_add".into(),
            params: vec![ptr_g, ptr_g, ptr_g, i32t],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: i32t,
                    op: Op::TidX,
                }, // 0
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(4),
                }, // 1
                Inst {
                    ty: i32t,
                    op: Op::Bin(BinOp::Mul, ValRef::Val(InstId(0)), ValRef::Val(InstId(1))),
                }, // 2: byte offset
                Inst {
                    ty: ptr_g,
                    op: Op::Cast(CastOp::Bitcast, i32t, ValRef::Val(InstId(2))),
                }, // 3
                Inst {
                    ty: ptr_g,
                    op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Val(InstId(3))),
                }, // 4: &a[i]
                Inst {
                    ty: ptr_g,
                    op: Op::Bin(BinOp::Add, ValRef::Param(1), ValRef::Val(InstId(3))),
                }, // 5: &b[i]
                Inst {
                    ty: ptr_g,
                    op: Op::Bin(BinOp::Add, ValRef::Param(2), ValRef::Val(InstId(3))),
                }, // 6: &c[i]
                Inst {
                    ty: f32t,
                    op: Op::Load {
                        ptr: ValRef::Val(InstId(4)),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                }, // 7: a[i]
                Inst {
                    ty: f32t,
                    op: Op::Load {
                        ptr: ValRef::Val(InstId(5)),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                }, // 8: b[i]
                Inst {
                    ty: f32t,
                    op: Op::Bin(BinOp::FAdd, ValRef::Val(InstId(7)), ValRef::Val(InstId(8))),
                }, // 9
                Inst {
                    ty: Ty::Void,
                    op: Op::Store {
                        ptr: ValRef::Val(InstId(6)),
                        val: ValRef::Val(InstId(9)),
                        ty: f32t,
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                }, // 10
            ],
            blocks: vec![Block {
                insts: (0..11).map(InstId).collect(),
                term: Term::Ret(None),
            }],
        }
    }

    fn func_f64_scope_cut() -> Function {
        Function {
            is_kernel: true,
            name: "f64_add".into(),
            params: vec![Ty::Scalar(Scalar::F64), Ty::Scalar(Scalar::F64)],
            ret: Ty::Scalar(Scalar::F64),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::F64),
                op: Op::Bin(BinOp::FAdd, ValRef::Param(0), ValRef::Param(1)),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    #[test]
    fn supports_a_module_using_only_implemented_ops() {
        assert_eq!(Rv32.supports(&wrap(func_ret_const())), Support::Supported);
        assert_eq!(Rv32.supports(&wrap(func_add_i32())), Support::Supported);
        assert_eq!(Rv32.supports(&wrap(func_max_i32())), Support::Supported);
        assert_eq!(Rv32.supports(&wrap(func_write_idx())), Support::Supported);
        assert_eq!(Rv32.supports(&wrap(func_vector_add())), Support::Supported);
    }

    #[test]
    fn refuses_f64_arithmetic_with_e091() {
        assert_eq!(
            Rv32.supports(&wrap(func_f64_scope_cut())),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_mma_with_e099() {
        let f = Function {
            is_kernel: true,
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
                    m: 2,
                    n: 2,
                    k: 2,
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
            Rv32.supports(&wrap(f)),
            Support::Unsupported(ECode::MatrixPathUnsupported)
        );
    }

    #[test]
    fn refuses_shuffle_with_e090() {
        let f = Function {
            is_kernel: true,
            name: "shuf".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Shuffle(
                    basalt_bir::ShuffleKind::Idx,
                    ValRef::Param(0),
                    ValRef::Param(0),
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            Rv32.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    /// P13-T1b's kernel-launch/CUDA-Runtime-API ops are sema-only today (see
    /// `basalt_bir::Op::KernelLaunch`'s own doc comment) — every backend refuses them cleanly.
    #[test]
    fn refuses_kernel_launch_and_cuda_runtime_api_ops_with_e090() {
        let f = Function {
            is_kernel: true,
            name: "launch_stub".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Void,
                op: Op::CudaDeviceSynchronize,
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            Rv32.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    /// `Op::Call` (P13-T-calls-i) has no lowering in this backend yet — refuse cleanly rather
    /// than falling through to the scalar per-op emitters, which have no case for it.
    #[test]
    fn refuses_function_call_with_e090() {
        let f = Function {
            is_kernel: true,
            name: "caller".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Call {
                    func: "callee".into(),
                    args: vec![ValRef::Param(0)],
                },
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        };
        assert_eq!(
            Rv32.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    #[test]
    fn refuses_multi_function_module_with_e093() {
        let module = Module {
            funcs: vec![func_ret_const(), func_add_i32()],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        };
        assert_eq!(
            Rv32.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_non_kernel_function_with_e093() {
        let mut f = func_ret_const();
        f.is_kernel = false;
        assert_eq!(
            Rv32.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_too_many_integer_params_with_e093() {
        // 8 i32 params leaves no register for the trailing nthreads argument.
        let f = Function {
            is_kernel: true,
            name: "toomany".into(),
            params: vec![Ty::Scalar(Scalar::I32); 8],
            ret: Ty::Void,
            insts: vec![],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            Rv32.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn emits_valid_elf_for_ret_const() {
        let artifact = Rv32
            .emit(&wrap(func_ret_const()), &EmitOpts::default())
            .expect("emit succeeds");
        assert_eq!(artifact.kind, ArtifactKind::Object);
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "ret_const");
    }

    #[test]
    fn emits_valid_elf_for_add_i32() {
        let artifact = Rv32
            .emit(&wrap(func_add_i32()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "add_i32");
    }

    #[test]
    fn emits_valid_elf_for_condbr() {
        let artifact = Rv32
            .emit(&wrap(func_max_i32()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "max_i32");
    }

    #[test]
    fn emits_valid_elf_for_thread_index_loop() {
        let artifact = Rv32
            .emit(&wrap(func_write_idx()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "write_idx");
    }

    #[test]
    fn emits_valid_elf_for_vector_add() {
        let artifact = Rv32
            .emit(&wrap(func_vector_add()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "vector_add");
    }

    #[test]
    fn emit_refuses_what_supports_refuses() {
        let f = Function {
            is_kernel: true,
            name: "shuf".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Shuffle(
                    basalt_bir::ShuffleKind::Idx,
                    ValRef::Param(0),
                    ValRef::Param(0),
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        let err = Rv32
            .emit(&wrap(f), &EmitOpts::default())
            .expect_err("must refuse, not guess");
        assert_eq!(err.code, ECode::UnsupportedOp);
    }

    #[test]
    fn emit_is_deterministic() {
        let module = wrap(func_vector_add());
        let a = Rv32.emit(&module, &EmitOpts::default()).unwrap();
        let b = Rv32.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(
            a, b,
            "same module in must yield byte-identical artifact out"
        );
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(Rv32.name(), "rv32im");
    }

    #[test]
    fn ignores_launch_bounds_metadata() {
        let mut module = wrap(func_ret_const());
        module.launch_bounds = Some(LaunchBounds {
            max_threads: 128,
            min_blocks: 2,
        });
        assert_eq!(Rv32.supports(&module), Support::Supported);
    }

    #[test]
    fn i64_add_sub_mul_and_icmp_are_supported() {
        let i64t = Ty::Scalar(Scalar::I64);
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
        ] {
            let f = Function {
                is_kernel: true,
                name: "i64bin".into(),
                params: vec![i64t, i64t],
                ret: i64t,
                insts: vec![Inst {
                    ty: i64t,
                    op: Op::Bin(op, ValRef::Param(0), ValRef::Param(1)),
                }],
                blocks: vec![Block {
                    insts: vec![InstId(0)],
                    term: Term::Ret(Some(ValRef::Val(InstId(0)))),
                }],
            };
            assert_eq!(Rv32.supports(&wrap(f.clone())), Support::Supported);
            let artifact = Rv32
                .emit(&wrap(f), &EmitOpts::default())
                .expect("emit succeeds");
            parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "i64bin");
        }
    }

    #[test]
    fn i64_shift_and_div_are_refused() {
        let i64t = Ty::Scalar(Scalar::I64);
        for op in [BinOp::Shl, BinOp::Lshr, BinOp::Ashr, BinOp::Div, BinOp::Rem] {
            let f = Function {
                is_kernel: true,
                name: "i64shift".into(),
                params: vec![i64t, i64t],
                ret: i64t,
                insts: vec![Inst {
                    ty: i64t,
                    op: Op::Bin(op, ValRef::Param(0), ValRef::Param(1)),
                }],
                blocks: vec![Block {
                    insts: vec![InstId(0)],
                    term: Term::Ret(Some(ValRef::Val(InstId(0)))),
                }],
            };
            assert_eq!(
                Rv32.supports(&wrap(f)),
                Support::Unsupported(ECode::UnsupportedType)
            );
        }
    }
}
