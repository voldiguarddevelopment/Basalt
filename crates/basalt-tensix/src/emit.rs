// BIR -> Metalium C++: a single-core, single-kernel text emitter for Tenstorrent Tensix. Unlike
// every other backend in this tree, Tensix has no register machine or ISA in the ordinary
// sense — a "kernel" is C++ compiled against the real `tt_metal` device API and run on one of a
// Tensix tile's own RISC-V cores (there is no general "any GPU kernel" story yet; that is
// P12-T4's Tile-DataFlow layer, later work). This file covers only the first real bring-up:
// prove a genuine, minimal, `tt-metal`-shaped kernel compiles and runs the same arithmetic as
// every other backend's `vector_add.cu`, and refuse everything else with a stable E-code rather
// than guess.
//
// # Real toolchain this was designed against and verified with
//
// Grounded in `tenstorrent/tt-metal`'s own example kernels, not invented: the data-movement
// kernel shape here is adapted from
// `tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp`
// — a plain `void kernel_main()` that reads `get_arg_val<uint32_t>(i)` runtime args, moves data
// DRAM<->L1 via `noc_async_read`/`noc_async_read_barrier`/`noc_async_write`/
// `noc_async_write_barrier`, and does ordinary scalar C++ arithmetic on the L1 bytes in between
// — the same real API a data-movement (`BRISC`/`NCRISC`) RISC-V core uses in production. The
// generated text was real-compiled (not merely "the Rust code didn't panic") against the actual
// `tt_metal` kernel headers using the real `sfpi` cross-toolchain (`riscv-tt-elf-g++`, the same
// compiler `tt_metal`'s own JIT build invokes), wrapped in `tt_metal`'s own real kernel firmware
// entry point (`tt_metal/hw/firmware/src/tt-1xx/brisck.cc`, which does
// `#include <kernel_includes.hpp>` -> the generated `.cpp`, exactly as `tt_metal`'s own
// `jit_build/genfiles.cpp` does for a legacy, non-Metal-2.0 kernel). See this task's own report
// for the exact commands and toolchain provenance (real `tenstorrent/sfpi` release, hash-
// verified).
//
// # Scope: what this backend actually lowers, and what it refuses
//
// One BIR function per module: a Metalium kernel file has exactly one fixed entry point name,
// `kernel_main()` — there is nowhere to put a second function, so a multi-function module
// refuses (`E090`) rather than pick one arbitrarily. Tensix has no SIMT thread grid the way a
// real GPU does — a kernel invocation runs once, serially, on one RISC-V core — so this backend
// synthesizes the same "run every thread of a flat launch one at a time in a native loop"
// technique `basalt-x86`'s oracle uses for the identical reason (see that crate's own module
// header): `tid.x` reads the loop's own counter, `bdim.x` reads a synthetic `nthreads` runtime
// arg (the same trailing-argument convention the oracle's calling convention documents),
// `bid.{x,y,z}`/`tid.{y,z}` are the constant `0`, `bdim.{y,z}`/`gdim.{x,y,z}` are the constant
// `1` — single-block scope, exactly the oracle's own documented limit. `barrier` becomes a
// comment-only no-op statement: within one synthesized loop, iterations already run strictly one
// after another, so there is nothing concurrent for a barrier to guard against, the same
// reasoning the oracle gives for its own `nop`. A BIR `ret` inside the per-thread body advances
// the loop (`continue;`) rather than returning from `kernel_main()` — mirroring the oracle's own
// `jmp __loop_incr` rather than a real `ret`.
//
// Every `Ty::Ptr` function parameter must be `AddrSpace::Global` (an ordinary DRAM buffer): this
// backend moves each one's entire `nthreads`-element extent DRAM<->L1 in one shot before/after
// the loop (`noc_async_read`/`_write` once per buffer, not once per element) — the honest
// single-core shape, no multi-core NoC coordination, no circular buffers, no tiles. A pointer
// parameter's element type is inferred from the one consistent scalar type every `load`/`store`
// reaching it (traced back through pointer arithmetic, `phi`, and `select` — see `root_param`)
// uses; more than one distinct type through the same parameter, or a parameter never actually
// dereferenced, refuses (`E091`) rather than guessing a width. Scalar function parameters are
// limited to `i1`/`i8`/`i16`/`i32` (a single `get_arg_val<uint32_t>` slot each, mirroring
// `add_2_integers_in_riscv`'s own runtime-arg convention); `i64`/`f32`/`f64` scalar parameters
// refuse (`E091`) rather than guess at a two-slot or bit-reinterpret convention this backend has
// not real-compile-verified.
//
// Refused outright, all `E090`/`E091`/`E092`, never guessed at: `Ty::Vec` (no per-lane story
// attempted this bring-up), `f16` (Tensix's native low-precision format is `bfloat16`, a
// different bit layout entirely — not a safe stand-in for BIR's presumed-IEEE `f16`), any
// non-`Global` address space (no L1 scratch/circular-buffer memory model yet — that is
// `P12-T4`'s job), `shuffle`/`ballot`/`vote.any`/`vote.all` (warp-collective, no single-core
// meaning), `atomic`/`atomic.cas` (unverified against this narrow scope), `mma` (no tile/matrix
// path), and `bitcast` (no bit-reinterpretation technique here has been real-compile-verified).
//
// # Value representation
//
// Every SSA value gets its own predeclared C++ local, typed by its BIR `Ty` (`bool`/`intN_t`/
// `float`/`double`/`uint8_t*`), assigned via a plain statement at its definition point and read
// back by name — never re-derived — the same one-slot-per-SSA-value discipline every other
// backend in this tree uses, translated to C++ locals instead of virtual/physical registers.
// Declaring every value ahead of the loop (rather than at first use) means a `goto`-based
// lowering of BIR's basic blocks (one `L<id>:` label per block; `br`/`condbr`/`switch` become
// `goto`/`if`/an `if`-chain) never jumps over a variable's initialization, which C++ forbids.
// `Ty::Ptr(Global)` values are represented uniformly as `uint8_t*` (byte-addressed), matching
// BIR's own byte-scaled pointer arithmetic convention (`basalt-sema/src/lower.rs`'s
// `lower_ptr_offset` always emits `add ptr, byte_off` with the pointer operand first) — so
// pointer arithmetic lowers as plain `uint8_t*` addition with no per-element-width bookkeeping
// in this file at all.

use std::collections::HashMap;

use basalt_backend::{Artifact, ArtifactKind, Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, BinOp, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId, Module, Op, Scalar, Term,
    Ty, ValRef,
};
use basalt_diag::{Diag, ECode};
use basalt_passes::construct_ssa;

// ---- type mapping ---------------------------------------------------------------------------

pub(crate) fn f16_refusal(what: &'static str) -> Diag {
    Diag::new(ECode::UnsupportedType).with_arg(format!(
        "f16: not a safe stand-in for Tensix's native bfloat16 format ({what})"
    ))
}

/// The C++ type a BIR scalar is represented as. Every non-pointer SSA value keeps the sign
/// convention its declared width implies — see the module header on why `zext`/`fptoui` still
/// need an explicit unsigned detour despite this.
pub(crate) fn cpp_scalar_ty(s: Scalar) -> Result<&'static str, Diag> {
    match s {
        Scalar::I1 => Ok("bool"),
        Scalar::I8 => Ok("int8_t"),
        Scalar::I16 => Ok("int16_t"),
        Scalar::I32 => Ok("int32_t"),
        Scalar::I64 => Ok("int64_t"),
        Scalar::F32 => Ok("float"),
        Scalar::F64 => Ok("double"),
        Scalar::F16 => Err(f16_refusal("scalar value")),
    }
}

pub(crate) fn cpp_unsigned_scalar_ty(s: Scalar) -> &'static str {
    match s {
        Scalar::I1 => "bool",
        Scalar::I8 => "uint8_t",
        Scalar::I16 => "uint16_t",
        Scalar::I32 => "uint32_t",
        Scalar::I64 => "uint64_t",
        Scalar::F32 | Scalar::F64 | Scalar::F16 => {
            unreachable!("cpp_unsigned_scalar_ty called on a non-integer scalar")
        }
    }
}

pub(crate) fn cpp_ty(ty: Ty) -> Result<String, Diag> {
    match ty {
        Ty::Scalar(s) => cpp_scalar_ty(s).map(|s| s.to_string()),
        Ty::Ptr(AddrSpace::Global) => Ok("uint8_t*".to_string()),
        Ty::Ptr(_) => Err(Diag::new(ECode::UnsupportedAddressSpace)
            .with_arg("only ptr.global is modeled by this backend's single-core bring-up")),
        Ty::Vec(..) => {
            Err(Diag::new(ECode::UnsupportedType).with_arg("vector types are not lowered yet"))
        }
        Ty::Void => unreachable!("cpp_ty called on a void-typed value"),
    }
}

fn zero_literal(ty: Ty) -> Result<&'static str, Diag> {
    match ty {
        Ty::Scalar(Scalar::I1) => Ok("false"),
        Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64) => Ok("0"),
        Ty::Scalar(Scalar::F32) => Ok("0.0f"),
        Ty::Scalar(Scalar::F64) => Ok("0.0"),
        Ty::Scalar(Scalar::F16) => Err(f16_refusal("zero-initializer")),
        Ty::Ptr(AddrSpace::Global) => Ok("nullptr"),
        Ty::Ptr(_) => Err(Diag::new(ECode::UnsupportedAddressSpace)
            .with_arg("only ptr.global is modeled by this backend's single-core bring-up")),
        Ty::Vec(..) => {
            Err(Diag::new(ECode::UnsupportedType).with_arg("vector types are not lowered yet"))
        }
        Ty::Void => unreachable!("zero_literal called on a void-typed value"),
    }
}

pub(crate) fn scalar_of(ty: Ty) -> Scalar {
    match ty {
        Ty::Scalar(s) => s,
        _ => unreachable!("scalar_of called on a non-scalar type"),
    }
}

// ---- value/type lookups ----------------------------------------------------------------------

fn valref_ty(f: &Function, v: ValRef) -> Ty {
    match v {
        ValRef::Param(i) => f.params[i as usize],
        ValRef::Val(id) => f.insts[id.0 as usize].ty,
    }
}

pub(crate) fn val_text(v: ValRef) -> String {
    match v {
        ValRef::Param(i) => format!("p{i}"),
        ValRef::Val(id) => format!("v{}", id.0),
    }
}

// ---- pointer provenance --------------------------------------------------------------------

/// Traces a `Ty::Ptr(Global)` value back to the single function parameter it ultimately derives
/// from, following the only pointer-producing shapes BIR/sema ever emit: a direct parameter
/// reference, `basalt-sema`'s own `lower_ptr_offset` (`add ptr, byte_off`, pointer operand
/// always first — see the module header), and `phi`/`select` merging the same parameter from
/// every arm. Anything else (a pointer whose arms disagree, or that isn't traceable at all)
/// returns `None`, which the caller turns into a clean refusal rather than a guess. `seen` guards
/// against a cycle through a loop-carried `phi` recursing forever.
fn root_param(f: &Function, v: ValRef, seen: &mut Vec<InstId>) -> Option<u32> {
    match v {
        ValRef::Param(i) => Some(i),
        ValRef::Val(id) => {
            if seen.contains(&id) {
                return None;
            }
            seen.push(id);
            let inst = &f.insts[id.0 as usize];
            let result = match &inst.op {
                Op::Bin(BinOp::Add, a, b) if matches!(inst.ty, Ty::Ptr(_)) => {
                    if matches!(valref_ty(f, *a), Ty::Ptr(_)) {
                        root_param(f, *a, seen)
                    } else if matches!(valref_ty(f, *b), Ty::Ptr(_)) {
                        root_param(f, *b, seen)
                    } else {
                        None
                    }
                }
                Op::Phi(preds) => {
                    let mut result = None;
                    for &(_, incoming) in preds {
                        let r = root_param(f, incoming, seen)?;
                        match result {
                            None => result = Some(r),
                            Some(existing) if existing == r => {}
                            _ => return None,
                        }
                    }
                    result
                }
                Op::Select(_, a, b) => {
                    let ra = root_param(f, *a, seen)?;
                    let rb = root_param(f, *b, seen)?;
                    if ra == rb {
                        Some(ra)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            seen.pop();
            result
        }
    }
}

fn cannot_trace_ptr() -> Diag {
    Diag::new(ECode::UnsupportedOp)
        .with_arg("cannot trace this pointer value back to a single originating kernel parameter")
}

// ---- per-parameter classification ------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct PtrAccess {
    pub(crate) scalar: Option<Scalar>,
    pub(crate) has_load: bool,
    pub(crate) has_store: bool,
}

impl PtrAccess {
    fn empty() -> PtrAccess {
        PtrAccess {
            scalar: None,
            has_load: false,
            has_store: false,
        }
    }
}

pub(crate) enum ParamKind {
    Scalar(Scalar),
    Ptr(PtrAccess),
}

/// Records one `load`/`store` access against the parameter `ptr` traces back to: the first
/// access fixes that parameter's element scalar type, every later access must agree, and
/// `is_load` sets the corresponding direction flag (a parameter can be both read and written).
fn record_access(
    kinds: &mut [ParamKind],
    f: &Function,
    ptr: ValRef,
    accessed_ty: Ty,
    is_load: bool,
) -> Result<(), Diag> {
    let root = root_param(f, ptr, &mut Vec::new()).ok_or_else(cannot_trace_ptr)?;
    let access = match &mut kinds[root as usize] {
        ParamKind::Ptr(access) => access,
        ParamKind::Scalar(_) => unreachable!("root_param never resolves to a scalar param"),
    };
    let scalar = scalar_of(accessed_ty);
    match access.scalar {
        None => access.scalar = Some(scalar),
        Some(existing) if existing.text() == scalar.text() => {}
        Some(_) => {
            return Err(Diag::new(ECode::UnsupportedType).with_arg(
                "a kernel pointer parameter is dereferenced at more than one scalar type",
            ))
        }
    }
    if is_load {
        access.has_load = true;
    } else {
        access.has_store = true;
    }
    Ok(())
}

/// Single source of truth for what a function's parameters mean to this backend: every scalar
/// parameter must be a type this backend can read as one `get_arg_val<uint32_t>` slot, and every
/// pointer parameter must be `ptr.global`, actually dereferenced somewhere, and dereferenced at
/// one consistent scalar type throughout (see the module header). Shared verbatim by
/// `supports()` and `emit()`.
pub(crate) fn analyze_params(f: &Function) -> Result<Vec<ParamKind>, Diag> {
    let mut kinds: Vec<ParamKind> = Vec::with_capacity(f.params.len());
    for &ty in &f.params {
        kinds.push(match ty {
            Ty::Scalar(s @ (Scalar::I1 | Scalar::I8 | Scalar::I16 | Scalar::I32)) => {
                ParamKind::Scalar(s)
            }
            Ty::Scalar(Scalar::I64 | Scalar::F32 | Scalar::F64) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "64-bit and floating-point scalar kernel parameters are not yet supported",
                ))
            }
            Ty::Scalar(Scalar::F16) => return Err(f16_refusal("scalar parameter")),
            Ty::Ptr(AddrSpace::Global) => ParamKind::Ptr(PtrAccess::empty()),
            Ty::Ptr(_) => {
                return Err(Diag::new(ECode::UnsupportedAddressSpace)
                    .with_arg("only ptr.global kernel parameters are modeled by this backend"))
            }
            Ty::Vec(..) => {
                return Err(
                    Diag::new(ECode::UnsupportedType).with_arg("vector-typed kernel parameter")
                )
            }
            Ty::Void => unreachable!("a function parameter is never void"),
        });
    }

    for inst in &f.insts {
        match &inst.op {
            Op::Load { ptr, space, .. } => {
                if *space != AddrSpace::Global {
                    return Err(Diag::new(ECode::UnsupportedAddressSpace)
                        .with_arg("only ptr.global loads are modeled by this backend"));
                }
                record_access(&mut kinds, f, *ptr, inst.ty, true)?;
            }
            Op::Store { ptr, ty, space, .. } => {
                if *space != AddrSpace::Global {
                    return Err(Diag::new(ECode::UnsupportedAddressSpace)
                        .with_arg("only ptr.global stores are modeled by this backend"));
                }
                record_access(&mut kinds, f, *ptr, *ty, false)?;
            }
            _ => {}
        }
    }

    for kind in &kinds {
        if let ParamKind::Ptr(access) = kind {
            if access.scalar.is_none() {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "a ptr.global kernel parameter is never dereferenced; its element width \
                     cannot be sized for a NoC transfer",
                ));
            }
        }
    }

    Ok(kinds)
}

// ---- structural refusal surface --------------------------------------------------------------

fn ty_has_f16(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F16) | Ty::Vec(Scalar::F16, _))
}

fn ty_has_vec(ty: Ty) -> bool {
    matches!(ty, Ty::Vec(..))
}

fn check_inst(inst: &Inst) -> Result<(), Diag> {
    if ty_has_f16(inst.ty) {
        return Err(f16_refusal("instruction result"));
    }
    if ty_has_vec(inst.ty) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed instruction result"));
    }
    match &inst.op {
        Op::Cast(op, sty, _) => {
            if ty_has_f16(*sty) {
                return Err(f16_refusal("cast source"));
            }
            if ty_has_vec(*sty) {
                return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed cast source"));
            }
            if matches!(op, CastOp::Bitcast) {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                    "bitcast has no real-compile-verified reinterpretation technique in this \
                     backend yet",
                ));
            }
        }
        Op::Store { ty: sty, .. } => {
            if ty_has_f16(*sty) {
                return Err(f16_refusal("store value"));
            }
            if ty_has_vec(*sty) {
                return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed store value"));
            }
        }
        Op::ConstInt(_) => {
            if matches!(inst.ty, Ty::Ptr(_)) {
                return Err(Diag::new(ECode::UnsupportedAddressSpace).with_arg(
                    "synthetic local/shared/constant/param slot addresses are not modeled by \
                     this backend; only parameter-derived ptr.global values are supported",
                ));
            }
        }
        Op::Bin(op, ..) => {
            if matches!(inst.ty, Ty::Ptr(_)) && *op != BinOp::Add {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                    "the only pointer-producing arithmetic this backend models is the \
                     pointer-plus-byte-offset shape basalt-sema's lower_ptr_offset emits",
                ));
            }
        }
        Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
            return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                "warp-collective ops have no single-core Tensix meaning in this backend",
            ));
        }
        Op::Atomic(..) | Op::AtomicCas(..) => {
            return Err(Diag::new(ECode::UnsupportedOp)
                .with_arg("atomics are not modeled by this backend's single-core bring-up"));
        }
        Op::Mma { .. } => {
            return Err(Diag::new(ECode::UnsupportedOp)
                .with_arg("mma has no tile/matrix-engine lowering in this backend yet"));
        }
        _ => {}
    }
    Ok(())
}

/// Single source of truth for what this backend refuses, shared verbatim by `supports()` and
/// `emit()`. A Metalium kernel file has exactly one `kernel_main()`; a module with more than one
/// function has nowhere to put the second one, so it refuses here too.
pub(crate) fn check_module(module: &Module) -> Result<(), Diag> {
    if module.funcs.len() != 1 {
        return Err(Diag::new(ECode::UnsupportedOp).with_arg(
            "this backend's single-core bring-up supports exactly one function per module \
             (a Metalium kernel file has exactly one kernel_main())",
        ));
    }
    let f = &module.funcs[0];
    if !f.is_kernel {
        return Err(Diag::new(ECode::UnsupportedOp)
            .with_arg("host/non-kernel function compilation is not yet implemented"));
    }
    if ty_has_f16(f.ret) {
        return Err(f16_refusal("return type"));
    }
    if !matches!(f.ret, Ty::Void) {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("a Tensix kernel entry point has no way to hand a value back to the host"));
    }
    for inst in &f.insts {
        check_inst(inst)?;
    }
    analyze_params(f)?;
    Ok(())
}

// ---- phi resolution ---------------------------------------------------------------------------

/// `(from_block, to_block) -> [(phi's own InstId, incoming value)]`, exactly `basalt-ptx`'s own
/// `PhiCopies` shape: every predecessor writes a phi's variable directly before jumping, so the
/// definition site itself (`Op::Phi`) emits nothing.
pub(crate) type PhiCopies = HashMap<(u32, u32), Vec<(InstId, ValRef)>>;

pub(crate) fn build_phi_copies(f: &Function) -> PhiCopies {
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

/// Every block id an actual `goto` targets somewhere in the function. Block 0 (the loop's own
/// entry) is reached by falling straight out of the `for` line, never by name — real GCC's
/// `-Wunused-label` flags a label with no `goto` naming it even when it is otherwise reachable
/// by fallthrough, so a label is only worth emitting for a block some other block actually jumps
/// to (`-Werror` in `tt_metal`'s own real kernel build would otherwise turn that warning into a
/// hard compile failure).
pub(crate) fn goto_targets(f: &Function) -> std::collections::HashSet<u32> {
    let mut targets = std::collections::HashSet::new();
    for block in &f.blocks {
        match &block.term {
            Term::Br(t) => {
                targets.insert(t.0);
            }
            Term::CondBr(_, t, e) => {
                targets.insert(t.0);
                targets.insert(e.0);
            }
            Term::Switch(_, default, cases) => {
                targets.insert(default.0);
                for &(_, t) in cases {
                    targets.insert(t.0);
                }
            }
            Term::Ret(_) => {}
        }
    }
    targets
}

// ---- code generation ---------------------------------------------------------------------

pub(crate) struct CodeGen<'a> {
    pub(crate) f: &'a Function,
    pub(crate) params: &'a [ParamKind],
    pub(crate) phi_copies: PhiCopies,
    pub(crate) out: String,
}

impl<'a> CodeGen<'a> {
    pub(crate) fn line(&mut self, text: &str) {
        self.out.push_str("    ");
        self.out.push_str(text);
        self.out.push('\n');
    }

    // ---- runtime args, buffer setup ---------------------------------------------------

    fn emit_prologue(&mut self) -> Result<(), Diag> {
        let mut arg_idx = 0u32;
        for (i, kind) in self.params.iter().enumerate() {
            match kind {
                ParamKind::Scalar(s) => {
                    let cty = cpp_scalar_ty(*s)?;
                    self.line(&format!(
                        "{cty} p{i} = ({cty})get_arg_val<uint32_t>({arg_idx});"
                    ));
                    arg_idx += 1;
                }
                ParamKind::Ptr(_) => {
                    self.line(&format!(
                        "uint32_t p{i}_dram = get_arg_val<uint32_t>({arg_idx});"
                    ));
                    arg_idx += 1;
                    self.line(&format!(
                        "uint32_t p{i}_l1 = get_arg_val<uint32_t>({arg_idx});"
                    ));
                    arg_idx += 1;
                }
            }
        }
        self.line(&format!(
            "uint32_t nthreads = get_arg_val<uint32_t>({arg_idx});"
        ));
        self.out.push('\n');

        let mut any_load = false;
        for (i, kind) in self.params.iter().enumerate() {
            if let ParamKind::Ptr(access) = kind {
                let scalar = access.scalar.expect(
                    "analyze_params requires every ptr.global parameter to be dereferenced",
                );
                let cty = cpp_scalar_ty(scalar)?;
                self.line(&format!("uint8_t* p{i} = (uint8_t*)(uintptr_t)p{i}_l1;"));
                self.line(&format!(
                    "InterleavedAddrGen<true> p{i}_gen = {{.bank_base_address = p{i}_dram, \
                     .page_size = sizeof({cty})}};"
                ));
                if access.has_load {
                    any_load = true;
                    self.line(&format!(
                        "noc_async_read(p{i}_gen.get_noc_addr(0), p{i}_l1, \
                         (uint32_t)(nthreads * sizeof({cty})));"
                    ));
                }
            }
        }
        if any_load {
            self.line("noc_async_read_barrier();");
        }
        self.out.push('\n');
        Ok(())
    }

    fn emit_epilogue(&mut self) -> Result<(), Diag> {
        self.out.push('\n');
        let mut any_store = false;
        for (i, kind) in self.params.iter().enumerate() {
            if let ParamKind::Ptr(access) = kind {
                if access.has_store {
                    any_store = true;
                    let scalar = access.scalar.expect("checked by analyze_params");
                    let cty = cpp_scalar_ty(scalar)?;
                    self.line(&format!(
                        "noc_async_write(p{i}_l1, p{i}_gen.get_noc_addr(0), \
                         (uint32_t)(nthreads * sizeof({cty})));"
                    ));
                }
            }
        }
        if any_store {
            self.line("noc_async_write_barrier();");
        }
        Ok(())
    }

    pub(crate) fn emit_ssa_decls(&mut self) -> Result<(), Diag> {
        for (id, inst) in self.f.insts.iter().enumerate() {
            if !inst.has_result() {
                continue;
            }
            let cty = cpp_ty(inst.ty)?;
            let zero = zero_literal(inst.ty)?;
            self.line(&format!("{cty} v{id} = {zero};"));
        }
        Ok(())
    }

    // ---- phi copies --------------------------------------------------------------------

    fn phi_copy_lines(&self, from: u32, to: u32) -> Vec<String> {
        let Some(copies) = self.phi_copies.get(&(from, to)) else {
            return Vec::new();
        };
        copies
            .iter()
            .map(|(phi_id, incoming)| format!("v{} = {};", phi_id.0, val_text(*incoming)))
            .collect()
    }

    // ---- instruction lowering -----------------------------------------------------------

    fn expr_bin(&self, op: BinOp, a: ValRef, b: ValRef, ty: Ty) -> Result<String, Diag> {
        let at = val_text(a);
        let bt = val_text(b);
        if let Ty::Ptr(AddrSpace::Global) = ty {
            // basalt-sema's lower_ptr_offset always puts the pointer operand first (see the
            // module header); check_module already refused any other pointer-producing Bin
            // shape, so `a` is the pointer here.
            return Ok(format!("({at} + {bt})"));
        }
        let scalar = scalar_of(ty);
        Ok(match op {
            BinOp::Add | BinOp::FAdd => format!("({at} + {bt})"),
            BinOp::Sub | BinOp::FSub => format!("({at} - {bt})"),
            BinOp::Mul | BinOp::FMul => format!("({at} * {bt})"),
            BinOp::Div | BinOp::FDiv => format!("({at} / {bt})"),
            BinOp::Rem => format!("({at} % {bt})"),
            BinOp::FRem => {
                // No libm dependency in a freestanding kernel build: truncate-toward-zero via a
                // cast, matching basalt-x86/basalt-ptx's own frem emulation (`a - trunc(a/b)*b`).
                let cty = cpp_scalar_ty(scalar)?;
                let itrunc = if matches!(scalar, Scalar::F64) {
                    "int64_t"
                } else {
                    "int32_t"
                };
                format!("({at} - ({cty})({itrunc})({at} / {bt}) * {bt})")
            }
            BinOp::And => format!("({at} & {bt})"),
            BinOp::Or => format!("({at} | {bt})"),
            BinOp::Xor => format!("({at} ^ {bt})"),
            BinOp::Shl => format!("({at} << {bt})"),
            BinOp::Ashr => format!("({at} >> {bt})"),
            BinOp::Lshr => {
                let cty = cpp_scalar_ty(scalar)?;
                let uty = cpp_unsigned_scalar_ty(scalar);
                format!("({cty})(({uty}){at} >> {bt})")
            }
        })
    }

    fn expr_icmp(&self, pred: ICmpPred, cty: Ty, a: ValRef, b: ValRef) -> String {
        let at = val_text(a);
        let bt = val_text(b);
        let op = match pred {
            ICmpPred::Eq => "==",
            ICmpPred::Ne => "!=",
            ICmpPred::Slt | ICmpPred::Ult => "<",
            ICmpPred::Sle | ICmpPred::Ule => "<=",
            ICmpPred::Sgt | ICmpPred::Ugt => ">",
            ICmpPred::Sge | ICmpPred::Uge => ">=",
        };
        let unsigned = matches!(
            pred,
            ICmpPred::Ult | ICmpPred::Ule | ICmpPred::Ugt | ICmpPred::Uge
        );
        if unsigned {
            if let Ty::Scalar(s) = cty {
                if !matches!(s, Scalar::I1) {
                    let uty = cpp_unsigned_scalar_ty(s);
                    return format!("(({uty}){at} {op} ({uty}){bt})");
                }
            }
        }
        format!("({at} {op} {bt})")
    }

    fn expr_fcmp(&self, pred: FCmpPred, a: ValRef, b: ValRef) -> String {
        let at = val_text(a);
        let bt = val_text(b);
        match pred {
            FCmpPred::Oeq => format!("({at} == {bt})"),
            FCmpPred::One => format!("({at} != {bt})"),
            FCmpPred::Olt => format!("({at} < {bt})"),
            FCmpPred::Ole => format!("({at} <= {bt})"),
            FCmpPred::Ogt => format!("({at} > {bt})"),
            FCmpPred::Oge => format!("({at} >= {bt})"),
            // No libm/<cmath> dependency: the self-compare NaN trick is plain arithmetic.
            FCmpPred::Ord => format!("(({at} == {at}) && ({bt} == {bt}))"),
            FCmpPred::Uno => format!("(({at} != {at}) || ({bt} != {bt}))"),
        }
    }

    fn expr_cast(&self, op: CastOp, sty: Ty, v: ValRef, dty: Ty) -> Result<String, Diag> {
        let vt = val_text(v);
        let dcty = cpp_ty(dty)?;
        Ok(match op {
            // `v` is already stored at its true declared width/signedness, so a plain narrowing
            // cast is exactly `trunc`, and a plain widening cast is exactly `sext` (C++ signed
            // integer promotion sign-extends) — see the module header on value representation.
            CastOp::Trunc
            | CastOp::Sext
            | CastOp::FpTrunc
            | CastOp::FpExt
            | CastOp::FpToSi
            | CastOp::SiToFp => format!("({dcty}){vt}"),
            CastOp::Zext => {
                let sscalar = scalar_of(sty);
                let uscty = cpp_unsigned_scalar_ty(sscalar);
                format!("({dcty})({uscty}){vt}")
            }
            CastOp::FpToUi => {
                let dscalar = scalar_of(dty);
                let udty = cpp_unsigned_scalar_ty(dscalar);
                format!("({dcty})({udty}){vt}")
            }
            CastOp::UiToFp => {
                let sscalar = scalar_of(sty);
                let uscty = cpp_unsigned_scalar_ty(sscalar);
                format!("({dcty})({uscty}){vt}")
            }
            CastOp::Bitcast => unreachable!("check_module refuses bitcast before codegen starts"),
        })
    }

    pub(crate) fn lower_inst(&mut self, id: InstId) -> Result<(), Diag> {
        let inst = &self.f.insts[id.0 as usize];
        let ty = inst.ty;
        match &inst.op {
            Op::ConstInt(n) => {
                let n = *n;
                let s = scalar_of(ty);
                let text = match s {
                    Scalar::I1 => (if n != 0 { "true" } else { "false" }).to_string(),
                    Scalar::I8 => format!("(int8_t){}", n as i8),
                    Scalar::I16 => format!("(int16_t){}", n as i16),
                    Scalar::I32 => format!("(int32_t){}", n as i32),
                    // Printed through the unsigned bit pattern first: a plain decimal literal
                    // for i64::MIN would overflow a signed literal's own type in C++.
                    Scalar::I64 => format!("(int64_t)(uint64_t){}ULL", n as u64),
                    Scalar::F32 | Scalar::F64 | Scalar::F16 => {
                        unreachable!("check_module refuses non-integer ConstInt")
                    }
                };
                self.line(&format!("v{} = {text};", id.0));
            }
            Op::ConstFloat(v) => {
                let v = *v;
                let s = scalar_of(ty);
                let text = match s {
                    Scalar::F32 => float_literal_f32(v as f32),
                    Scalar::F64 => float_literal_f64(v),
                    _ => unreachable!("check_module refuses non-float ConstFloat"),
                };
                self.line(&format!("v{} = {text};", id.0));
            }
            Op::Bin(op, a, b) => {
                let (op, a, b) = (*op, *a, *b);
                let expr = self.expr_bin(op, a, b, ty)?;
                self.line(&format!("v{} = {expr};", id.0));
            }
            Op::ICmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                let expr = self.expr_icmp(pred, cty, a, b);
                self.line(&format!("v{} = {expr};", id.0));
            }
            Op::FCmp(pred, _cty, a, b) => {
                let (pred, a, b) = (*pred, *a, *b);
                let expr = self.expr_fcmp(pred, a, b);
                self.line(&format!("v{} = {expr};", id.0));
            }
            Op::Select(c, a, b) => {
                let (c, a, b) = (val_text(*c), val_text(*a), val_text(*b));
                self.line(&format!("v{} = {c} ? {a} : {b};", id.0));
            }
            Op::Cast(cop, sty, v) => {
                let (cop, sty, v) = (*cop, *sty, *v);
                let expr = self.expr_cast(cop, sty, v, ty)?;
                self.line(&format!("v{} = {expr};", id.0));
            }
            Op::Load { ptr, .. } => {
                let cty = cpp_scalar_ty(scalar_of(ty))?;
                self.line(&format!("v{} = *({cty}*)({});", id.0, val_text(*ptr)));
            }
            Op::Store {
                ptr, val, ty: sty, ..
            } => {
                let cty = cpp_scalar_ty(scalar_of(*sty))?;
                self.line(&format!(
                    "*({cty}*)({}) = {};",
                    val_text(*ptr),
                    val_text(*val)
                ));
            }
            Op::Phi(_) => {
                // Every predecessor writes this phi's own variable before jumping here — see
                // `phi_copy_lines`.
            }
            Op::TidX => self.line(&format!("v{} = ({})__tid;", id.0, cpp_ty(ty)?)),
            Op::BdimX => self.line(&format!("v{} = ({})nthreads;", id.0, cpp_ty(ty)?)),
            Op::TidY | Op::TidZ | Op::BidX | Op::BidY | Op::BidZ => {
                self.line(&format!("v{} = ({})0;", id.0, cpp_ty(ty)?));
            }
            Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ => {
                self.line(&format!("v{} = ({})1;", id.0, cpp_ty(ty)?));
            }
            Op::Barrier => {
                // Iterations of the synthesized per-thread loop already run strictly one after
                // another (see the module header), so there is nothing for a barrier to guard —
                // an intentional, documented no-op, not a dropped instruction.
                self.line("/* barrier: no-op under this backend's one-thread-at-a-time loop */;");
            }
            Op::Shuffle(..)
            | Op::Ballot(..)
            | Op::VoteAny(..)
            | Op::VoteAll(..)
            | Op::Atomic(..)
            | Op::AtomicCas(..)
            | Op::Mma { .. } => {
                unreachable!("check_module refuses these before codegen starts")
            }
        }
        Ok(())
    }

    // ---- terminators --------------------------------------------------------------------

    pub(crate) fn lower_term(&mut self, from_block: u32, term: &Term) {
        match term {
            Term::Br(target) => {
                for copy in self.phi_copy_lines(from_block, target.0) {
                    self.line(&copy);
                }
                self.line(&format!("goto L{};", target.0));
            }
            Term::CondBr(cond, t, f) => {
                self.line(&format!("if ({}) {{", val_text(*cond)));
                for copy in self.phi_copy_lines(from_block, t.0) {
                    self.line(&copy);
                }
                self.line(&format!("goto L{};", t.0));
                self.line("} else {");
                for copy in self.phi_copy_lines(from_block, f.0) {
                    self.line(&copy);
                }
                self.line(&format!("goto L{};", f.0));
                self.line("}");
            }
            Term::Switch(scrut, default, cases) => {
                let sv = val_text(*scrut);
                for &(case_val, target) in cases {
                    self.line(&format!("if ({sv} == {case_val}) {{"));
                    for copy in self.phi_copy_lines(from_block, target.0) {
                        self.line(&copy);
                    }
                    self.line(&format!("goto L{};", target.0));
                    self.line("}");
                }
                for copy in self.phi_copy_lines(from_block, default.0) {
                    self.line(&copy);
                }
                self.line(&format!("goto L{};", default.0));
            }
            Term::Ret(_) => {
                // Every thread must still advance the loop, not actually return from
                // kernel_main() — see the module header.
                self.line("continue;");
            }
        }
    }
}

fn float_literal_f32(v: f32) -> String {
    if v.is_nan() {
        "__builtin_nanf(\"\")".to_string()
    } else if v.is_infinite() {
        if v < 0.0 {
            "(-__builtin_inff())".to_string()
        } else {
            "__builtin_inff()".to_string()
        }
    } else {
        format!("{v}f")
    }
}

fn float_literal_f64(v: f64) -> String {
    if v.is_nan() {
        "__builtin_nan(\"\")".to_string()
    } else if v.is_infinite() {
        if v < 0.0 {
            "(-__builtin_inf())".to_string()
        } else {
            "__builtin_inf()".to_string()
        }
    } else {
        format!("{v}")
    }
}

// ---- module/function assembly ------------------------------------------------------------

fn emit_function(f: &Function) -> Result<String, Diag> {
    let params = analyze_params(f)?;
    let phi_copies = build_phi_copies(f);
    let mut cg = CodeGen {
        f,
        params: &params,
        phi_copies,
        out: String::new(),
    };

    cg.out.push_str("void kernel_main() {\n");
    cg.emit_prologue()?;
    cg.emit_ssa_decls()?;
    cg.out.push('\n');
    cg.line("for (uint32_t __tid = 0; __tid < nthreads; ++__tid) {");
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
    cg.emit_epilogue()?;
    cg.out.push_str("}\n");
    Ok(cg.out)
}

// ---- Backend impl -------------------------------------------------------------------------

/// The Tenstorrent Metalium C++ text backend. `name()` returns `"tensix"`, matching
/// `basalt-cli`'s own `--tensix` flag spelling. See the module header for the full design.
#[derive(Debug, Default, Clone, Copy)]
pub struct Tensix;

impl Backend for Tensix {
    fn name(&self) -> &'static str {
        "tensix"
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
        let text = emit_function(&ssa_module.funcs[0])?;
        Ok(Artifact::text(ArtifactKind::Source, text))
    }
}

#[cfg(test)]
mod tests;
