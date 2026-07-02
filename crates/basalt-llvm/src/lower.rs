// BIR -> LLVM IR lowering: core scalar/memory/control-flow ops plus the GPU intrinsic
// surface. This builds an in-memory `inkwell::module::Module` and nothing else — no
// `TargetMachine`, no object/bitcode emission. That is later work layered on top of
// `lower_module`.
//
// # Scope
//
// Implemented: `const.i`/`const.f` (including `basalt-sema`'s local/param/shared/constant
// address-synthesis idiom, `const.i ptr.<space> N` — an opaque per-function slot key, not a
// real address; see `build_local_slots`, which gives each distinct key a real `alloca` in the
// function's entry block, the same pattern `basalt-x86`'s oracle documents and handles with
// its own per-function frame. A literal `ptr.global` constant, unusual in practice since real
// global pointers always arrive as parameters, instead becomes a genuine constant `inttoptr`,
// since LLVM has no integer-literal pointer constant of its own), `Bin` (signed `div`/`rem`,
// matching the convention already established by `basalt-x86`'s oracle and `basalt-ptx`'s
// emitter — BIR's `Bin` carries no signed/unsigned distinction, so this lane makes the same
// documented choice; `add`/`sub` additionally accept a pointer operand, BIR's own
// address-arithmetic idiom, via `ptrtoint`/`inttoptr` rather than the typed integer path),
// `icmp`/`fcmp`, `select`, every `CastOp` variant, `load`/`store`, `phi`, every `Term` variant,
// and the GPU intrinsic surface described below.
//
// Refused with `Err(Diag)` rather than guessed at: `Ty::Vec`.
//
// # GPU dialects
//
// `llvm.nvvm.*` and `llvm.amdgcn.*` intrinsics are mutually exclusive within one LLVM
// module — they are tied to whichever target's `TargetMachine` will eventually compile the
// IR — so `lower_module` takes a `GpuDialect` selecting which family a module's GPU ops lower
// into. A module with no GPU ops in it never looks at the dialect at all.
//
// Every intrinsic name/signature named below was confirmed against a real LLVM 18 install
// (`Intrinsic::find` resolving, then inspecting the declared function type) rather than
// assumed from documentation.
//
// Thread/block index (`tid.{x,y,z}`, `bid.{x,y,z}`, `bdim.{x,y,z}`, `gdim.{x,y,z}`):
//   - Nvptx: `llvm.nvvm.read.ptx.sreg.{tid,ctaid,ntid,nctaid}.{x,y,z}`, all `i32 ()`.
//   - Amdgpu: `llvm.amdgcn.workitem.id.{x,y,z}` for `tid`, `llvm.amdgcn.workgroup.id.{x,y,z}`
//     for `bid`, both `i32 ()`. `bdim`/`gdim` (workgroup size / grid size) have **no**
//     no-argument LLVM 18 intrinsic the way NVPTX does — the only path is loading fixed-offset
//     fields out of `llvm.amdgcn.dispatch.ptr`'s HSA dispatch packet, and that offset layout is
//     an ABI-version-specific detail this lane is not confident enough in to hardcode. `bdim`/
//     `gdim` therefore return `Err(Diag)` for `Amdgpu` only; `Nvptx` fully supports all twelve.
//
// `barrier`: `llvm.nvvm.barrier0` (`void ()`) on Nvptx, `llvm.amdgcn.s.barrier` (`void ()`) on
// Amdgpu. Both dialects supported.
//
// `shuffle.*`: Nvptx uses `llvm.nvvm.shfl.sync.{idx,up,down,bfly}.i32(mask, val, b, c)` — a
// full-warp `mask` of `0xffffffff` (BIR carries no narrower mask, matching `basalt-ptx`'s own
// documented stance), `b` the lane offset/index (BIR's `amt` operand), and `c` the segment
// clamp, reusing `basalt-ptx`'s own `0x1f`/`0x0` convention so the two backends agree on what
// "up" means. Implemented for `i32`- and `f32`-typed shuffles (the latter via a bitcast in and
// out of `i32`); wider/narrower scalar widths would need the multi-word split `basalt-ptx`
// does and are left as `Err(Diag)`. Amdgpu has no settled mapping — `llvm.amdgcn.ds.bpermute`/
// `.ds.permute` move data by byte-addressed lane offset rather than BIR's idx/up/down/xor
// shuffle kinds, and the up/down clamp behavior differs between wave32 and wave64 parts — so
// `shuffle.*` is `Err(Diag)` for `Amdgpu`.
//
// `ballot`/`vote.any`/`vote.all`: BIR's operand is a truthy int (sema coerces a predicate
// expression to `int`, not `i1`), so every dialect first derives a real `i1` via `icmp ne 0`
// (a no-op when the operand is already `i1`). Nvptx: `llvm.nvvm.vote.ballot.sync(mask, pred)`
// -> `i32`, `llvm.nvvm.vote.{any,all}.sync(mask, pred)` -> `i1` (zero/sign-extended to the
// instruction's declared result width). Amdgpu: `llvm.amdgcn.ballot` is overloaded on its
// return width (confirmed both an `i32` and an `i64` lane-mask overload resolve), so `ballot`
// is implemented for both dialects by declaring it at the instruction's own result type.
// `vote.any`/`vote.all` have no Amdgpu mapping: without a `TargetMachine`/subtarget this file
// never learns whether the wavefront is 32 or 64 lanes, and "all lanes voted true" only has a
// definite answer once that width is known — so both are `Err(Diag)` for `Amdgpu`.
//
// `atomic`/`atomic.cas`: dialect-independent — `atomicrmw`/`cmpxchg` are target-agnostic LLVM
// IR instructions, not intrinsics. `atomic` (read-modify-write) covers `i8`/`i16`/`i32`/`i64`
// through inkwell's `build_atomicrmw`; `Min`/`Max` map to LLVM's *signed* `Min`/`Max` variants,
// matching this project's uniform signed-arithmetic convention. Floating-point RMW is `Err
// (Diag)`: LLVM's `atomicrmw` instruction itself supports `fadd`/`fsub`/`fmax`/`fmin`/`xchg` on
// float operands, but inkwell 0.9's `build_atomicrmw` wrapper is typed to accept only
// `IntValue` (its own source comments this as unimplemented), and reaching for `unsafe` FFI
// around that wrapper is out of scope for this lane. `atomic.cas` covers `i8`/`i16`/`i32`/
// `i64`/`Ty::Ptr` through `build_cmpxchg`, taking the pre-swap value (`cmpxchg`'s struct field
// 0) as the op's result, matching `basalt-ptx`'s and `basalt-x86`'s documented "raw bit
// pattern, not a reinterpreted float" semantics for `atomic.cas`. Every atomic uses sequentially
// consistent ordering — the simplest ordering that is always correct; relaxing it for
// performance is later work.
//
// # Type mapping
//
// BIR scalars map onto LLVM's own scalar types one-to-one (`i1` -> LLVM's `i1`, `i8`/`i16`/
// `i32`/`i64` likewise, `f16` -> `half`, `f32` -> `float`, `f64` -> `double`). `Ty::Ptr` maps
// to LLVM 18's opaque `ptr` type regardless of `AddrSpace` — every load/store lands in
// address space 0. Mapping BIR's five address spaces onto real numeric address-space
// annotations (NVPTX's `1`/`3`/`4`, AMDGCN's own numbering, ...) is target-machine-lowering
// territory a later task owns.
//
// # Two-and-a-half-pass function lowering
//
// Every basic block is created first (`bb<id>` naming matches BIR's own numbering directly),
// then each block's instructions and terminator are lowered in BIR block order. A `phi`'s own
// value is created as an empty placeholder at this point, so any instruction after it in
// program order can already reference it. Only once every block in the function has been
// built does a final pass fill in each phi's incoming edges — this is what lets a loop's
// back-edge phi reference a value defined in a block that had not been visited yet when the
// phi itself was created.

use std::collections::HashMap;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module as LlvmModule;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, CallSiteValue, FunctionValue, IntValue,
    PhiValue, PointerValue,
};
use inkwell::AddressSpace;
use inkwell::{AtomicOrdering, AtomicRMWBinOp, FloatPredicate, IntPredicate};

use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, Module, Op,
    Scalar, ShuffleKind, Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};

/// Which GPU intrinsic dialect a module's `tid.x`/`barrier`/`shuffle.*`/`ballot`/`vote.*`
/// ops lower into. `llvm.nvvm.*` and `llvm.amdgcn.*` intrinsics are mutually exclusive within
/// one LLVM module (they are tied to whichever target machine eventually compiles the IR), so
/// this is a lowering-time choice, not a per-op one — see the module header for exactly what
/// each dialect supports. A module with no GPU ops in it ignores this parameter entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDialect {
    Nvptx,
    Amdgpu,
}

/// The `ctx`/`llvm_mod`/`builder`/`dialect` quartet every instruction-lowering function needs
/// at hand, bundled so passing it around doesn't blow out each function's own argument count.
/// Plain borrows plus a `Copy` tag, so this is itself `Copy` and passed by value throughout.
#[derive(Clone, Copy)]
struct LowerCtx<'ctx, 'a> {
    ctx: &'ctx Context,
    llvm_mod: &'a LlvmModule<'ctx>,
    builder: &'a Builder<'ctx>,
    dialect: GpuDialect,
}

/// Builds an `inkwell::module::Module` from a BIR `Module`. Every function's blocks and
/// instructions lower one to one; anything outside this file's documented scope comes back
/// as `Err(Diag)` rather than a guess. `dialect` selects which GPU intrinsic family a
/// module's GPU ops (if any) lower into.
pub fn lower_module<'ctx>(
    module: &Module,
    llvm_ctx: &'ctx Context,
    dialect: GpuDialect,
) -> Result<LlvmModule<'ctx>, Diag> {
    let llvm_mod = llvm_ctx.create_module("basalt");
    let builder = llvm_ctx.create_builder();
    let cx = LowerCtx {
        ctx: llvm_ctx,
        llvm_mod: &llvm_mod,
        builder: &builder,
        dialect,
    };
    for func in &module.funcs {
        lower_function(cx, func)?;
    }
    Ok(llvm_mod)
}

fn scalar_llvm_ty(ctx: &Context, s: Scalar) -> BasicTypeEnum<'_> {
    match s {
        Scalar::I1 => ctx.bool_type().into(),
        Scalar::I8 => ctx.i8_type().into(),
        Scalar::I16 => ctx.i16_type().into(),
        Scalar::I32 => ctx.i32_type().into(),
        Scalar::I64 => ctx.i64_type().into(),
        Scalar::F16 => ctx.f16_type().into(),
        Scalar::F32 => ctx.f32_type().into(),
        Scalar::F64 => ctx.f64_type().into(),
    }
}

/// Maps a BIR `Ty` that must denote a real (non-void) value onto its LLVM basic type.
fn basic_ty<'ctx>(ctx: &'ctx Context, ty: Ty) -> Result<BasicTypeEnum<'ctx>, Diag> {
    match ty {
        Ty::Scalar(s) => Ok(scalar_llvm_ty(ctx, s)),
        // Opaque pointer, LLVM 18's default; see the module header for why `AddrSpace` is
        // not carried through to a numeric LLVM address space in this task.
        Ty::Ptr(_) => Ok(ctx.ptr_type(AddressSpace::default()).into()),
        Ty::Vec(..) => {
            Err(Diag::new(ECode::UnsupportedType).with_arg("vector type (out of scope for T1)"))
        }
        Ty::Void => Err(Diag::new(ECode::UnsupportedType)
            .with_arg("void is only valid as a function return type")),
    }
}

fn icmp_pred(p: ICmpPred) -> IntPredicate {
    match p {
        ICmpPred::Eq => IntPredicate::EQ,
        ICmpPred::Ne => IntPredicate::NE,
        ICmpPred::Slt => IntPredicate::SLT,
        ICmpPred::Sle => IntPredicate::SLE,
        ICmpPred::Sgt => IntPredicate::SGT,
        ICmpPred::Sge => IntPredicate::SGE,
        ICmpPred::Ult => IntPredicate::ULT,
        ICmpPred::Ule => IntPredicate::ULE,
        ICmpPred::Ugt => IntPredicate::UGT,
        ICmpPred::Uge => IntPredicate::UGE,
    }
}

fn fcmp_pred(p: FCmpPred) -> FloatPredicate {
    match p {
        FCmpPred::Oeq => FloatPredicate::OEQ,
        FCmpPred::One => FloatPredicate::ONE,
        FCmpPred::Olt => FloatPredicate::OLT,
        FCmpPred::Ole => FloatPredicate::OLE,
        FCmpPred::Ogt => FloatPredicate::OGT,
        FCmpPred::Oge => FloatPredicate::OGE,
        FCmpPred::Ord => FloatPredicate::ORD,
        FCmpPred::Uno => FloatPredicate::UNO,
    }
}

fn get_val<'ctx>(
    params: &[BasicValueEnum<'ctx>],
    values: &[Option<BasicValueEnum<'ctx>>],
    v: ValRef,
) -> BasicValueEnum<'ctx> {
    match v {
        ValRef::Param(i) => params[i as usize],
        ValRef::Val(id) => values[id.0 as usize]
            .expect("operand instruction not yet lowered (BIR dominance invariant violated)"),
    }
}

/// `basalt-sema`'s own address space tags a slot's identity, not the identity of the
/// discriminant value — this is only ever used as a `HashMap` key alongside the slot's
/// integer id, so any injective mapping onto a hashable type works. `AddrSpace` itself does
/// not derive `Hash`.
fn space_tag(space: AddrSpace) -> u8 {
    match space {
        AddrSpace::Global => 0,
        AddrSpace::Shared => 1,
        AddrSpace::Constant => 2,
        AddrSpace::Local => 3,
        AddrSpace::Param => 4,
    }
}

/// Whether `space` is one of `basalt-sema`'s synthesized address spaces, whose `const.i
/// ptr.<space> N` values are opaque per-function slot keys rather than real addresses — see
/// `build_local_slots`. `Global` pointers are real addresses from the moment they arrive (a
/// function parameter, or arithmetic on one) and never take this path.
fn is_local_like(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Shared | AddrSpace::Constant
    )
}

/// `basalt-sema`'s lowering has no `alloca`: every local/parameter/shared/constant storage
/// location is handed a small integer slot id and materialized, wherever BIR needs its
/// address, as `const.i ptr.<space> (slot_id * 65536)` — an opaque per-`(space, id)` key, not
/// a real address (`basalt-x86`'s oracle documents and handles the identical pattern in its
/// own module header). This builds one real `alloca` per distinct key used in `f`, in `f`'s
/// entry block, so `lower_inst`'s `Op::ConstInt` case has genuine backing storage to hand
/// back instead of treating the opaque key as a literal pointer value (which is simply wrong
/// on every target: address `0`, `65536`, `131072`, ... are not valid addresses anywhere).
/// Every slot is a flat 8 bytes, mirroring the oracle's own "uniformly 8 bytes regardless of
/// the value's real width" scope limit; a local/shared aggregate larger than that is out of
/// scope for this lowering, matching the oracle.
fn build_local_slots<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    f: &Function,
) -> HashMap<(u8, i64), PointerValue<'ctx>> {
    let mut slots = HashMap::new();
    for inst in &f.insts {
        if let (Op::ConstInt(n), Ty::Ptr(space)) = (&inst.op, inst.ty) {
            if is_local_like(space) {
                slots.entry((space_tag(space), *n)).or_insert_with(|| {
                    builder
                        .build_alloca(ctx.i64_type(), "")
                        .expect("valid alloca")
                });
            }
        }
    }
    slots
}

/// LLVM's numeric id for the `amdgpu_kernel` calling convention (`CallingConv::AMDGPU_KERNEL`
/// in `llvm/IR/CallingConv.h`), not exposed as a named constant by inkwell 0.9.
const AMDGPU_KERNEL_CALL_CONV: u32 = 91;

fn lower_function<'ctx>(cx: LowerCtx<'ctx, '_>, f: &Function) -> Result<(), Diag> {
    let LowerCtx {
        ctx,
        llvm_mod,
        builder,
        dialect,
    } = cx;
    let param_tys: Vec<BasicMetadataTypeEnum> = f
        .params
        .iter()
        .map(|&t| basic_ty(ctx, t).map(BasicMetadataTypeEnum::from))
        .collect::<Result<_, _>>()?;

    let fn_ty = match f.ret {
        Ty::Void => ctx.void_type().fn_type(&param_tys, false),
        ret => basic_ty(ctx, ret)?.fn_type(&param_tys, false),
    };
    let llvm_fn: FunctionValue<'ctx> = llvm_mod.add_function(&f.name, fn_ty, None);
    // Every BIR function reaching this lowering is a kernel entry point (there is no
    // separate device-function concept yet — `basalt-ptx` makes the same assumption, always
    // emitting `.visible .entry`). On the Amdgpu dialect this must be spelled out explicitly:
    // without the `amdgpu_kernel` calling convention, LLVM's AMDGPU backend lowers the
    // function as an ordinary subroutine — no kernel descriptor, no kernarg ABI, an empty
    // `amdhsa.kernels` metadata array — which is structurally valid ELF but not a
    // dispatchable HSA kernel. LLVM's verifier rejects `amdgpu_kernel` on anything but a
    // void-returning function, which every real `__global__` kernel is (the source language
    // itself disallows a non-void `__global__`) — this file's own non-void test fixtures
    // exist only to exercise a single op's lowering in isolation and were never meant to be
    // dispatchable kernels, so they correctly fall back to the plain calling convention.
    if dialect == GpuDialect::Amdgpu && f.ret == Ty::Void {
        llvm_fn.set_call_conventions(AMDGPU_KERNEL_CALL_CONV);
    }

    let params: Vec<BasicValueEnum<'ctx>> = (0..f.params.len() as u32)
        .map(|i| {
            llvm_fn
                .get_nth_param(i)
                .expect("declared parameter count matches BIR signature")
        })
        .collect();

    let blocks: Vec<BasicBlock<'ctx>> = (0..f.blocks.len())
        .map(|i| ctx.append_basic_block(llvm_fn, &format!("bb{i}")))
        .collect();

    // Every local/param/shared/constant slot's `alloca` lives at the top of the entry block,
    // ahead of that block's own instructions, so it dominates every use regardless of which
    // block references it later.
    builder.position_at_end(blocks[0]);
    let local_slots = build_local_slots(ctx, builder, f);

    let mut values: Vec<Option<BasicValueEnum<'ctx>>> = vec![None; f.insts.len()];
    let mut phi_fixups: Vec<(PhiValue<'ctx>, Vec<(BlockId, ValRef)>)> = Vec::new();

    for (bi, block) in f.blocks.iter().enumerate() {
        builder.position_at_end(blocks[bi]);
        for &inst_id in &block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            let val = lower_inst(cx, &params, &values, inst, &mut phi_fixups, &local_slots)?;
            values[inst_id.0 as usize] = val;
        }
        lower_term(builder, &params, &values, &blocks, &block.term);
    }

    for (phi, incoming) in phi_fixups {
        let pairs: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> = incoming
            .iter()
            .map(|&(pred, val_ref)| (get_val(&params, &values, val_ref), blocks[pred.0 as usize]))
            .collect();
        let refs: Vec<(&dyn BasicValue<'ctx>, BasicBlock<'ctx>)> = pairs
            .iter()
            .map(|(v, b)| (v as &dyn BasicValue<'ctx>, *b))
            .collect();
        phi.add_incoming(&refs);
    }

    Ok(())
}

/// `a + offset`/`a - offset` where `a` (or, for `Add`, `b`) is a pointer: `basalt-sema`'s own
/// address computations (`base + byte_offset`, the documented mechanism behind BIR's array
/// indexing) arrive as an ordinary `Bin::Add`/`Bin::Sub` with a pointer-typed operand — BIR
/// carries no separate "pointer add" op. LLVM's opaque pointers have no arithmetic
/// instructions of their own, so this round-trips through `ptrtoint`/`inttoptr` at `i64` (the
/// same raw-address-arithmetic model `basalt-x86`'s oracle already uses for every value,
/// pointer or not) rather than the typed integer path below.
fn lower_ptr_offset<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    op: BinOp,
    ptr: inkwell::values::PointerValue<'ctx>,
    offset: IntValue<'ctx>,
) -> BasicValueEnum<'ctx> {
    let i64t = ctx.i64_type();
    let base = builder
        .build_ptr_to_int(ptr, i64t, "")
        .expect("valid ptrtoint");
    let addr = match op {
        BinOp::Add => builder.build_int_add(base, offset, "").expect("valid add"),
        BinOp::Sub => builder.build_int_sub(base, offset, "").expect("valid sub"),
        _ => unreachable!("lower_ptr_offset called with a non add/sub BinOp"),
    };
    builder
        .build_int_to_ptr(addr, ptr.get_type(), "")
        .expect("valid inttoptr")
        .into()
}

fn lower_bin<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    op: BinOp,
    a: BasicValueEnum<'ctx>,
    b: BasicValueEnum<'ctx>,
) -> BasicValueEnum<'ctx> {
    if matches!(op, BinOp::Add | BinOp::Sub) {
        if let BasicValueEnum::PointerValue(pv) = a {
            return lower_ptr_offset(ctx, builder, op, pv, b.into_int_value());
        }
        if op == BinOp::Add {
            if let BasicValueEnum::PointerValue(pv) = b {
                return lower_ptr_offset(ctx, builder, op, pv, a.into_int_value());
            }
        }
    }
    match op {
        BinOp::Add => builder
            .build_int_add(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Sub => builder
            .build_int_sub(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Mul => builder
            .build_int_mul(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        // Signed division/remainder: BIR's `Bin` carries no signed/unsigned distinction for
        // these, so this lane takes the same documented signed-first stance as the x86
        // oracle and the PTX emitter rather than inventing an unsigned form BIR cannot ask
        // for.
        BinOp::Div => builder
            .build_int_signed_div(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Rem => builder
            .build_int_signed_rem(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::FAdd => builder
            .build_float_add(a.into_float_value(), b.into_float_value(), "")
            .expect("valid float operands")
            .into(),
        BinOp::FSub => builder
            .build_float_sub(a.into_float_value(), b.into_float_value(), "")
            .expect("valid float operands")
            .into(),
        BinOp::FMul => builder
            .build_float_mul(a.into_float_value(), b.into_float_value(), "")
            .expect("valid float operands")
            .into(),
        BinOp::FDiv => builder
            .build_float_div(a.into_float_value(), b.into_float_value(), "")
            .expect("valid float operands")
            .into(),
        BinOp::FRem => builder
            .build_float_rem(a.into_float_value(), b.into_float_value(), "")
            .expect("valid float operands")
            .into(),
        BinOp::And => builder
            .build_and(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Or => builder
            .build_or(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Xor => builder
            .build_xor(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Shl => builder
            .build_left_shift(a.into_int_value(), b.into_int_value(), "")
            .expect("valid int operands")
            .into(),
        BinOp::Lshr => builder
            .build_right_shift(a.into_int_value(), b.into_int_value(), false, "")
            .expect("valid int operands")
            .into(),
        BinOp::Ashr => builder
            .build_right_shift(a.into_int_value(), b.into_int_value(), true, "")
            .expect("valid int operands")
            .into(),
    }
}

fn lower_cast<'ctx>(
    builder: &Builder<'ctx>,
    op: CastOp,
    dst_ty: BasicTypeEnum<'ctx>,
    src: BasicValueEnum<'ctx>,
) -> BasicValueEnum<'ctx> {
    match op {
        CastOp::Trunc => builder
            .build_int_truncate(src.into_int_value(), dst_ty.into_int_type(), "")
            .expect("valid truncating cast")
            .into(),
        CastOp::Zext => builder
            .build_int_z_extend(src.into_int_value(), dst_ty.into_int_type(), "")
            .expect("valid zero-extending cast")
            .into(),
        CastOp::Sext => builder
            .build_int_s_extend(src.into_int_value(), dst_ty.into_int_type(), "")
            .expect("valid sign-extending cast")
            .into(),
        CastOp::FpTrunc => builder
            .build_float_trunc(src.into_float_value(), dst_ty.into_float_type(), "")
            .expect("valid float-truncating cast")
            .into(),
        CastOp::FpExt => builder
            .build_float_ext(src.into_float_value(), dst_ty.into_float_type(), "")
            .expect("valid float-extending cast")
            .into(),
        CastOp::FpToSi => builder
            .build_float_to_signed_int(src.into_float_value(), dst_ty.into_int_type(), "")
            .expect("valid float-to-signed-int cast")
            .into(),
        CastOp::FpToUi => builder
            .build_float_to_unsigned_int(src.into_float_value(), dst_ty.into_int_type(), "")
            .expect("valid float-to-unsigned-int cast")
            .into(),
        CastOp::SiToFp => builder
            .build_signed_int_to_float(src.into_int_value(), dst_ty.into_float_type(), "")
            .expect("valid signed-int-to-float cast")
            .into(),
        CastOp::UiToFp => builder
            .build_unsigned_int_to_float(src.into_int_value(), dst_ty.into_float_type(), "")
            .expect("valid unsigned-int-to-float cast")
            .into(),
        CastOp::Bitcast => {
            // Opaque pointers make a same-type "cast" (e.g. `ptr.global` -> `ptr.shared`,
            // both LLVM `ptr`) an identity: LLVM's own `bitcast` instruction requires source
            // and destination types to differ, so pass the value through unchanged rather
            // than emit an instruction LLVM would reject.
            if src.get_type() == dst_ty {
                src
            } else {
                builder
                    .build_bit_cast(src, dst_ty, "")
                    .expect("valid bitcast")
            }
        }
    }
}

// ---- GPU intrinsics ---------------------------------------------------------------------
//
// See the module header for exactly which ops/dialects are supported and why the gaps are
// gaps. `declare_intrinsic`/`call_intrinsic` do the `Intrinsic::find` + `get_declaration` +
// `build_call` dance every op below needs; a missing intrinsic there is a mismatch between
// this file's documented names and the LLVM this crate was built against, not a normal BIR
// lowering failure, so it panics like this file's other "should never happen given a
// well-formed module" invariants rather than threading another `Result` layer through.

fn declare_intrinsic<'ctx>(
    llvm_mod: &LlvmModule<'ctx>,
    name: &str,
    overload_tys: &[BasicTypeEnum<'ctx>],
) -> FunctionValue<'ctx> {
    Intrinsic::find(name)
        .and_then(|intr| intr.get_declaration(llvm_mod, overload_tys))
        .unwrap_or_else(|| panic!("intrinsic `{name}` not available in this LLVM build"))
}

fn call_intrinsic<'ctx>(
    llvm_mod: &LlvmModule<'ctx>,
    builder: &Builder<'ctx>,
    name: &str,
    overload_tys: &[BasicTypeEnum<'ctx>],
    args: &[BasicMetadataValueEnum<'ctx>],
) -> CallSiteValue<'ctx> {
    let f = declare_intrinsic(llvm_mod, name, overload_tys);
    builder
        .build_call(f, args, "")
        .expect("valid intrinsic call")
}

fn adapt_int_width<'ctx>(
    builder: &Builder<'ctx>,
    v: IntValue<'ctx>,
    dst: IntType<'ctx>,
) -> IntValue<'ctx> {
    let (sw, dw) = (v.get_type().get_bit_width(), dst.get_bit_width());
    match sw.cmp(&dw) {
        std::cmp::Ordering::Equal => v,
        std::cmp::Ordering::Less => builder.build_int_z_extend(v, dst, "").expect("valid zext"),
        std::cmp::Ordering::Greater => builder.build_int_truncate(v, dst, "").expect("valid trunc"),
    }
}

/// Predicate operands to `ballot`/`vote.any`/`vote.all` arrive as a truthy `int` (sema coerces
/// a boolean expression to `int`, never to a bare `i1`), so every dialect needs a real `i1`
/// before it can feed a vote/ballot intrinsic. Already-`i1` operands pass through unchanged.
fn to_i1_pred<'ctx>(builder: &Builder<'ctx>, v: BasicValueEnum<'ctx>) -> IntValue<'ctx> {
    let iv = v.into_int_value();
    if iv.get_type().get_bit_width() == 1 {
        iv
    } else {
        let zero = iv.get_type().const_zero();
        builder
            .build_int_compare(IntPredicate::NE, iv, zero, "")
            .expect("valid predicate compare")
    }
}

fn lower_gpu_index<'ctx>(cx: LowerCtx<'ctx, '_>, op: &Op) -> Result<BasicValueEnum<'ctx>, Diag> {
    let LowerCtx {
        llvm_mod,
        builder,
        dialect,
        ..
    } = cx;
    let name = match (dialect, op) {
        (GpuDialect::Nvptx, Op::TidX) => "llvm.nvvm.read.ptx.sreg.tid.x",
        (GpuDialect::Nvptx, Op::TidY) => "llvm.nvvm.read.ptx.sreg.tid.y",
        (GpuDialect::Nvptx, Op::TidZ) => "llvm.nvvm.read.ptx.sreg.tid.z",
        (GpuDialect::Nvptx, Op::BidX) => "llvm.nvvm.read.ptx.sreg.ctaid.x",
        (GpuDialect::Nvptx, Op::BidY) => "llvm.nvvm.read.ptx.sreg.ctaid.y",
        (GpuDialect::Nvptx, Op::BidZ) => "llvm.nvvm.read.ptx.sreg.ctaid.z",
        (GpuDialect::Nvptx, Op::BdimX) => "llvm.nvvm.read.ptx.sreg.ntid.x",
        (GpuDialect::Nvptx, Op::BdimY) => "llvm.nvvm.read.ptx.sreg.ntid.y",
        (GpuDialect::Nvptx, Op::BdimZ) => "llvm.nvvm.read.ptx.sreg.ntid.z",
        (GpuDialect::Nvptx, Op::GdimX) => "llvm.nvvm.read.ptx.sreg.nctaid.x",
        (GpuDialect::Nvptx, Op::GdimY) => "llvm.nvvm.read.ptx.sreg.nctaid.y",
        (GpuDialect::Nvptx, Op::GdimZ) => "llvm.nvvm.read.ptx.sreg.nctaid.z",
        (GpuDialect::Amdgpu, Op::TidX) => "llvm.amdgcn.workitem.id.x",
        (GpuDialect::Amdgpu, Op::TidY) => "llvm.amdgcn.workitem.id.y",
        (GpuDialect::Amdgpu, Op::TidZ) => "llvm.amdgcn.workitem.id.z",
        (GpuDialect::Amdgpu, Op::BidX) => "llvm.amdgcn.workgroup.id.x",
        (GpuDialect::Amdgpu, Op::BidY) => "llvm.amdgcn.workgroup.id.y",
        (GpuDialect::Amdgpu, Op::BidZ) => "llvm.amdgcn.workgroup.id.z",
        (
            GpuDialect::Amdgpu,
            Op::BdimX | Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ,
        ) => {
            return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                "block/grid dimension on the amdgpu dialect: no no-argument LLVM 18 \
                 intrinsic exists (would need an HSA-ABI-version-specific field load out of \
                 llvm.amdgcn.dispatch.ptr, which this lane is not confident enough in to emit)",
            ));
        }
        _ => unreachable!("lower_gpu_index called with a non-index op"),
    };
    Ok(call_intrinsic(llvm_mod, builder, name, &[], &[])
        .try_as_basic_value()
        .expect_basic("index intrinsic returns i32"))
}

fn lower_barrier<'ctx>(cx: LowerCtx<'ctx, '_>) {
    let LowerCtx {
        llvm_mod,
        builder,
        dialect,
        ..
    } = cx;
    let name = match dialect {
        GpuDialect::Nvptx => "llvm.nvvm.barrier0",
        GpuDialect::Amdgpu => "llvm.amdgcn.s.barrier",
    };
    call_intrinsic(llvm_mod, builder, name, &[], &[]);
}

fn lower_shuffle<'ctx>(
    cx: LowerCtx<'ctx, '_>,
    kind: ShuffleKind,
    val: BasicValueEnum<'ctx>,
    amt: BasicValueEnum<'ctx>,
    ty: Ty,
) -> Result<BasicValueEnum<'ctx>, Diag> {
    let LowerCtx {
        ctx,
        llvm_mod,
        builder,
        dialect,
    } = cx;
    if dialect == GpuDialect::Amdgpu {
        return Err(Diag::new(ECode::UnsupportedOp).with_arg(
            "shuffle on the amdgpu dialect: ds_bpermute/ds_permute move data by byte-addressed \
             lane offset rather than BIR's idx/up/down/xor kinds, and up/down clamp behavior \
             differs between wave32 and wave64 parts, so there is no settled mapping here",
        ));
    }
    let mode_suffix = match kind {
        ShuffleKind::Idx => "idx",
        ShuffleKind::Up => "up",
        ShuffleKind::Down => "down",
        ShuffleKind::Xor => "bfly",
    };
    // Segment mask/clamp operand: `0x1f` is "whole warp"; `up` clamps at the lowest source
    // lane (`0`) rather than the highest, matching `basalt-ptx`'s own documented convention.
    let clamp = if kind == ShuffleKind::Up { 0x0 } else { 0x1f };
    let name = format!("llvm.nvvm.shfl.sync.{mode_suffix}.i32");
    let i32t = ctx.i32_type();
    let mask = i32t.const_int(0xffffffff, false);
    let amt_i32 = amt.into_int_value();
    let clamp_v = i32t.const_int(clamp, false);
    let raw_val = match ty {
        Ty::Scalar(Scalar::I32) => val.into_int_value(),
        Ty::Scalar(Scalar::F32) => builder
            .build_bit_cast(val, i32t, "")
            .expect("valid bitcast")
            .into_int_value(),
        _ => {
            return Err(Diag::new(ECode::UnsupportedType).with_arg(
                "shuffle on the nvptx dialect is implemented only for 32-bit-wide scalar \
                 values (i32/f32) in this lane; wider/narrower widths need the multi-word \
                 split basalt-ptx does, which this lane does not yet replicate",
            ))
        }
    };
    let args: [BasicMetadataValueEnum; 4] =
        [mask.into(), raw_val.into(), amt_i32.into(), clamp_v.into()];
    let result = call_intrinsic(llvm_mod, builder, &name, &[], &args)
        .try_as_basic_value()
        .expect_basic("shfl.sync intrinsic returns i32");
    Ok(match ty {
        Ty::Scalar(Scalar::F32) => builder
            .build_bit_cast(result, ctx.f32_type(), "")
            .expect("valid bitcast"),
        _ => result,
    })
}

fn lower_ballot<'ctx>(
    cx: LowerCtx<'ctx, '_>,
    pred: BasicValueEnum<'ctx>,
    dst_ty: BasicTypeEnum<'ctx>,
) -> BasicValueEnum<'ctx> {
    let LowerCtx {
        ctx,
        llvm_mod,
        builder,
        dialect,
    } = cx;
    let p = to_i1_pred(builder, pred);
    let raw = match dialect {
        GpuDialect::Nvptx => {
            let mask = ctx.i32_type().const_int(0xffffffff, false);
            call_intrinsic(
                llvm_mod,
                builder,
                "llvm.nvvm.vote.ballot.sync",
                &[],
                &[mask.into(), p.into()],
            )
            .try_as_basic_value()
            .expect_basic("vote.ballot.sync returns i32")
            .into_int_value()
        }
        GpuDialect::Amdgpu => {
            // Overloaded on its own return width, so declaring it at `dst_ty` sidesteps the
            // wave32-vs-wave64 question entirely: the lane mask comes back exactly as wide as
            // BIR asked for.
            call_intrinsic(
                llvm_mod,
                builder,
                "llvm.amdgcn.ballot",
                &[dst_ty],
                &[p.into()],
            )
            .try_as_basic_value()
            .expect_basic("amdgcn.ballot returns an integer lane mask")
            .into_int_value()
        }
    };
    adapt_int_width(builder, raw, dst_ty.into_int_type()).into()
}

fn lower_vote<'ctx>(
    cx: LowerCtx<'ctx, '_>,
    all: bool,
    pred: BasicValueEnum<'ctx>,
    dst_ty: BasicTypeEnum<'ctx>,
) -> Result<BasicValueEnum<'ctx>, Diag> {
    let LowerCtx {
        ctx,
        llvm_mod,
        builder,
        dialect,
    } = cx;
    if dialect == GpuDialect::Amdgpu {
        let mnemonic = if all { "vote.all" } else { "vote.any" };
        return Err(Diag::new(ECode::UnsupportedOp).with_arg(format!(
            "{mnemonic} on the amdgpu dialect: \"every/any lane voted true\" only has a \
             definite answer once the wavefront width (32 or 64) is known, and this file \
             builds plain LLVM IR with no TargetMachine/subtarget to learn that from"
        )));
    }
    let p = to_i1_pred(builder, pred);
    let mask = ctx.i32_type().const_int(0xffffffff, false);
    let name = if all {
        "llvm.nvvm.vote.all.sync"
    } else {
        "llvm.nvvm.vote.any.sync"
    };
    let raw = call_intrinsic(llvm_mod, builder, name, &[], &[mask.into(), p.into()])
        .try_as_basic_value()
        .expect_basic("vote.{any,all}.sync returns i1")
        .into_int_value();
    Ok(adapt_int_width(builder, raw, dst_ty.into_int_type()).into())
}

fn atomic_rmw_binop(op: AtomicOp) -> AtomicRMWBinOp {
    match op {
        AtomicOp::Add => AtomicRMWBinOp::Add,
        AtomicOp::Sub => AtomicRMWBinOp::Sub,
        AtomicOp::Exch => AtomicRMWBinOp::Xchg,
        // Signed, matching this project's uniform signed-arithmetic convention (see the
        // module header and `lower_bin`'s `Div`/`Rem`).
        AtomicOp::Min => AtomicRMWBinOp::Min,
        AtomicOp::Max => AtomicRMWBinOp::Max,
        AtomicOp::And => AtomicRMWBinOp::And,
        AtomicOp::Or => AtomicRMWBinOp::Or,
        AtomicOp::Xor => AtomicRMWBinOp::Xor,
    }
}

fn lower_atomic<'ctx>(
    builder: &Builder<'ctx>,
    op: AtomicOp,
    ptr: BasicValueEnum<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: Ty,
) -> Result<BasicValueEnum<'ctx>, Diag> {
    match ty {
        Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64) => Ok(builder
            .build_atomicrmw(
                atomic_rmw_binop(op),
                ptr.into_pointer_value(),
                val.into_int_value(),
                AtomicOrdering::SequentiallyConsistent,
            )
            .expect("valid atomicrmw")
            .into()),
        _ => Err(Diag::new(ECode::UnsupportedType).with_arg(
            "atomic rmw on this type: LLVM's atomicrmw instruction itself supports \
             float operands (fadd/fsub/fmax/fmin/xchg), but inkwell 0.9's build_atomicrmw \
             wrapper only accepts an IntValue, and this lane does not reach for unsafe FFI \
             to work around that",
        )),
    }
}

fn lower_atomic_cas<'ctx>(
    builder: &Builder<'ctx>,
    ptr: BasicValueEnum<'ctx>,
    cmp: BasicValueEnum<'ctx>,
    newv: BasicValueEnum<'ctx>,
    ty: Ty,
) -> Result<BasicValueEnum<'ctx>, Diag> {
    match ty {
        Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64) | Ty::Ptr(_) => {
            let swap = builder
                .build_cmpxchg(
                    ptr.into_pointer_value(),
                    cmp,
                    newv,
                    AtomicOrdering::SequentiallyConsistent,
                    AtomicOrdering::SequentiallyConsistent,
                )
                .expect("valid cmpxchg");
            Ok(builder
                .build_extract_value(swap, 0, "")
                .expect("cmpxchg's result struct has the pre-swap value at field 0"))
        }
        _ => Err(Diag::new(ECode::UnsupportedType).with_arg(
            "atomic.cas on this type: only integer/pointer compare-and-swap is implemented \
             in this lane",
        )),
    }
}

fn lower_inst<'ctx>(
    cx: LowerCtx<'ctx, '_>,
    params: &[BasicValueEnum<'ctx>],
    values: &[Option<BasicValueEnum<'ctx>>],
    inst: &Inst,
    phi_fixups: &mut Vec<(PhiValue<'ctx>, Vec<(BlockId, ValRef)>)>,
    local_slots: &HashMap<(u8, i64), PointerValue<'ctx>>,
) -> Result<Option<BasicValueEnum<'ctx>>, Diag> {
    let LowerCtx { ctx, builder, .. } = cx;
    let val = match &inst.op {
        // `basalt-sema`'s own opaque local/param/shared/constant slot key (see
        // `build_local_slots`): hand back that slot's real `alloca`, never a literal
        // `inttoptr` of the key itself.
        Op::ConstInt(v) if matches!(inst.ty, Ty::Ptr(space) if is_local_like(space)) => {
            let space = match inst.ty {
                Ty::Ptr(space) => space,
                _ => unreachable!("guarded by the match arm above"),
            };
            let ptr = *local_slots
                .get(&(space_tag(space), *v))
                .expect("build_local_slots pre-scans every local-like slot constant");
            Some(ptr.into())
        }
        // A literal `ptr.global` address: real global pointers always arrive as function
        // parameters or arithmetic on one (see this file's own header), so this is an
        // unusual case in practice (e.g. a null pointer), handled with a genuine constant
        // `inttoptr` since LLVM has no integer-literal pointer constant of its own.
        Op::ConstInt(v) if matches!(inst.ty, Ty::Ptr(_)) => {
            let ptrty = basic_ty(ctx, inst.ty)?.into_pointer_type();
            let iv = ctx.i64_type().const_int(*v as u64, true);
            Some(iv.const_to_pointer(ptrty).into())
        }
        Op::ConstInt(v) => {
            let ity = basic_ty(ctx, inst.ty)?.into_int_type();
            Some(ity.const_int(*v as u64, true).into())
        }
        Op::ConstFloat(v) => {
            let fty = basic_ty(ctx, inst.ty)?.into_float_type();
            Some(fty.const_float(*v).into())
        }
        Op::Bin(op, a, b) => {
            let av = get_val(params, values, *a);
            let bv = get_val(params, values, *b);
            Some(lower_bin(ctx, builder, *op, av, bv))
        }
        Op::ICmp(pred, _ty, a, b) => {
            let av = get_val(params, values, *a).into_int_value();
            let bv = get_val(params, values, *b).into_int_value();
            Some(
                builder
                    .build_int_compare(icmp_pred(*pred), av, bv, "")
                    .expect("valid icmp operands")
                    .into(),
            )
        }
        Op::FCmp(pred, _ty, a, b) => {
            let av = get_val(params, values, *a).into_float_value();
            let bv = get_val(params, values, *b).into_float_value();
            Some(
                builder
                    .build_float_compare(fcmp_pred(*pred), av, bv, "")
                    .expect("valid fcmp operands")
                    .into(),
            )
        }
        Op::Select(c, a, b) => {
            let cv = get_val(params, values, *c).into_int_value();
            let av = get_val(params, values, *a);
            let bv = get_val(params, values, *b);
            Some(
                builder
                    .build_select(cv, av, bv, "")
                    .expect("valid select operands"),
            )
        }
        Op::Cast(op, _src_ty, v) => {
            let sv = get_val(params, values, *v);
            let dty = basic_ty(ctx, inst.ty)?;
            Some(lower_cast(builder, *op, dty, sv))
        }
        Op::Load { ptr, .. } => {
            let pv = get_val(params, values, *ptr).into_pointer_value();
            let ldty = basic_ty(ctx, inst.ty)?;
            Some(builder.build_load(ldty, pv, "").expect("valid load"))
        }
        Op::Store { ptr, val, .. } => {
            let pv = get_val(params, values, *ptr).into_pointer_value();
            let vv = get_val(params, values, *val);
            builder.build_store(pv, vv).expect("valid store");
            None
        }
        Op::Phi(incoming) => {
            let pty = basic_ty(ctx, inst.ty)?;
            let phi = builder.build_phi(pty, "").expect("valid phi");
            phi_fixups.push((phi, incoming.clone()));
            Some(phi.as_basic_value())
        }
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
        | Op::GdimZ => Some(lower_gpu_index(cx, &inst.op)?),
        Op::Barrier => {
            lower_barrier(cx);
            None
        }
        Op::Shuffle(kind, val, amt) => {
            let vv = get_val(params, values, *val);
            let av = get_val(params, values, *amt);
            Some(lower_shuffle(cx, *kind, vv, av, inst.ty)?)
        }
        Op::Ballot(v) => {
            let pv = get_val(params, values, *v);
            let dst_ty = basic_ty(ctx, inst.ty)?;
            Some(lower_ballot(cx, pv, dst_ty))
        }
        Op::VoteAny(v) => {
            let pv = get_val(params, values, *v);
            let dst_ty = basic_ty(ctx, inst.ty)?;
            Some(lower_vote(cx, false, pv, dst_ty)?)
        }
        Op::VoteAll(v) => {
            let pv = get_val(params, values, *v);
            let dst_ty = basic_ty(ctx, inst.ty)?;
            Some(lower_vote(cx, true, pv, dst_ty)?)
        }
        Op::Atomic(op, ptr, val, _space) => {
            let pv = get_val(params, values, *ptr);
            let vv = get_val(params, values, *val);
            Some(lower_atomic(builder, *op, pv, vv, inst.ty)?)
        }
        Op::AtomicCas(ptr, cmp, newv, _space) => {
            let pv = get_val(params, values, *ptr);
            let cv = get_val(params, values, *cmp);
            let nv = get_val(params, values, *newv);
            Some(lower_atomic_cas(builder, pv, cv, nv, inst.ty)?)
        }
        // `wmma`/`mma.sync` intrinsic lowering is separate, later work — refuse cleanly
        // rather than guess at a mapping.
        Op::Mma { .. } => {
            return Err(Diag::new(ECode::UnsupportedOp)
                .with_arg("mma has no LLVM IR lowering in this lane yet"))
        }
    };
    Ok(val)
}

fn lower_term<'ctx>(
    builder: &Builder<'ctx>,
    params: &[BasicValueEnum<'ctx>],
    values: &[Option<BasicValueEnum<'ctx>>],
    blocks: &[BasicBlock<'ctx>],
    term: &Term,
) {
    match term {
        Term::Br(b) => {
            builder
                .build_unconditional_branch(blocks[b.0 as usize])
                .expect("valid br");
        }
        Term::CondBr(c, t, e) => {
            let cv = get_val(params, values, *c).into_int_value();
            builder
                .build_conditional_branch(cv, blocks[t.0 as usize], blocks[e.0 as usize])
                .expect("valid condbr");
        }
        Term::Switch(scrutinee, default, cases) => {
            let sv = get_val(params, values, *scrutinee).into_int_value();
            let ity = sv.get_type();
            let case_pairs: Vec<(IntValue<'ctx>, BasicBlock<'ctx>)> = cases
                .iter()
                .map(|&(v, bid)| (ity.const_int(v as u64, true), blocks[bid.0 as usize]))
                .collect();
            builder
                .build_switch(sv, blocks[default.0 as usize], &case_pairs)
                .expect("valid switch");
        }
        Term::Ret(None) => {
            builder.build_return(None).expect("valid ret");
        }
        Term::Ret(Some(v)) => {
            let rv = get_val(params, values, *v);
            builder
                .build_return(Some(&rv as &dyn BasicValue<'ctx>))
                .expect("valid ret");
        }
    }
}

#[cfg(test)]
mod tests;
