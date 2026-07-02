// BIR -> LLVM IR lowering: core scalar/memory/control-flow ops only. This builds an
// in-memory `inkwell::module::Module` and nothing else — no `TargetMachine`, no object/
// bitcode emission. That is later work layered on top of `lower_module`.
//
// # Scope
//
// Implemented: `const.i`/`const.f`, `Bin` (signed `div`/`rem`, matching the convention
// already established by `basalt-x86`'s oracle and `basalt-ptx`'s emitter — BIR's `Bin`
// carries no signed/unsigned distinction, so this lane makes the same documented choice),
// `icmp`/`fcmp`, `select`, every `CastOp` variant, `load`/`store`, `phi`, and every `Term`
// variant.
//
// Refused with `Err(Diag)` rather than guessed at: the twelve GPU index intrinsics
// (`tid.x`..`gdim.z`), `barrier`, `shuffle.*`, `ballot`/`vote.*`, `atomic`/`atomic.cas`, and
// `Ty::Vec`. These need target-specific intrinsic mappings (NVPTX/AMDGCN) that a later task
// owns.
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

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module as LlvmModule;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PhiValue};
use inkwell::AddressSpace;
use inkwell::{FloatPredicate, IntPredicate};

use basalt_bir::{
    BinOp, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, Module, Op, Scalar, Term, Ty,
    ValRef,
};
use basalt_diag::{Diag, ECode};

/// Builds an `inkwell::module::Module` from a BIR `Module`. Every function's blocks and
/// instructions lower one to one; anything outside this file's documented scope comes back
/// as `Err(Diag)` rather than a guess.
pub fn lower_module<'ctx>(
    module: &Module,
    llvm_ctx: &'ctx Context,
) -> Result<LlvmModule<'ctx>, Diag> {
    let llvm_mod = llvm_ctx.create_module("basalt");
    let builder = llvm_ctx.create_builder();
    for func in &module.funcs {
        lower_function(llvm_ctx, &llvm_mod, &builder, func)?;
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

/// Mnemonic for a BIR op this file explicitly refuses to lower — used only to build the
/// `Diag`'s argument text, never matched on.
fn refused_op_name(op: &Op) -> &'static str {
    match op {
        Op::TidX => "tid.x",
        Op::TidY => "tid.y",
        Op::TidZ => "tid.z",
        Op::BidX => "bid.x",
        Op::BidY => "bid.y",
        Op::BidZ => "bid.z",
        Op::BdimX => "bdim.x",
        Op::BdimY => "bdim.y",
        Op::BdimZ => "bdim.z",
        Op::GdimX => "gdim.x",
        Op::GdimY => "gdim.y",
        Op::GdimZ => "gdim.z",
        Op::Barrier => "barrier",
        Op::Shuffle(..) => "shuffle",
        Op::Ballot(_) => "ballot",
        Op::VoteAny(_) => "vote.any",
        Op::VoteAll(_) => "vote.all",
        Op::Atomic(..) => "atomic",
        Op::AtomicCas(..) => "atomic.cas",
        _ => "unsupported op",
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

fn lower_function<'ctx>(
    ctx: &'ctx Context,
    llvm_mod: &LlvmModule<'ctx>,
    builder: &Builder<'ctx>,
    f: &Function,
) -> Result<(), Diag> {
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

    let mut values: Vec<Option<BasicValueEnum<'ctx>>> = vec![None; f.insts.len()];
    let mut phi_fixups: Vec<(PhiValue<'ctx>, Vec<(BlockId, ValRef)>)> = Vec::new();

    for (bi, block) in f.blocks.iter().enumerate() {
        builder.position_at_end(blocks[bi]);
        for &inst_id in &block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            let val = lower_inst(ctx, builder, &params, &values, inst, &mut phi_fixups)?;
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

fn lower_bin<'ctx>(
    builder: &Builder<'ctx>,
    op: BinOp,
    a: BasicValueEnum<'ctx>,
    b: BasicValueEnum<'ctx>,
) -> BasicValueEnum<'ctx> {
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

fn lower_inst<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    params: &[BasicValueEnum<'ctx>],
    values: &[Option<BasicValueEnum<'ctx>>],
    inst: &Inst,
    phi_fixups: &mut Vec<(PhiValue<'ctx>, Vec<(BlockId, ValRef)>)>,
) -> Result<Option<BasicValueEnum<'ctx>>, Diag> {
    let val = match &inst.op {
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
            Some(lower_bin(builder, *op, av, bv))
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
        other => {
            return Err(Diag::new(ECode::UnsupportedOp).with_arg(refused_op_name(other)));
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
