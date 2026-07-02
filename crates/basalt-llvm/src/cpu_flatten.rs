// This project's x86 oracle (`basalt-x86`) interprets a BIR kernel's thread/block-index ops
// by wrapping the whole function body in a native loop at machine-code-emission time — see
// that crate's own module header for the "SIMT-via-a-native-loop" design and exactly what
// each GPU index op means under it. `lower_module` has no equivalent: it is a straight
// one-to-one BIR-to-LLVM-IR lowering with GPU intrinsics standing in for GPU index ops,
// correct for a real NVPTX/AMDGCN target and meaningless on x86 (there is no
// `llvm.nvvm.*`/`llvm.amdgcn.*` lowering for that target).
//
// This file bridges the gap for `LlvmTarget::X86` specifically: given a BIR module that
// actually uses a GPU index op, it rewrites the module, before it ever reaches
// `lower_module`, into the exact shape the x86 oracle's own convention describes — one
// trailing `i64` `nthreads` parameter, the whole original function body wrapped in a
// `for (i = 0; i < nthreads; i++)` loop, `tid.x` reading the loop counter, `bdim.x` reading
// `nthreads`, every other GPU index op the fixed constant the oracle's own table specifies,
// and `barrier` a genuine no-op (dropped) for the same one-thread-at-a-time reason the oracle
// documents. This is what lets `emit_object(..., LlvmTarget::X86)` link against the exact
// same host shim (`examples/cpu_launch_vadd.c`) the oracle's own tests use, and is the only
// way the two independent x86 codegen paths are even comparable.
//
// A module with no GPU index op is passed through unchanged — an ordinary scalar function has
// no "threads" to loop over, and forcing every x86 module through this shape would be wrong.
//
// Scope, matching the oracle's own documented limits: exactly one `Void`-returning function,
// no warp-collective op (`shuffle`/`ballot`/`vote.*`, refused for the same reason the oracle
// refuses them: no meaning under one-thread-at-a-time execution), and `tid.x`/`bdim.x` typed
// exactly `i32` (what CUDA's own builtins always produce). Anything outside that refuses with
// a `Diag` rather than guessing.

use basalt_bir::{
    BinOp, Block, BlockId, CastOp, Function, ICmpPred, Inst, InstId, Module, Op, Scalar, Term, Ty,
    ValRef,
};
use basalt_diag::{Diag, ECode};

const I64: Ty = Ty::Scalar(Scalar::I64);
const I32: Ty = Ty::Scalar(Scalar::I32);

fn is_gpu_index(op: &Op) -> bool {
    matches!(
        op,
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
    )
}

/// Whether any function in `module` reads a thread/block-index op — the trigger for
/// `flatten_to_native_cpu_loop`. A module with none of these has no "threads" to loop over
/// and is left untouched.
pub(crate) fn uses_gpu_index_ops(module: &Module) -> bool {
    module
        .funcs
        .iter()
        .any(|f| f.insts.iter().any(|i| is_gpu_index(&i.op)))
}

/// Rewrites `module` into the one-function, loop-wrapped shape described in this file's own
/// header.
pub(crate) fn flatten_to_native_cpu_loop(module: &Module) -> Result<Module, Diag> {
    if module.funcs.len() != 1 {
        return Err(Diag::new(ECode::UnsupportedFeature)
            .with_arg("cpu-loop flattening: multi-function module"));
    }
    let f = &module.funcs[0];
    if f.ret != Ty::Void {
        return Err(Diag::new(ECode::UnsupportedType)
            .with_arg("cpu-loop flattening: only a Void-returning kernel function is supported"));
    }
    for inst in &f.insts {
        if matches!(
            inst.op,
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..)
        ) {
            return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                "cpu-loop flattening: warp-collective op has no meaning under \
                 one-thread-at-a-time execution",
            ));
        }
    }

    Ok(Module {
        funcs: vec![flatten_function(f)?],
        launch_bounds: module.launch_bounds,
        shared_mem_bytes: module.shared_mem_bytes,
        target_dtypes: module.target_dtypes.clone(),
    })
}

fn remap_val(v: ValRef, inst_offset: u32) -> ValRef {
    match v {
        ValRef::Param(i) => ValRef::Param(i),
        ValRef::Val(id) => ValRef::Val(InstId(id.0 + inst_offset)),
    }
}

fn remap_block(b: BlockId, block_offset: u32) -> BlockId {
    BlockId(b.0 + block_offset)
}

/// Rewrites one GPU index op into its native-loop equivalent, per the table in this file's
/// own header. `counter` is the loop-check block's phi (the running thread index, `i64`);
/// `nthreads_param` is the flattened function's trailing parameter index.
fn rewrite_gpu_index(op: &Op, ty: Ty, counter: InstId, nthreads_param: u32) -> Result<Op, Diag> {
    let require_i32 = |what: &str| -> Result<(), Diag> {
        if ty == I32 {
            Ok(())
        } else {
            Err(Diag::new(ECode::UnsupportedType).with_arg(format!(
                "cpu-loop flattening: `{what}` typed other than i32 is out of scope"
            )))
        }
    };
    Ok(match op {
        Op::TidX => {
            require_i32("tid.x")?;
            Op::Cast(CastOp::Trunc, I64, ValRef::Val(counter))
        }
        Op::BdimX => {
            require_i32("bdim.x")?;
            Op::Cast(CastOp::Trunc, I64, ValRef::Param(nthreads_param))
        }
        Op::TidY | Op::TidZ | Op::BidX | Op::BidY | Op::BidZ => Op::ConstInt(0),
        Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ => Op::ConstInt(1),
        _ => unreachable!("rewrite_gpu_index called with a non-index op"),
    })
}

/// Remaps every `ValRef`/`BlockId` an op carries by the given offsets. Used for every
/// instruction as it is copied into the flattened function, except GPU index ops (rewritten
/// by `rewrite_gpu_index` instead, which they never reach here).
fn remap_op(op: &Op, inst_offset: u32, block_offset: u32) -> Op {
    let rv = |v: ValRef| remap_val(v, inst_offset);
    match op {
        Op::ConstInt(n) => Op::ConstInt(*n),
        Op::ConstFloat(n) => Op::ConstFloat(*n),
        Op::Bin(o, a, b) => Op::Bin(*o, rv(*a), rv(*b)),
        Op::ICmp(p, t, a, b) => Op::ICmp(*p, *t, rv(*a), rv(*b)),
        Op::FCmp(p, t, a, b) => Op::FCmp(*p, *t, rv(*a), rv(*b)),
        Op::Select(c, a, b) => Op::Select(rv(*c), rv(*a), rv(*b)),
        Op::Cast(c, t, v) => Op::Cast(*c, *t, rv(*v)),
        Op::Load {
            ptr,
            space,
            align,
            volatile,
        } => Op::Load {
            ptr: rv(*ptr),
            space: *space,
            align: *align,
            volatile: *volatile,
        },
        Op::Store {
            ptr,
            val,
            ty,
            space,
            align,
            volatile,
        } => Op::Store {
            ptr: rv(*ptr),
            val: rv(*val),
            ty: *ty,
            space: *space,
            align: *align,
            volatile: *volatile,
        },
        Op::Phi(incoming) => Op::Phi(
            incoming
                .iter()
                .map(|&(b, v)| (remap_block(b, block_offset), rv(v)))
                .collect(),
        ),
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
        | Op::GdimZ => {
            unreachable!("GPU index ops are handled by rewrite_gpu_index, not remap_op")
        }
        Op::Barrier => Op::Barrier,
        Op::Shuffle(k, v, a) => Op::Shuffle(*k, rv(*v), rv(*a)),
        Op::Ballot(v) => Op::Ballot(rv(*v)),
        Op::VoteAny(v) => Op::VoteAny(rv(*v)),
        Op::VoteAll(v) => Op::VoteAll(rv(*v)),
        Op::Atomic(o, ptr, val, s) => Op::Atomic(*o, rv(*ptr), rv(*val), *s),
        Op::AtomicCas(ptr, cmp, newv, s) => Op::AtomicCas(rv(*ptr), rv(*cmp), rv(*newv), *s),
        Op::Mma {
            a,
            b,
            c,
            d,
            m,
            n,
            k,
            in_dtype,
            acc_dtype,
            layout_a,
            layout_b,
        } => Op::Mma {
            a: rv(*a),
            b: rv(*b),
            c: rv(*c),
            d: rv(*d),
            m: *m,
            n: *n,
            k: *k,
            in_dtype: *in_dtype,
            acc_dtype: *acc_dtype,
            layout_a: *layout_a,
            layout_b: *layout_b,
        },
    }
}

fn remap_term(term: &Term, inst_offset: u32, block_offset: u32, loop_incr: BlockId) -> Term {
    let rv = |v: ValRef| remap_val(v, inst_offset);
    let rb = |b: BlockId| remap_block(b, block_offset);
    match term {
        Term::Br(b) => Term::Br(rb(*b)),
        Term::CondBr(c, t, e) => Term::CondBr(rv(*c), rb(*t), rb(*e)),
        Term::Switch(scrut, default, cases) => Term::Switch(
            rv(*scrut),
            rb(*default),
            cases.iter().map(|&(v, b)| (v, rb(b))).collect(),
        ),
        // Every thread must advance the loop rather than actually return — the same
        // convention the x86 oracle documents for the same reason.
        Term::Ret(_) => Term::Br(loop_incr),
    }
}

fn flatten_function(f: &Function) -> Result<Function, Diag> {
    let n_orig_insts = f.insts.len() as u32;
    let n_orig_blocks = f.blocks.len() as u32;

    let nthreads_param = f.params.len() as u32;
    let mut params = f.params.clone();
    params.push(I64);

    const PREHEADER: BlockId = BlockId(0);
    const LOOP_CHECK: BlockId = BlockId(1);
    let block_offset = 2u32;
    let orig_entry = BlockId(block_offset);
    let loop_incr = BlockId(2 + n_orig_blocks);
    let loop_end = BlockId(3 + n_orig_blocks);

    const INST_CONST0: InstId = InstId(0);
    const INST_PHI: InstId = InstId(1);
    const INST_CMP: InstId = InstId(2);
    let inst_offset = 3u32;
    let inst_const1 = InstId(3 + n_orig_insts);
    let inst_incr = InstId(4 + n_orig_insts);

    let mut insts: Vec<Inst> = Vec::with_capacity((n_orig_insts + 5) as usize);
    insts.push(Inst {
        ty: I64,
        op: Op::ConstInt(0),
    });
    insts.push(Inst {
        ty: I64,
        op: Op::Phi(vec![
            (PREHEADER, ValRef::Val(INST_CONST0)),
            (loop_incr, ValRef::Val(inst_incr)),
        ]),
    });
    insts.push(Inst {
        ty: Ty::Scalar(Scalar::I1),
        op: Op::ICmp(
            ICmpPred::Ult,
            I64,
            ValRef::Val(INST_PHI),
            ValRef::Param(nthreads_param),
        ),
    });

    for inst in &f.insts {
        let op = if is_gpu_index(&inst.op) {
            rewrite_gpu_index(&inst.op, inst.ty, INST_PHI, nthreads_param)?
        } else {
            remap_op(&inst.op, inst_offset, block_offset)
        };
        insts.push(Inst { ty: inst.ty, op });
    }

    insts.push(Inst {
        ty: I64,
        op: Op::ConstInt(1),
    });
    insts.push(Inst {
        ty: I64,
        op: Op::Bin(BinOp::Add, ValRef::Val(INST_PHI), ValRef::Val(inst_const1)),
    });

    let mut blocks: Vec<Block> = Vec::with_capacity((n_orig_blocks + 4) as usize);
    blocks.push(Block {
        insts: vec![INST_CONST0],
        term: Term::Br(LOOP_CHECK),
    });
    blocks.push(Block {
        insts: vec![INST_PHI, INST_CMP],
        term: Term::CondBr(ValRef::Val(INST_CMP), orig_entry, loop_end),
    });

    for block in &f.blocks {
        let mapped_insts: Vec<InstId> = block
            .insts
            .iter()
            .filter(|id| !matches!(f.insts[id.0 as usize].op, Op::Barrier))
            .map(|id| InstId(id.0 + inst_offset))
            .collect();
        let term = remap_term(&block.term, inst_offset, block_offset, loop_incr);
        blocks.push(Block {
            insts: mapped_insts,
            term,
        });
    }

    blocks.push(Block {
        insts: vec![inst_const1, inst_incr],
        term: Term::Br(LOOP_CHECK),
    });
    blocks.push(Block {
        insts: vec![],
        term: Term::Ret(None),
    });

    Ok(Function {
        name: f.name.clone(),
        params,
        ret: Ty::Void,
        blocks,
        insts,
    })
}

#[cfg(test)]
mod tests;
