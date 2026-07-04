// Structural coverage for the flatten transform itself: refusal paths, and that the happy
// path produces a module `lower_module` accepts and LLVM's own verifier is satisfied with.
// The real end-to-end proof (linking and running the flattened `vector_add` through a real
// x86 `TargetMachine`) lives in `tests/link_and_run.rs`, one level up — this file only checks
// the transform's own shape.

use super::*;
use basalt_bir::{AddrSpace, LaunchBounds, ShuffleKind};
use inkwell::context::Context;

fn wrap(f: Function) -> Module {
    Module {
        funcs: vec![f],
        launch_bounds: None::<LaunchBounds>,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

/// `void bump(ptr) { *ptr = *ptr + tid.x; }` — small enough to hand-check the flattened shape,
/// but still exercises a real GPU index op feeding real arithmetic.
fn tidx_bump_module() -> Module {
    let i32t = I32;
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let f = Function {
        is_kernel: true,
        name: "bump".into(),
        params: vec![ptrt],
        ret: Ty::Void,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::TidX,
            },
            Inst {
                ty: i32t,
                op: Op::Load {
                    ptr: ValRef::Param(0),
                    space: AddrSpace::Global,
                    align: 4,
                    volatile: false,
                },
            },
            Inst {
                ty: i32t,
                op: Op::Bin(BinOp::Add, ValRef::Val(InstId(1)), ValRef::Val(InstId(0))),
            },
            Inst {
                ty: Ty::Void,
                op: Op::Store {
                    ptr: ValRef::Param(0),
                    val: ValRef::Val(InstId(2)),
                    ty: i32t,
                    space: AddrSpace::Global,
                    align: 4,
                    volatile: false,
                },
            },
        ],
        blocks: vec![Block {
            insts: vec![InstId(0), InstId(1), InstId(2), InstId(3)],
            term: Term::Ret(None),
        }],
    };
    wrap(f)
}

#[test]
fn a_module_with_no_gpu_index_op_is_not_flagged_for_flattening() {
    let i32t = I32;
    let f = Function {
        is_kernel: true,
        name: "add_i32".into(),
        params: vec![i32t, i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        }],
        blocks: vec![Block {
            insts: vec![InstId(0)],
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };
    assert!(!uses_gpu_index_ops(&wrap(f)));
}

#[test]
fn tid_x_module_is_flagged_and_flattens_to_a_loop_with_a_trailing_nthreads_param() {
    let module = tidx_bump_module();
    assert!(uses_gpu_index_ops(&module));

    let flat = flatten_to_native_cpu_loop(&module).expect("flattening succeeds");
    let f = &flat.funcs[0];
    assert_eq!(f.ret, Ty::Void);
    // Original single pointer param, plus the trailing i64 nthreads param.
    assert_eq!(f.params, vec![Ty::Ptr(AddrSpace::Global), I64]);
    // Loop skeleton (preheader, loop_check) + original block + (loop_incr, loop_end).
    assert_eq!(f.blocks.len(), 5);

    let ctx = Context::create();
    let llvm_mod =
        crate::lower_module(&flat, &ctx, crate::GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("flattened module verifies");
}

#[test]
fn non_void_return_is_refused_not_guessed_at() {
    let i32t = I32;
    let f = Function {
        is_kernel: true,
        name: "usestid".into(),
        params: vec![],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::TidX,
        }],
        blocks: vec![Block {
            insts: vec![InstId(0)],
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };
    let err = flatten_to_native_cpu_loop(&wrap(f))
        .expect_err("non-void kernel return is out of scope for cpu-loop flattening");
    assert_eq!(err.code, ECode::UnsupportedType);
}

#[test]
fn warp_collective_op_is_refused_not_guessed_at() {
    let i32t = I32;
    let f = Function {
        is_kernel: true,
        name: "usesshuffle".into(),
        params: vec![i32t, i32t],
        ret: Ty::Void,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::TidX,
            },
            Inst {
                ty: i32t,
                op: Op::Shuffle(ShuffleKind::Idx, ValRef::Param(0), ValRef::Param(1)),
            },
        ],
        blocks: vec![Block {
            insts: vec![InstId(0), InstId(1)],
            term: Term::Ret(None),
        }],
    };
    let err = flatten_to_native_cpu_loop(&wrap(f))
        .expect_err("shuffle has no meaning under one-thread-at-a-time execution");
    assert_eq!(err.code, ECode::UnsupportedOp);
}
