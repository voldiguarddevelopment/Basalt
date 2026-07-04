// The regalloc backend's moment of truth: does the machine code `X86Regalloc::emit` actually
// produces link, via the real system C compiler, and run to the correct answer? Mirrors
// `link_and_run.rs` (the oracle's own version of this file) exactly in structure and intent —
// see that file's header for the general rationale. This file additionally covers what is
// unique to a real register allocator: forced register pressure (some values must spill) and a
// real `phi` resolved through predecessor-block copies (an if/else merge).

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{
    BinOp, Block, BlockId, Function, ICmpPred, Inst, InstId, Module, Op, Scalar, Term, Ty, ValRef,
};
use basalt_frontend_c::PpOpts;
use basalt_x86::X86Regalloc;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

fn cc_available() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_cc(args: &[&OsStr]) {
    let out = Command::new("cc")
        .args(args)
        .output()
        .expect("cc is present and spawns");
    assert!(
        out.status.success(),
        "cc {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn run_and_check(exe: &Path) {
    let out = Command::new(exe).output().expect("built executable runs");
    assert!(
        out.status.success(),
        "{} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
        exe.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    print!("{}", String::from_utf8_lossy(&out.stdout));
}

fn write_object(bytes: &[u8], path: &Path) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}

fn compile_link_and_run(root: &Path, shim_c: &str, payload_o: &Path, tag: &str) {
    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let shim_o = scratch.join(format!("basalt_ra_{tag}_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_ra_{tag}_exe_{pid}"));

    let shim_path = root.join(shim_c);
    run_cc(&[
        OsStr::new("-c"),
        shim_path.as_os_str(),
        OsStr::new("-o"),
        shim_o.as_os_str(),
    ]);
    run_cc(&[
        shim_o.as_os_str(),
        payload_o.as_os_str(),
        OsStr::new("-o"),
        exe.as_os_str(),
    ]);

    run_and_check(&exe);

    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&exe);
}

#[test]
fn vector_add_links_and_runs_via_full_pipeline() {
    if !cc_available() {
        eprintln!("skipping vector_add_links_and_runs_via_full_pipeline: `cc` not found");
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/vector_add.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = PpOpts {
        include_dirs: vec![],
        defines: vec![],
        base_dir: src_path.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &opts);
    assert!(pp_errors.is_empty(), "{pp_errors:?}");
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(parse_errors.is_empty(), "{parse_errors:?}");

    let sema_diags = basalt_sema::check(&tu);
    assert!(sema_diags.is_empty(), "{sema_diags:?}");

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(lower_diags.is_empty(), "{lower_diags:?}");

    assert_eq!(X86Regalloc.supports(&module), Support::Supported);
    let artifact = X86Regalloc
        .emit(&module, &EmitOpts::default())
        .expect("regalloc emit succeeds for vector_add");
    let bytes = artifact
        .as_bytes()
        .expect("regalloc backend emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_ra_vadd_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_vadd.c", &obj, "vadd");

    let _ = std::fs::remove_file(&obj);
}

/// `add_i32(i32, i32) -> i32`, the same fixture shape as `link_and_run.rs`'s own
/// `hand_built_add_i32`, isolating the basic scalar calling-convention/return-value path (both
/// params land in real registers under this backend's default 4-int-register pool) from
/// everything the vector_add proof already covers.
fn hand_built_add_i32() -> Module {
    let f = Function {
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
    };
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

#[test]
fn hand_built_add_i32_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping hand_built_add_i32_links_and_runs: `cc` not found");
        return;
    }

    let module = hand_built_add_i32();
    assert_eq!(X86Regalloc.supports(&module), Support::Supported);
    let artifact = X86Regalloc
        .emit(&module, &EmitOpts::default())
        .expect("regalloc emit succeeds for hand-built add_i32");
    let bytes = artifact
        .as_bytes()
        .expect("regalloc backend emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_ra_add_i32_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_add_i32.c", &obj, "add_i32");

    let _ = std::fs::remove_file(&obj);
}

/// Forces register pressure high enough that some int-class values must spill: 5 params (the
/// most SysV allows this backend's ABI classification to accept alongside the trailing
/// `nthreads` argument) chained into 4 adds without ever letting the params die early. Every
/// param is defined at function entry (position 0) and not consumed until its own link of the
/// chain, so at the very first instruction all 5 params plus that instruction's own result are
/// simultaneously live — 6 concurrently-live int-class values against this backend's
/// 4-register int pool, which the linear-scan spill rule cannot avoid spilling at least two
/// of. Confirms the spill-slot load/store path (not just the register path) computes the right
/// answer, not just that the allocator's own unit tests report a spill count.
fn hand_built_spill_heavy_sum() -> Module {
    let params = vec![Ty::Scalar(Scalar::I32); 5];
    let insts = vec![
        Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Bin(BinOp::Add, ValRef::Val(InstId(0)), ValRef::Param(2)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Bin(BinOp::Add, ValRef::Val(InstId(1)), ValRef::Param(3)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Bin(BinOp::Add, ValRef::Val(InstId(2)), ValRef::Param(4)),
        },
    ];

    let f = Function {
        is_kernel: true,
        name: "spill_heavy_sum".into(),
        params,
        ret: Ty::Scalar(Scalar::I32),
        insts,
        blocks: vec![Block {
            insts: (0..4u32).map(InstId).collect(),
            term: Term::Ret(Some(ValRef::Val(InstId(3)))),
        }],
    };
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

#[test]
fn spill_heavy_sum_links_and_runs_correctly() {
    if !cc_available() {
        eprintln!("skipping spill_heavy_sum_links_and_runs_correctly: `cc` not found");
        return;
    }

    let module = hand_built_spill_heavy_sum();
    assert_eq!(X86Regalloc.supports(&module), Support::Supported);
    let artifact = X86Regalloc
        .emit(&module, &EmitOpts::default())
        .expect("regalloc emit succeeds for spill_heavy_sum");
    let bytes = artifact
        .as_bytes()
        .expect("regalloc backend emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_ra_spill_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(
        &root,
        "examples/cpu_launch_spill_heavy_sum.c",
        &obj,
        "spill",
    );

    let _ = std::fs::remove_file(&obj);
}

/// `func @max_i32(i32, i32) -> i32`: `if (a > b) { m = a } else { m = b }; return m;`, lowered
/// to a real `phi` at the merge block — exercises this backend's copy-insertion phi resolution
/// (predecessors write into the phi's own location before branching) via real execution, not
/// just structural inspection.
fn hand_built_max_i32_via_phi() -> Module {
    let f = Function {
        is_kernel: true,
        name: "max_i32".into(),
        params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
        ret: Ty::Scalar(Scalar::I32),
        insts: vec![
            Inst {
                ty: Ty::Scalar(Scalar::I1),
                op: Op::ICmp(
                    ICmpPred::Sgt,
                    Ty::Scalar(Scalar::I32),
                    ValRef::Param(0),
                    ValRef::Param(1),
                ),
            },
            Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Phi(vec![
                    (BlockId(1), ValRef::Param(0)),
                    (BlockId(2), ValRef::Param(1)),
                ]),
            },
        ],
        blocks: vec![
            Block {
                insts: vec![InstId(0)],
                term: Term::CondBr(ValRef::Val(InstId(0)), BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![InstId(1)],
                term: Term::Ret(Some(ValRef::Val(InstId(1)))),
            },
        ],
    };
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

#[test]
fn max_i32_via_phi_links_and_runs_correctly() {
    if !cc_available() {
        eprintln!("skipping max_i32_via_phi_links_and_runs_correctly: `cc` not found");
        return;
    }

    let module = hand_built_max_i32_via_phi();
    assert_eq!(X86Regalloc.supports(&module), Support::Supported);
    let artifact = X86Regalloc
        .emit(&module, &EmitOpts::default())
        .expect("regalloc emit succeeds for max_i32_via_phi");
    let bytes = artifact
        .as_bytes()
        .expect("regalloc backend emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_ra_max_phi_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_max_i32.c", &obj, "max_phi");

    let _ = std::fs::remove_file(&obj);
}
