// The oracle's moment of truth: does the machine code `X86Oracle::emit` actually produces
// link, via the real system C compiler, and run to the correct answer? Everything before this
// point (unit tests, `--ir` dumps, ELF-shape assertions in `oracle.rs`) only checks structure;
// nothing has been executed. These two tests shell out to `cc` and a real subprocess, no
// in-process JIT.
//
// Proofs, each covering a different slice:
//   - `vector_add_links_and_runs_via_full_pipeline` runs the actual lex/preprocess/parse/
//     check/lower pipeline over `tests/kernels/vector_add.cu`, exactly like `basalt-cli`'s own
//     `--ir` path, then links the oracle's output against a real C caller.
//   - `hand_built_add_i32_links_and_runs` builds BIR directly, skipping the frontend/sema
//     stages entirely, to isolate the oracle's basic scalar calling-convention/return-value
//     path from everything upstream of it. If the first test ever breaks, this one narrows
//     down whether the fault is in the oracle or further up the pipeline.
//   - `hand_built_host_launches_kernel_links_and_runs` (P13-T1c-i) isolates the oracle's own
//     real intra-object `call` machinery the same way: a hand-built host function launches a
//     hand-built kernel via a real `Op::KernelLaunch`, with no frontend/sema involved, so a
//     failure here narrows the fault to the call/argument-marshaling path itself rather than
//     anything upstream of it. `cuda_kernel_launch_links_and_runs_via_full_pipeline` is the
//     real end-to-end proof: a genuine `.cu` host function launching `vector_add.cu`'s own
//     kernel via real `<<<>>>` syntax, through the whole frontend/sema/lower pipeline.
//   - `cuda_malloc_memcpy_free_links_and_runs_via_full_pipeline` (P13-T1c-ii) is the real
//     libc-relocation proof: a genuine `.cu` host function that allocates its own device
//     buffers via real `cudaMalloc`/`cudaMemcpy`/`cudaFree` calls against libc (this
//     project's first ever real ELF relocation), through the same whole pipeline.
//   - `device_helper_call_links_and_runs_via_full_pipeline` (P13-T-calls-i) is the real
//     device-helper-call proof: a genuine `.cu` file (`tests/kernels/device_helper_square.cu`)
//     where a `__global__` kernel calls a real `__device__` helper function via a genuine
//     `Op::Call`, through the whole frontend/sema/lower/oracle pipeline — the kernel itself is
//     this object's own callable entry point (the `ModuleShape::KernelWithHelpers` shape has no
//     separate host function), so the C shim calls it directly, the same calling convention as
//     `vector_add_links_and_runs_via_full_pipeline`'s plain kernel.
//   - `device_helper_chain_links_and_runs_via_full_pipeline` (P13-T-calls-ii) goes one further:
//     a genuine `.cu` file (`tests/kernels/device_helper_chain.cu`) where the kernel calls a
//     `__device__` helper that calls another `__device__` helper that calls a third — a real
//     three-level device-to-device call chain, confirming each link gets its own correct,
//     independent stack frame rather than assuming it from the single-hop proof above.
//
// All three scalar-calling-convention proofs read the exact calling convention off
// `oracle.rs`'s own module header and its `INT_ARG_REGS`/`SSE_ARG_REGS` classification, not off
// any assumption: every param BIR sees is integer-class here (pointers and `i32`), so they
// consume the SysV integer registers in order, and — for a kernel launched the old way,
// directly by a C caller rather than by a real `Op::KernelLaunch` — the trailing `nthreads`
// argument always takes the next integer register after the function's own params, always read
// back a full 8 bytes on the oracle side, hence the C shims declare it `int64_t`, not `int`. A
// host function's own entry point (the `cpu_launch_host_write_sum.c`/`cpu_launch_vadd_host.c`
// shims below) takes no such trailing argument at all: it is an ordinary function called
// directly, not a kernel launched by anything.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, BinOp, Block, Function, Inst, InstId, MmaLayout, Module, Op, Scalar, Term, Ty,
    ValRef,
};
use basalt_frontend_c::PpOpts;
use basalt_x86::X86Oracle;

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

/// Runs `cc`, failing the test with its stderr if it doesn't exit 0 — a compile/link failure
/// must be diagnosable, not a silent test failure.
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

/// Runs `exe`, asserting a zero exit status; the process's own stdout/stderr are folded into
/// the panic message so a wrong-answer failure shows exactly what mismatched.
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

/// Compiles `shim_c` and links it with `payload_o` into `exe`, then runs and checks it. The
/// common tail shared by both tests below.
fn compile_link_and_run(root: &Path, shim_c: &str, payload_o: &Path, tag: &str) {
    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let shim_o = scratch.join(format!("basalt_{tag}_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_{tag}_exe_{pid}"));

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
    assert!(
        pp_errors.is_empty(),
        "preprocessing vector_add.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing vector_add.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking vector_add.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering vector_add.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for vector_add");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_vadd_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_vadd.c", &obj, "vadd");

    let _ = std::fs::remove_file(&obj);
}

/// `add_i32(i32, i32) -> i32`, built directly from `basalt_bir` types (the same shape as
/// `oracle.rs`'s own private `func_add_i32` fixture, reconstructed here since that fixture
/// lives in a `#[cfg(test)]` module private to that crate). Both params are integer-class, so
/// `nthreads` is the third integer register — this exercises a non-void scalar return, which
/// `vector_add` (a `void` kernel) never does.
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
    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for hand-built add_i32");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_add_i32_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_add_i32.c", &obj, "add_i32");

    let _ = std::fs::remove_file(&obj);
}

/// `mma2x2(ptr.global, ptr.global, ptr.global, ptr.global) -> void`: `D = A*B + C` at
/// `M=N=K=2`, row-major `A`/`B`, `f32` throughout — the same fixture as `oracle.rs`'s own
/// private `func_mma2x2` (reconstructed here for the same reason `hand_built_add_i32` mirrors
/// `func_add_i32`). All four params are pointers (integer-class), so `nthreads` is the fifth
/// integer register, `r8`.
fn hand_built_mma2x2() -> Module {
    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let f = Function {
        is_kernel: true,
        name: "mma2x2".into(),
        params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
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
                layout_a: MmaLayout::RowMajor,
                layout_b: MmaLayout::RowMajor,
            },
        }],
        blocks: vec![Block {
            insts: vec![InstId(0)],
            term: Term::Ret(None),
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
fn hand_built_mma2x2_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping hand_built_mma2x2_links_and_runs: `cc` not found");
        return;
    }

    let module = hand_built_mma2x2();
    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for hand-built mma2x2");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_mma2x2_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_mma2x2.c", &obj, "mma2x2");

    let _ = std::fs::remove_file(&obj);
}

/// `write_sum(ptr.global out, i32 a, i32 b) -> void { *out = a + b; }`, a real kernel with no
/// frontend involvement — isolates the store/arithmetic path the launched kernel itself needs
/// from the call-machinery under test below.
fn hand_built_write_sum_kernel() -> Function {
    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    Function {
        is_kernel: true,
        name: "write_sum".into(),
        params: vec![ptr_global, i32t, i32t],
        ret: Ty::Void,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::Bin(BinOp::Add, ValRef::Param(1), ValRef::Param(2)),
            },
            Inst {
                ty: Ty::Void,
                op: Op::Store {
                    ptr: ValRef::Param(0),
                    val: ValRef::Val(InstId(0)),
                    ty: i32t,
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

/// `host_write_sum(ptr.global out) -> void`, a real hand-built host function launching
/// `write_sum` above via a genuine `Op::KernelLaunch`: `grid=(1,1,1)`, `block=(1,1,1)`
/// (`nthreads` at the call site is the flattened product `1`), args `(out, 10, 20)`, so a
/// correct run leaves `30` in `*out`. Instruction indices: 0-2 grid.xyz, 3-5 block.xyz, 6
/// shared (default), 7 stream (default), 8-9 the two scalar launch args, 10 the launch.
fn hand_built_host_launches_write_sum() -> Module {
    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let i64t = Ty::Scalar(Scalar::I64);
    let host = Function {
        is_kernel: false,
        name: "host_write_sum".into(),
        params: vec![ptr_global],
        ret: Ty::Void,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            }, // 0: grid.x
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            }, // 1: grid.y
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            }, // 2: grid.z
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            }, // 3: block.x
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            }, // 4: block.y
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            }, // 5: block.z
            Inst {
                ty: i64t,
                op: Op::ConstInt(0),
            }, // 6: shared (default)
            Inst {
                ty: ptr_global,
                op: Op::ConstInt(0),
            }, // 7: stream (default)
            Inst {
                ty: i32t,
                op: Op::ConstInt(10),
            }, // 8: arg a
            Inst {
                ty: i32t,
                op: Op::ConstInt(20),
            }, // 9: arg b
            Inst {
                ty: Ty::Void,
                op: Op::KernelLaunch {
                    kernel: "write_sum".into(),
                    grid: [
                        ValRef::Val(InstId(0)),
                        ValRef::Val(InstId(1)),
                        ValRef::Val(InstId(2)),
                    ],
                    block: [
                        ValRef::Val(InstId(3)),
                        ValRef::Val(InstId(4)),
                        ValRef::Val(InstId(5)),
                    ],
                    shared: ValRef::Val(InstId(6)),
                    stream: ValRef::Val(InstId(7)),
                    args: vec![
                        ValRef::Param(0),
                        ValRef::Val(InstId(8)),
                        ValRef::Val(InstId(9)),
                    ],
                },
            }, // 10
        ],
        blocks: vec![Block {
            insts: (0..=10).map(InstId).collect(),
            term: Term::Ret(None),
        }],
    };
    Module {
        funcs: vec![host, hand_built_write_sum_kernel()],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

#[test]
fn hand_built_host_launches_kernel_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping hand_built_host_launches_kernel_links_and_runs: `cc` not found");
        return;
    }

    let module = hand_built_host_launches_write_sum();
    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for a hand-built host function launching a kernel");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_host_write_sum_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(
        &root,
        "examples/cpu_launch_host_write_sum.c",
        &obj,
        "host_write_sum",
    );

    let _ = std::fs::remove_file(&obj);
}

/// The real end-to-end proof (P13-T1c-i): a genuine `.cu` host function
/// (`tests/kernels/cpu_launch_vadd_host.cu`, `launch_vector_add`) launching
/// `tests/kernels/vector_add.cu`'s own existing, unmodified kernel via real `<<<>>>` syntax.
/// `a`/`b`/`c` are ordinary pointer parameters of `launch_vector_add` itself — the buffers are
/// pre-allocated by this test's own C driver (`examples/cpu_launch_vadd_host.c`), not by any
/// `cudaMalloc` (`P13-T1c-ii`'s job) and not by a local array inside the `.cu` file (a local
/// *array*'s real multi-byte stack home is a separate, general gap in this oracle's `Frame` —
/// see that file's own header — unrelated to this task's own call-machinery scope). Runs
/// through the exact same lex/preprocess/parse/check/lower pipeline as `--cpu`, then the
/// oracle, then a real link against the C driver and run.
#[test]
fn cuda_kernel_launch_links_and_runs_via_full_pipeline() {
    if !cc_available() {
        eprintln!("skipping cuda_kernel_launch_links_and_runs_via_full_pipeline: `cc` not found");
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/cpu_launch_vadd_host.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = PpOpts {
        include_dirs: vec![],
        defines: vec![],
        base_dir: src_path.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &opts);
    assert!(
        pp_errors.is_empty(),
        "preprocessing cpu_launch_vadd_host.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing cpu_launch_vadd_host.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking cpu_launch_vadd_host.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering cpu_launch_vadd_host.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for cpu_launch_vadd_host.cu + vector_add");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_vadd_host_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_vadd_host.c", &obj, "vadd_host");

    let _ = std::fs::remove_file(&obj);
}

/// The real end-to-end proof (P13-T1c-ii): a genuine `.cu` host function
/// (`tests/kernels/cpu_launch_vadd_malloc.cu`, `launch_vector_add_malloc`) that allocates its
/// own device buffers via real `cudaMalloc`/`cudaMemcpy`/`cudaFree` calls against libc, rather
/// than relying on this test's own C driver to pre-allocate them (unlike
/// `cuda_kernel_launch_links_and_runs_via_full_pipeline` above). Closes the loop this project's
/// first ever real ELF relocation (`R_X86_64_PLT32` against `malloc`/`memcpy`/`free`,
/// `basalt_backend::elf::ElfRelocation`) exists to make possible: if the relocation's offset,
/// addend, or symbol were wrong, `cc`'s own link step would either fail outright or produce a
/// binary that corrupts memory at `malloc`/`memcpy`/`free`'s real call sites, not just silently
/// mismatch a value the way a wrong scalar computation would.
#[test]
fn cuda_malloc_memcpy_free_links_and_runs_via_full_pipeline() {
    if !cc_available() {
        eprintln!(
            "skipping cuda_malloc_memcpy_free_links_and_runs_via_full_pipeline: `cc` not found"
        );
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/cpu_launch_vadd_malloc.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = PpOpts {
        include_dirs: vec![],
        defines: vec![],
        base_dir: src_path.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &opts);
    assert!(
        pp_errors.is_empty(),
        "preprocessing cpu_launch_vadd_malloc.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing cpu_launch_vadd_malloc.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking cpu_launch_vadd_malloc.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering cpu_launch_vadd_malloc.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for cpu_launch_vadd_malloc.cu + vector_add");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_vadd_malloc_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(
        &root,
        "examples/cpu_launch_vadd_malloc.c",
        &obj,
        "vadd_malloc",
    );

    let _ = std::fs::remove_file(&obj);
}

/// The real end-to-end proof (P13-T-calls-i): a genuine `.cu` file where a `__global__`
/// kernel calls a real `__device__` helper function via a genuine `Op::Call`, closing the gap
/// `basalt_sema::lower.rs`'s own module header used to document ("BIR has no call instruction
/// at all") — see `tests/kernels/device_helper_square.cu`. `square_vector` is this object's own
/// callable entry point (`ModuleShape::KernelWithHelpers` has no separate host function, unlike
/// the `cuda_kernel_launch_*`/`cuda_malloc_*` tests above), so if the intra-object `call rel32`
/// to `square`, its argument marshaling, or its return-value handling were wrong, this would
/// either fail to link (a bad `call` target) or produce a value mismatch the C shim's own
/// per-element check below catches, not a silent pass.
#[test]
fn device_helper_call_links_and_runs_via_full_pipeline() {
    if !cc_available() {
        eprintln!("skipping device_helper_call_links_and_runs_via_full_pipeline: `cc` not found");
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/device_helper_square.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = PpOpts {
        include_dirs: vec![],
        defines: vec![],
        base_dir: src_path.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &opts);
    assert!(
        pp_errors.is_empty(),
        "preprocessing device_helper_square.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing device_helper_square.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking device_helper_square.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering device_helper_square.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for device_helper_square.cu");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_device_helper_square_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(
        &root,
        "examples/cpu_launch_device_helper_square.c",
        &obj,
        "device_helper_square",
    );

    let _ = std::fs::remove_file(&obj);
}

/// The real end-to-end proof (P13-T-calls-ii): a genuine `.cu` file
/// (`tests/kernels/device_helper_chain.cu`) where a `__global__` kernel calls a `__device__`
/// helper that itself calls another `__device__` helper that itself calls a third — a real
/// three-level device-to-device call chain, not just the one kernel-to-helper hop
/// `device_helper_call_links_and_runs_via_full_pipeline` above already covers. Each link's own
/// stack frame, argument register, and return-value handoff has to be independently correct for
/// the final value to come out right, since each of `negate_then_scale`/`scale_then_inc`/`inc`
/// is lowered as its own real function with its own real prologue/epilogue
/// (`emit_function_body`'s `is_host = true` shape) sharing one `Enc` — a corrupted or aliased
/// frame between nested calls would show up here as a wrong-answer failure, not a crash.
#[test]
fn device_helper_chain_links_and_runs_via_full_pipeline() {
    if !cc_available() {
        eprintln!("skipping device_helper_chain_links_and_runs_via_full_pipeline: `cc` not found");
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/device_helper_chain.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = PpOpts {
        include_dirs: vec![],
        defines: vec![],
        base_dir: src_path.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &opts);
    assert!(
        pp_errors.is_empty(),
        "preprocessing device_helper_chain.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing device_helper_chain.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking device_helper_chain.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering device_helper_chain.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for device_helper_chain.cu");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_device_helper_chain_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(
        &root,
        "examples/cpu_launch_device_helper_chain.c",
        &obj,
        "device_helper_chain",
    );

    let _ = std::fs::remove_file(&obj);
}
