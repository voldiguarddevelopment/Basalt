// A genuinely tiled matmul through the oracle: `mma2x2` (`link_and_run.rs`) proves the
// triple-loop `Op::Mma` lowering on a single tile; this proves multiple tiles chained into a
// larger result — the CPU counterpart to `basalt-llvm`'s WMMA lane, run at a scale that needs
// real accumulation across K-steps rather than one call in isolation.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, BinOp, Block, Function, Inst, InstId, MmaLayout, Module, Op, Scalar, Term, Ty,
    ValRef,
};
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

fn push(insts: &mut Vec<Inst>, ty: Ty, op: Op) -> ValRef {
    insts.push(Inst { ty, op });
    ValRef::Val(InstId((insts.len() - 1) as u32))
}

/// `base + offset` as a real `ptr.global` value — `Op::Bin(Add, ..)` on a pointer operand,
/// BIR's own address-arithmetic idiom (see `basalt-sema`'s `lower_index_lvalue`, which this
/// mirrors by hand).
fn addr(insts: &mut Vec<Inst>, base: ValRef, offset: i64) -> ValRef {
    let off = push(insts, Ty::Scalar(Scalar::I64), Op::ConstInt(offset));
    push(
        insts,
        Ty::Ptr(AddrSpace::Global),
        Op::Bin(BinOp::Add, base, off),
    )
}

fn load_at(insts: &mut Vec<Inst>, base: ValRef, offset: i64, ty: Ty, align: u32) -> ValRef {
    let p = addr(insts, base, offset);
    push(
        insts,
        ty,
        Op::Load {
            ptr: p,
            space: AddrSpace::Global,
            align,
            volatile: false,
        },
    )
}

fn store_at(insts: &mut Vec<Inst>, base: ValRef, offset: i64, ty: Ty, align: u32, val: ValRef) {
    let p = addr(insts, base, offset);
    push(
        insts,
        Ty::Void,
        Op::Store {
            ptr: p,
            val,
            ty,
            space: AddrSpace::Global,
            align,
            volatile: false,
        },
    );
}

/// Copies a 16x16 block element by element: source read at `(src_row0+i, src_col0+j)` against
/// `src_row_stride`, destination written at `(dst_row0+i, dst_col0+j)` against
/// `dst_row_stride` — one call covers either direction (real matrix -> compact scratch tile,
/// or compact scratch tile -> real matrix) depending on which side gets the tile-sized stride
/// and which gets the full matrix's own stride.
#[allow(clippy::too_many_arguments)]
fn copy_tile(
    insts: &mut Vec<Inst>,
    src_param: ValRef,
    src_base_off: i64,
    src_row_stride: i64,
    src_row0: i64,
    src_col0: i64,
    dst_param: ValRef,
    dst_base_off: i64,
    dst_row_stride: i64,
    dst_row0: i64,
    dst_col0: i64,
    elem_bytes: i64,
    ty: Ty,
) {
    let align = elem_bytes as u32;
    for i in 0..16i64 {
        for j in 0..16i64 {
            let src_off =
                src_base_off + ((src_row0 + i) * src_row_stride + (src_col0 + j)) * elem_bytes;
            let v = load_at(insts, src_param, src_off, ty, align);
            let dst_off =
                dst_base_off + ((dst_row0 + i) * dst_row_stride + (dst_col0 + j)) * elem_bytes;
            store_at(insts, dst_param, dst_off, ty, align, v);
        }
    }
}

/// `tiled_sgemm_f32(ptr.global a, ptr.global b, ptr.global c, ptr.global d) -> void`:
/// `D = A@B + C` at `M=N=K=32`, row-major throughout, decomposed into a 2x2 grid of 16x16
/// output tiles, each accumulated over two 16-deep K-steps via `Op::Mma` at the fixed
/// `m=n=k=16` tile shape (the only shape the WMMA lane on the LLVM side lowers for real —
/// this fixture's own `in_dtype`/`in_elem_bytes` are parameters precisely so the same
/// construction produces both this crate's f32-in/f32-acc oracle variant and that lane's
/// f16-in/f32-acc variant from one shared shape).
///
/// `Op::Mma` has no stride field: each operand's leading dimension is always its own `m`/`n`/
/// `k` extent (see the op's own doc comment in `basalt-bir`), so a 16x16 block carved directly
/// out of a 32-wide row-major matrix cannot be handed to it as-is (its rows are 32 elements
/// apart, not 16). Every tile this kernel feeds `Op::Mma` is therefore staged into a compact,
/// contiguous 16x16 scratch buffer first via an explicit element-by-element copy (unrolled
/// here at BIR-construction time, not a BIR-level loop), the same "stage into a tile-shaped
/// buffer before the tensor op" a real shared-memory GEMM kernel does. There is no local-array
/// support wide enough for a 256-element tile (`basalt-sema`'s local/shared slot convention —
/// see `oracle.rs`'s and `lower.rs`'s own module headers — reserves a flat 8 bytes per opaque
/// slot key, not a real array), so the scratch tiles live in extra space appended past `d`'s
/// own real 1024-float payload instead of a fifth kernel parameter: `d`'s host allocation must
/// be wider than the real output, and everything from `D_REAL_BYTES` onward is this kernel's
/// own private workspace, never read back by a caller.
///
/// One scratch buffer of each kind (`a`/`b`/`c`) is reused across all eight `Op::Mma` calls —
/// safe because everything runs in strict program order, one tile fully at a time. `c`'s
/// scratch is read by `Op::Mma` as the accumulator input and written as `d` in the same call
/// (`d` aliases `c` by design, per `Op::Mma`'s own doc comment): staged from the real `C` tile
/// once at the first K-step, then carried forward by the op itself into the second.
fn build_tiled_sgemm(name: &str, in_dtype: Scalar, in_elem_bytes: i64) -> Module {
    const FULL: i64 = 32;
    const TILE: i64 = 16;
    let d_real_bytes = FULL * FULL * 4;
    let a_scratch_off = d_real_bytes;
    let b_scratch_off = a_scratch_off + TILE * TILE * in_elem_bytes;
    let c_scratch_off = b_scratch_off + TILE * TILE * in_elem_bytes;

    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let in_ty = Ty::Scalar(in_dtype);
    let acc_ty = Ty::Scalar(Scalar::F32);

    let mut insts: Vec<Inst> = Vec::new();
    let a_p = ValRef::Param(0);
    let b_p = ValRef::Param(1);
    let c_p = ValRef::Param(2);
    let d_p = ValRef::Param(3);

    for tr in 0..2i64 {
        for tc in 0..2i64 {
            for ks in 0..2i64 {
                copy_tile(
                    &mut insts,
                    a_p,
                    0,
                    FULL,
                    tr * TILE,
                    ks * TILE,
                    d_p,
                    a_scratch_off,
                    TILE,
                    0,
                    0,
                    in_elem_bytes,
                    in_ty,
                );
                copy_tile(
                    &mut insts,
                    b_p,
                    0,
                    FULL,
                    ks * TILE,
                    tc * TILE,
                    d_p,
                    b_scratch_off,
                    TILE,
                    0,
                    0,
                    in_elem_bytes,
                    in_ty,
                );
                if ks == 0 {
                    copy_tile(
                        &mut insts,
                        c_p,
                        0,
                        FULL,
                        tr * TILE,
                        tc * TILE,
                        d_p,
                        c_scratch_off,
                        TILE,
                        0,
                        0,
                        4,
                        acc_ty,
                    );
                }

                // A real launch runs this kernel with a full warp of redundant threads (see
                // this fixture's callers): every lane stages the identical tile data via its
                // own scalar stores above, and `Op::Mma` itself is warp-collective, but
                // cross-lane visibility of one lane's global-memory stores to another lane's
                // subsequent load is not guaranteed without an explicit barrier — a plain
                // SIMT "same instruction, same cycle" execution model does not imply that.
                // `barrier` is a genuine no-op under the oracle's own one-thread-at-a-time
                // execution (see `oracle.rs`'s module header) so this is free there; on a real
                // GPU it is the fence that makes the scratch tile's data actually visible
                // before `Op::Mma` reads it.
                push(&mut insts, Ty::Void, Op::Barrier);
                let a_scratch_ptr = addr(&mut insts, d_p, a_scratch_off);
                let b_scratch_ptr = addr(&mut insts, d_p, b_scratch_off);
                let c_scratch_ptr = addr(&mut insts, d_p, c_scratch_off);
                push(
                    &mut insts,
                    Ty::Void,
                    Op::Mma {
                        a: a_scratch_ptr,
                        b: b_scratch_ptr,
                        c: c_scratch_ptr,
                        d: c_scratch_ptr,
                        m: 16,
                        n: 16,
                        k: 16,
                        in_dtype,
                        acc_dtype: Scalar::F32,
                        layout_a: MmaLayout::RowMajor,
                        layout_b: MmaLayout::RowMajor,
                    },
                );
                // Same reasoning in reverse: the next K-step's re-staging of `a`/`b` scratch,
                // or (after the last K-step) the D write-back below, must not race the wmma
                // store this call just issued.
                push(&mut insts, Ty::Void, Op::Barrier);
            }
            copy_tile(
                &mut insts,
                d_p,
                c_scratch_off,
                TILE,
                0,
                0,
                d_p,
                0,
                FULL,
                tr * TILE,
                tc * TILE,
                4,
                acc_ty,
            );
        }
    }

    let n_insts = insts.len();
    let f = Function {
        is_kernel: true,
        name: name.into(),
        params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
        ret: Ty::Void,
        insts,
        blocks: vec![Block {
            insts: (0..n_insts as u32).map(InstId).collect(),
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
fn hand_built_tiled_sgemm_f32_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping hand_built_tiled_sgemm_f32_links_and_runs: `cc` not found");
        return;
    }

    let module = build_tiled_sgemm("tiled_sgemm_f32", Scalar::F32, 4);
    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for hand-built tiled_sgemm_f32");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_tiled_sgemm_{pid}.o"));
    write_object(bytes, &obj);

    let scratch = std::env::temp_dir();
    let shim_o = scratch.join(format!("basalt_tiled_sgemm_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_tiled_sgemm_exe_{pid}"));

    let shim_path = root.join("examples/cpu_launch_tiled_sgemm.c");
    run_cc(&[
        OsStr::new("-c"),
        shim_path.as_os_str(),
        OsStr::new("-o"),
        shim_o.as_os_str(),
    ]);
    run_cc(&[
        shim_o.as_os_str(),
        obj.as_os_str(),
        OsStr::new("-o"),
        exe.as_os_str(),
    ]);

    run_and_check(&exe);

    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_file(&obj);
}
