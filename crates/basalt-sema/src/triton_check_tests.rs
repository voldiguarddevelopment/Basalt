// Fixture-driven tests for `triton_check`: parse a small Triton kernel with the real
// `basalt-frontend-triton` pipeline, run `check_triton`, and assert on the collected
// `ECode`s (never message text) plus, where it matters, the concrete `TileTy` inferred for a
// specific expression — matches `basalt-sema`'s own `tests.rs` rigor for the CUDA-C checker.

use basalt_diag::ECode;
use basalt_frontend_triton::ast::{Expr, Stmt};
use basalt_frontend_triton::parse;

use super::check_triton;
use crate::triton_ty::{Dim, Elem, TileTy};

fn parse_ok(src: &str) -> basalt_frontend_triton::ast::Module {
    let (module, diags) = parse(src);
    assert!(diags.is_empty(), "unexpected parse diags: {diags:?}");
    module
}

fn codes(diags: &[basalt_diag::Diag]) -> Vec<ECode> {
    diags.iter().map(|d| d.code).collect()
}

/// The right-hand side of the `idx`-th statement, assuming it is a plain `Assign`.
fn assign_value(body: &[Stmt], idx: usize) -> &Expr {
    match &body[idx] {
        Stmt::Assign { value, .. } => value,
        other => panic!("statement {idx} is not a plain assignment: {other:?}"),
    }
}

const VECTOR_ADD: &str = r#"
import triton
import triton.language as tl


@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
"#;

#[test]
fn vector_add_infers_rank1_throughout_with_no_diagnostics() {
    let module = parse_ok(VECTOR_ADD);
    let (shapes, diags) = check_triton(&module);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    assert_eq!(shapes.len(), 1);
    let k = &shapes[0];
    assert_eq!(k.name, "vector_add");

    // a_ptr, b_ptr, c_ptr, n: ordinary (non-constexpr) params, all rank-0/untracked dtype.
    for p in &k.param_types[..4] {
        assert_eq!(*p, TileTy::Scalar(Elem::Unknown));
    }
    // BLOCK_SIZE: constexpr, no default -> a rank-0 int carrying a symbolic dimension name.
    assert_eq!(k.param_types[4], TileTy::Scalar(Elem::Int));

    let body = &module.kernels[0].body;
    let offsets_ty = k.expr_types.get(&assign_value(body, 2).span()).unwrap();
    assert_eq!(
        *offsets_ty,
        TileTy::Rank1(Dim::Symbolic("BLOCK_SIZE".to_string()))
    );

    let mask_ty = k.expr_types.get(&assign_value(body, 3).span()).unwrap();
    assert_eq!(
        *mask_ty,
        TileTy::Rank1(Dim::Symbolic("BLOCK_SIZE".to_string()))
    );

    let a_ty = k.expr_types.get(&assign_value(body, 4).span()).unwrap();
    assert_eq!(
        *a_ty,
        TileTy::Rank1(Dim::Symbolic("BLOCK_SIZE".to_string()))
    );
}

const MATMUL: &str = r#"
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = acc + tl.dot(a, b)
        a_ptrs = a_ptrs + BLOCK_K
        b_ptrs = b_ptrs + BLOCK_K * N
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)
"#;

#[test]
fn matmul_shaped_kernel_builds_rank2_tiles_via_reshape() {
    let module = parse_ok(MATMUL);
    let (shapes, diags) = check_triton(&module);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    let k = &shapes[0];
    let body = &module.kernels[0].body;

    let block_m = Dim::Symbolic("BLOCK_M".to_string());
    let block_n = Dim::Symbolic("BLOCK_N".to_string());
    let block_k = Dim::Symbolic("BLOCK_K".to_string());

    // a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]  ->  [BLOCK_M, BLOCK_K]
    let a_ptrs_ty = k.expr_types.get(&assign_value(body, 5).span()).unwrap();
    assert_eq!(*a_ptrs_ty, TileTy::Rank2(block_m.clone(), block_k.clone()));

    // b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]  ->  [BLOCK_K, BLOCK_N]
    let b_ptrs_ty = k.expr_types.get(&assign_value(body, 6).span()).unwrap();
    assert_eq!(*b_ptrs_ty, TileTy::Rank2(block_k.clone(), block_n.clone()));

    // acc = tl.zeros((BLOCK_M, BLOCK_N), ...)  ->  [BLOCK_M, BLOCK_N], same symbolic names as
    // the reshaped tiles above (the identity-simplification in `combine_dim` is what keeps
    // an arange-derived dimension and a bare constexpr reference rendering identically).
    let acc_ty = k.expr_types.get(&assign_value(body, 7).span()).unwrap();
    assert_eq!(*acc_ty, TileTy::Rank2(block_m.clone(), block_n.clone()));

    // mask = (rm[:, None] < M) & (rn[None, :] < N)  ->  [BLOCK_M, BLOCK_N]
    let mask_idx = body.len() - 2;
    let mask_ty = k
        .expr_types
        .get(&assign_value(body, mask_idx).span())
        .unwrap();
    assert_eq!(*mask_ty, TileTy::Rank2(block_m, block_n));
}

#[test]
fn unequal_concrete_ranges_report_a_clean_diagnostic_not_a_panic() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def bad_kernel(x_ptr, y_ptr):
    a = tl.arange(0, 16)
    b = tl.arange(0, 32)
    c = a + b
"#,
    );
    let (shapes, diags) = check_triton(&module);
    assert_eq!(shapes.len(), 1);
    assert!(codes(&diags).contains(&ECode::TileShapeMismatch));
}

#[test]
fn rank_mismatch_between_two_tile_operands_is_refused() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def bad_rank_kernel(x_ptr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    tile = row[:, None] * BLOCK
    bad = row + tile
"#,
    );
    let (_shapes, diags) = check_triton(&module);
    assert!(codes(&diags).contains(&ECode::TileShapeMismatch));
}

#[test]
fn partial_slice_subscript_is_refused_cleanly() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def partial_slice_kernel(x_ptr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    bad = row[0:BLOCK]
"#,
    );
    let (_shapes, diags) = check_triton(&module);
    assert!(codes(&diags).contains(&ECode::TileConstructUnsupported));
}

#[test]
fn matmul_operator_is_refused_in_favor_of_tl_dot() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def matmul_operator_kernel(a_ptr, b_ptr, BLOCK: tl.constexpr):
    a = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    b = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    c = a @ b
"#,
    );
    let (_shapes, diags) = check_triton(&module);
    assert!(codes(&diags).contains(&ECode::TileConstructUnsupported));
}

#[test]
fn reshape_beyond_rank_two_is_refused() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def rank3_kernel(x_ptr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    tile = row[:, None]
    cube = tile[:, :, None]
"#,
    );
    let (_shapes, diags) = check_triton(&module);
    assert!(codes(&diags).contains(&ECode::TileConstructUnsupported));
}

#[test]
fn constexpr_param_with_literal_default_resolves_concretely() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def const_block_kernel(x_ptr, BLOCK: tl.constexpr = 128):
    idx = tl.arange(0, BLOCK)
"#,
    );
    let (shapes, diags) = check_triton(&module);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    let body = &module.kernels[0].body;
    let idx_ty = shapes[0]
        .expr_types
        .get(&assign_value(body, 0).span())
        .unwrap();
    assert_eq!(*idx_ty, TileTy::Rank1(Dim::Const(128)));
}

#[test]
fn constexpr_param_without_default_stays_symbolic_not_guessed() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def sym_block_kernel(x_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
"#,
    );
    let (shapes, diags) = check_triton(&module);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    let body = &module.kernels[0].body;
    let idx_ty = shapes[0]
        .expr_types
        .get(&assign_value(body, 0).span())
        .unwrap();
    assert_eq!(*idx_ty, TileTy::Rank1(Dim::Symbolic("BLOCK".to_string())));
}

#[test]
fn undefined_name_reference_reports_e301() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def undef_kernel(x_ptr):
    y = totally_undefined + x_ptr
"#,
    );
    let (_shapes, diags) = check_triton(&module);
    assert!(codes(&diags).contains(&ECode::UndefinedSymbol));
}

#[test]
fn docstring_and_pass_produce_no_diagnostics() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def documented_kernel(x_ptr):
    """A kernel with a docstring; this should not be treated as a tile value."""
    pass
"#,
    );
    let (_shapes, diags) = check_triton(&module);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn many_kernels_in_one_module_are_each_checked_independently() {
    let module = parse_ok(
        r#"
import triton
import triton.language as tl


@triton.jit
def first(x_ptr, BLOCK: tl.constexpr):
    a = tl.arange(0, BLOCK)


@triton.jit
def second(y_ptr, N: tl.constexpr):
    b = tl.arange(0, N)
"#,
    );
    let (shapes, diags) = check_triton(&module);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
    assert_eq!(shapes.len(), 2);
    assert_eq!(shapes[0].name, "first");
    assert_eq!(shapes[1].name, "second");
}
