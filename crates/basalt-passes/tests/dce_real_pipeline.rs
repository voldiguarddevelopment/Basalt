// Runs `eliminate_dead_code` over BIR lowered by the real frontend/sema pipeline (rather than
// a hand-built fixture). `basalt-sema`'s lowering pass documents (see that crate's `lower.rs`
// header, "trailing dead block") that it always opens a fresh block after a
// `return`/`break`/`continue`/`goto`, whether or not anything ever branches into it, and says
// a later cleanup pass is expected to fold these away.
//
// `tests/kernels/vector_add.cu` (the kernel `ARCHITECTURE.md` cites as the canonical example)
// does not actually exercise that path: its body has no explicit `return`/`break`/`continue`/
// `goto`, only an `if` with no `else`, whose merge block is always reachable from both arms.
// Lowering it produces no genuinely unreachable block (confirmed below), so it is used here
// only for the round-trip/idempotence checks, which do apply to any kernel. The trailing-
// dead-block cleanup itself is demonstrated on a minimal kernel, still lowered through the
// same real `basalt-frontend-c` -> `basalt-sema` pipeline, whose last statement is an explicit
// `return` — exactly the shape the lowering pass's own comment describes.

use basalt_frontend_c::PpOpts;
use basalt_passes::eliminate_dead_code;

const VECTOR_ADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/vector_add.cu"
));

const TAIL_RETURN_SRC: &str = r#"
__global__ void tail_return(int *out, int n) {
    out[0] = n;
    return;
}
"#;

fn lower_src(src: &str) -> basalt_bir::Module {
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(src, &PpOpts::default());
    assert!(pp_errors.is_empty(), "preprocess errors: {pp_errors:?}");
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(parse_errors.is_empty(), "parse errors: {parse_errors:?}");
    let sema_diags = basalt_sema::check(&tu);
    assert!(sema_diags.is_empty(), "sema diagnostics: {sema_diags:?}");
    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering diagnostics: {lower_diags:?}"
    );
    module
}

#[test]
fn trailing_dead_block_after_explicit_return_is_folded_away() {
    let module = lower_src(TAIL_RETURN_SRC);
    let before = &module.funcs[0];
    let before_blocks = before.blocks.len();
    assert_eq!(
        before_blocks,
        2,
        "fixture assumption broken: expected lowering to open exactly one unreachable block \
         after the explicit `return`: {}",
        basalt_bir::print(&module)
    );

    let out = eliminate_dead_code(&module);
    let after = &out.funcs[0];

    assert!(
        after.blocks.len() < before_blocks,
        "expected the lowering pass's documented trailing dead block to be folded away: \
         before = {before_blocks}, after = {}\n--- before ---\n{}\n--- after ---\n{}",
        after.blocks.len(),
        basalt_bir::print(&module),
        basalt_bir::print(&out)
    );
    assert_eq!(after.blocks.len(), 1);

    let text = basalt_bir::print(&out);
    let reparsed =
        basalt_bir::parse(&text).expect("parse(print(eliminate_dead_code(m))) must parse");
    assert_eq!(
        reparsed, out,
        "parse(print(m)) != m on the real tail_return lowering"
    );
}

#[test]
fn vector_add_has_no_dead_block_but_dce_round_trips_and_is_idempotent() {
    // `vector_add.cu` never hits the trailing-dead-block path (see this file's header), so
    // this test only checks the parts of DCE that must hold on *any* real lowering: it is a
    // round-trippable no-op change here, and running it twice must be idempotent.
    let module = lower_src(VECTOR_ADD_SRC);
    let before = &module.funcs[0];

    let once = eliminate_dead_code(&module);
    let after = &once.funcs[0];
    assert_eq!(
        after.blocks.len(),
        before.blocks.len(),
        "vector_add.cu's blocks are all reachable and none are known dead pre-SSA: {}",
        basalt_bir::print(&once)
    );

    let twice = eliminate_dead_code(&once);
    assert_eq!(
        once, twice,
        "running DCE again on already-clean BIR must be a no-op"
    );

    let text = basalt_bir::print(&once);
    let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
    assert_eq!(
        reparsed, once,
        "parse(print(m)) != m on the real vector_add.cu lowering"
    );
}

#[test]
fn vector_add_after_ssa_round_trips_through_dce() {
    // Once `construct_ssa` has promoted the local/param slots to real SSA values, DCE gets a
    // genuine chance to sweep away the load/store scaffolding `ssa.rs` leaves behind for
    // anything it didn't need. This is the two-pass pipeline `ARCHITECTURE.md` §7 describes:
    // SSA construction, then DCE, both BIR-to-BIR.
    let module = lower_src(VECTOR_ADD_SRC);
    let ssa_form = basalt_passes::construct_ssa(&module);

    let out = eliminate_dead_code(&ssa_form);
    assert!(
        out.funcs[0].insts.len() <= ssa_form.funcs[0].insts.len(),
        "DCE must never grow the instruction count"
    );

    let text = basalt_bir::print(&out);
    let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
    assert_eq!(reparsed, out, "parse(print(m)) != m after SSA + DCE");
}
