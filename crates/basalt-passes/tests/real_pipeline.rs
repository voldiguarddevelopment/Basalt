// Runs `construct_ssa` over BIR lowered by the real frontend/sema pipeline (rather than a
// hand-built fixture), on the same `vector_add.cu` kernel `ARCHITECTURE.md` cites as the
// canonical example.

use basalt_bir::Op;
use basalt_frontend_c::PpOpts;
use basalt_passes::construct_ssa;

const VECTOR_ADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/vector_add.cu"
));

fn lower_vector_add() -> basalt_bir::Module {
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(VECTOR_ADD_SRC, &PpOpts::default());
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

fn local_or_param_mem_ops(f: &basalt_bir::Function) -> usize {
    f.insts
        .iter()
        .filter(|i| match &i.op {
            Op::Load { space, .. } | Op::Store { space, .. } => {
                matches!(
                    space,
                    basalt_bir::AddrSpace::Local | basalt_bir::AddrSpace::Param
                )
            }
            _ => false,
        })
        .count()
}

#[test]
fn vector_add_locals_and_params_promote_away() {
    let module = lower_vector_add();
    let before = &module.funcs[0];
    let before_local_param_ops = local_or_param_mem_ops(before);
    assert!(
        before_local_param_ops > 0,
        "fixture assumption broken: expected the un-promoted lowering to still route through \
         local/param slots"
    );

    let out = construct_ssa(&module);
    let after = &out.funcs[0];

    assert_eq!(
        local_or_param_mem_ops(after),
        0,
        "every local/param slot access should have promoted away: {}",
        basalt_bir::print(&out)
    );
    assert!(
        after.insts.len() < before.insts.len(),
        "expected the instruction count to drop once redundant loads/stores are eliminated"
    );

    // Global-space memory (the actual `float*` dereferences) must survive untouched.
    let global_ops_before = before
        .insts
        .iter()
        .filter(|i| {
            matches!(
                &i.op,
                Op::Load {
                    space: basalt_bir::AddrSpace::Global,
                    ..
                } | Op::Store {
                    space: basalt_bir::AddrSpace::Global,
                    ..
                }
            )
        })
        .count();
    let global_ops_after = after
        .insts
        .iter()
        .filter(|i| {
            matches!(
                &i.op,
                Op::Load {
                    space: basalt_bir::AddrSpace::Global,
                    ..
                } | Op::Store {
                    space: basalt_bir::AddrSpace::Global,
                    ..
                }
            )
        })
        .count();
    assert_eq!(global_ops_before, global_ops_after);

    let text = basalt_bir::print(&out);
    let reparsed = basalt_bir::parse(&text).expect("parse(print(construct_ssa(m))) must parse");
    assert_eq!(
        reparsed, out,
        "parse(print(m)) != m on the real vector_add.cu lowering"
    );
}
