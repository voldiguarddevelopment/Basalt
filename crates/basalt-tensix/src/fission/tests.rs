// Coverage: the real `tests/kernels/vector_add.cu` pipeline through the fission pass end to
// end (channels, NoC arcs, both kernel bodies, determinism), plus the one extra refusal this
// pass adds on top of `emit.rs`'s own (`check_fissionable`'s load-and-store conflict) and a
// couple of hand-built shape checks (no load params, no store params).

use super::*;
use crate::tdf::print_tdf;
use basalt_bir::{Block, Function, InstId, Op, Scalar, Term, Ty, ValRef};

fn wrap(f: Function) -> Module {
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

const VECTOR_ADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/vector_add.cu"
));

fn lower_vector_add() -> Module {
    let (tokens, pp_errors) =
        basalt_frontend_c::preprocess(VECTOR_ADD_SRC, &basalt_frontend_c::PpOpts::default());
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
    basalt_passes::optimize(&module)
}

#[test]
fn vector_add_fissions_into_four_regions_and_real_channels() {
    let module = lower_vector_add();
    let prog = build_tdf(&module).expect("vector_add.cu is fissionable");

    assert_eq!(prog.grid_x, 2);
    assert_eq!(prog.grid_y, 2);
    assert_eq!(prog.regions.len(), 4);

    // p0 (a), p1 (b) are load-only -> reader-produced channels; p2 (c) is store-only -> a
    // compute_writer-internal scratch channel. p3 (n) is a plain scalar, no channel at all.
    assert_eq!(prog.channels.len(), 3);
    let cb_in0 = prog
        .channels
        .iter()
        .find(|c| c.param == 0)
        .expect("p0 has a channel");
    assert_eq!(cb_in0.producer, Role::Reader);
    assert_eq!(cb_in0.consumer, Role::ComputeWriter);
    assert!(!cb_in0.is_internal_scratch());

    let cb_in1 = prog
        .channels
        .iter()
        .find(|c| c.param == 1)
        .expect("p1 has a channel");
    assert_eq!(cb_in1.producer, Role::Reader);

    let cb_out0 = prog
        .channels
        .iter()
        .find(|c| c.param == 2)
        .expect("p2 has a channel");
    assert_eq!(cb_out0.producer, Role::ComputeWriter);
    assert_eq!(cb_out0.consumer, Role::ComputeWriter);
    assert!(cb_out0.is_internal_scratch());
    assert_eq!(cb_out0.cb_index, 16);

    assert!(!prog.channels.iter().any(|c| c.param == 3));

    assert_eq!(prog.noc_arcs.len(), 3);
    assert!(prog
        .noc_arcs
        .iter()
        .any(|a| a.param == 0 && a.dir == NocDir::Read));
    assert!(prog
        .noc_arcs
        .iter()
        .any(|a| a.param == 2 && a.dir == NocDir::Write));
}

#[test]
fn reader_kernel_reads_load_params_at_the_region_offset() {
    let module = lower_vector_add();
    let prog = build_tdf(&module).expect("vector_add.cu is fissionable");
    let r = &prog.reader_kernel;

    assert!(r.starts_with("void kernel_main() {\n"));
    assert!(r.contains("uint32_t start_tile_id = get_arg_val<uint32_t>(0);"));
    assert!(r.contains("uint32_t n_tiles = get_arg_val<uint32_t>(1);"));
    assert!(r.contains("uint32_t p0_dram = get_arg_val<uint32_t>(2);"));
    assert!(r.contains("uint32_t p1_dram = get_arg_val<uint32_t>(3);"));
    assert!(!r.contains("p2_dram"), "p2 is store-only, not read here");
    assert!(r.contains("cb_reserve_back(0, 1);"));
    assert!(r.contains("cb_reserve_back(1, 1);"));
    assert!(r.contains("noc_async_read(p0_gen.get_noc_addr(start_tile_id)"));
    assert!(r.contains("noc_async_read(p1_gen.get_noc_addr(start_tile_id)"));
    assert!(r.contains("noc_async_read_barrier();"));
    assert!(r.contains("cb_push_back(0, 1);"));
    assert!(r.contains("cb_push_back(1, 1);"));
}

#[test]
fn compute_writer_kernel_shifts_channel_pointers_and_writes_back() {
    let module = lower_vector_add();
    let prog = build_tdf(&module).expect("vector_add.cu is fissionable");
    let w = &prog.compute_writer_kernel;

    assert!(w.starts_with("void kernel_main() {\n"));
    assert!(w.contains("uint32_t start_tile_id = get_arg_val<uint32_t>(0);"));
    assert!(w.contains("uint32_t n_tiles = get_arg_val<uint32_t>(1);"));
    assert!(w.contains("uint32_t nthreads = get_arg_val<uint32_t>(2);"));
    assert!(w.contains("uint32_t p2_dram = get_arg_val<uint32_t>(3);"));
    assert!(w.contains("int32_t p3 = (int32_t)get_arg_val<uint32_t>(4);"));
    assert!(w.contains("cb_wait_front(0, 1);"));
    assert!(w.contains("cb_wait_front(1, 1);"));
    assert!(w.contains(
        "uint8_t* p0 = (uint8_t*)(uintptr_t)cb_in0_addr - (uintptr_t)start_tile_id * sizeof(float);"
    ));
    assert!(w.contains("cb_reserve_back(16, 1);"));
    assert!(w.contains(
        "uint8_t* p2 = (uint8_t*)(uintptr_t)cb_out0_addr - (uintptr_t)start_tile_id * sizeof(float);"
    ));
    assert!(w.contains(
        "for (uint32_t __tid = start_tile_id; __tid < start_tile_id + n_tiles; ++__tid) {"
    ));
    assert!(w.contains("noc_async_write(cb_out0_addr, p2_gen.get_noc_addr(start_tile_id)"));
    assert!(w.contains("noc_async_write_barrier();"));
    assert!(w.contains("cb_pop_front(0, 1);"));
    assert!(w.contains("cb_pop_front(1, 1);"));
    // never crosses a kernel boundary — see the module header.
    assert!(!w.contains("cb_wait_front(16"));
    assert!(!w.contains("cb_push_back(16"));
}

#[test]
fn dump_is_deterministic_and_contains_both_kernels() {
    let module = lower_vector_add();
    let a = print_tdf(&build_tdf(&module).unwrap());
    let b = print_tdf(&build_tdf(&module).unwrap());
    assert_eq!(a, b);
    assert!(a.contains("tdf module for `vector_add`"));
    assert!(a.contains("grid 2x2 (4 regions, one core each)"));
    assert!(a.contains("region r0 core(0,0)"));
    assert!(a.contains("region r3 core(1,1)"));
    assert!(a.contains("==== reader kernel"));
    assert!(a.contains("==== compute_writer kernel"));
}

#[test]
fn load_and_store_on_the_same_pointer_param_refuses() {
    let ptrt = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![
        basalt_bir::Inst {
            ty: i32t,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: basalt_bir::AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
        basalt_bir::Inst {
            ty: Ty::Void,
            op: Op::Store {
                ptr: ValRef::Param(0),
                val: ValRef::Val(InstId(0)),
                ty: i32t,
                space: basalt_bir::AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
    ];
    let f = Function {
        name: "inplace".into(),
        params: vec![ptrt],
        ret: Ty::Void,
        insts,
        blocks: vec![Block {
            insts: vec![InstId(0), InstId(1)],
            term: Term::Ret(None),
        }],
    };
    let module = wrap(f);
    match check_fissionable(&module) {
        Err(diag) => assert_eq!(diag.code, ECode::UnsupportedOp),
        Ok(()) => panic!("a pointer that is both read and written should refuse fission"),
    }
}

#[test]
fn a_kernel_with_no_pointer_params_still_fissions_with_no_channels() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "scalar_only".into(),
        params: vec![i32t],
        ret: Ty::Void,
        insts: vec![],
        blocks: vec![Block {
            insts: vec![],
            term: Term::Ret(None),
        }],
    };
    let prog = build_tdf(&wrap(f)).expect("a scalar-only kernel is trivially fissionable");
    assert!(prog.channels.is_empty());
    assert!(prog.noc_arcs.is_empty());
    assert!(!prog.reader_kernel.contains("cb_reserve_back"));
    assert!(prog.compute_writer_kernel.contains("int32_t p0"));
}
