// Coverage: one representative case per BIR op category this backend claims to support (hand-
// built modules, mirroring `basalt-ptx`'s own test style), a phi/if-else control-flow test, the
// real frontend/sema/passes pipeline over `tests/kernels/vector_add.cu`, determinism, and the
// refusals this backend actually takes. Every test that inspects emitted structure does so via
// `rspirv::dr::load_bytes` re-parsing this backend's own output — this both exercises the
// "does it parse back as well-formed SPIR-V" round-trip and gives a robust, non-string-matching
// way to assert on opcodes/decorations/capabilities.
//
// This suite does not itself shell out to `spirv-val` (that would make these tests fail on any
// machine without SPIRV-Tools installed, and `basalt-spirv` has no dependency on it); real
// `spirv-val` validation of this backend's output for several of these same module shapes was
// run separately, by hand, during development — see the module header's "Validation tier"
// section for the exact modules checked and the one real bug that run caught.

use super::*;
use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{Block, Inst, InstId};
use rspirv::dr;
use rspirv::spirv::Op as SOp;

fn wrap(f: Function) -> Module {
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

/// A single-block function: `insts` in order, terminated by `term`.
fn simple_fn(name: &str, params: Vec<Ty>, insts: Vec<Inst>, term: Term) -> Function {
    let ids = (0..insts.len() as u32).map(InstId).collect();
    Function {
        is_kernel: true,
        name: name.into(),
        params,
        ret: Ty::Void,
        insts,
        blocks: vec![Block { insts: ids, term }],
    }
}

fn emit_bytes(module: &Module) -> Vec<u8> {
    assert_eq!(Spirv.supports(module), Support::Supported);
    let artifact = Spirv
        .emit(module, &EmitOpts::default())
        .expect("emit succeeds for a supported module");
    artifact
        .as_bytes()
        .expect("the SPIR-V backend emits a bytes payload, never text")
        .to_vec()
}

/// Emits `module` and re-parses the result with `rspirv`'s own loader — the "well-formed
/// SPIR-V" half of this backend's validation story (see the module header).
fn emit_and_parse(module: &Module) -> dr::Module {
    let bytes = emit_bytes(module);
    dr::load_bytes(&bytes).expect("this backend's own output must parse back as well-formed SPIR-V")
}

fn all_opcodes(func: &dr::Function) -> Vec<SOp> {
    func.blocks
        .iter()
        .flat_map(|b| b.instructions.iter().map(|i| i.class.opcode))
        .collect()
}

fn only_fn(m: &dr::Module) -> &dr::Function {
    assert_eq!(m.functions.len(), 1);
    &m.functions[0]
}

// ---- glcompute path -------------------------------------------------------------------------

fn glcompute_opts() -> EmitOpts {
    EmitOpts {
        target_variant: Some("glcompute".to_string()),
        ..EmitOpts::default()
    }
}

fn emit_glcompute_bytes(module: &Module) -> Vec<u8> {
    let artifact = Spirv
        .emit(module, &glcompute_opts())
        .expect("glcompute emit succeeds for a supported module");
    artifact
        .as_bytes()
        .expect("the SPIR-V backend emits a bytes payload, never text")
        .to_vec()
}

fn emit_and_parse_glcompute(module: &Module) -> dr::Module {
    let bytes = emit_glcompute_bytes(module);
    dr::load_bytes(&bytes)
        .expect("this backend's glcompute output must parse back as well-formed SPIR-V")
}

/// Builds the exact four-instruction shape `basalt-sema`'s `lower_index_lvalue` produces for
/// `base[index]` at `elem_ty` (see `emit/glcompute.rs`'s header): widen `index_param` (an `i32`
/// parameter) to `i64`, multiply by `elem_ty`'s real byte size, add to `base_param` (a
/// `Ty::Ptr(Global)` parameter). The four instructions land at `InstId`s `base_id..base_id + 4`;
/// the last (`base_id + 3`) is the address a `Load`/`Store` should reference.
fn index_addr_insts(base_id: u32, base_param: u32, index_param: u32, elem_ty: Ty) -> Vec<Inst> {
    let i64t = Ty::Scalar(Scalar::I64);
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let stride = match elem_ty {
        Ty::Scalar(Scalar::I32 | Scalar::F32) => 4,
        Ty::Scalar(Scalar::I64 | Scalar::F64) => 8,
        _ => panic!("index_addr_insts only supports i32/i64/f32/f64 element types"),
    };
    vec![
        Inst {
            ty: i64t,
            op: Op::Cast(
                CastOp::Sext,
                Ty::Scalar(Scalar::I32),
                ValRef::Param(index_param),
            ),
        },
        Inst {
            ty: i64t,
            op: Op::ConstInt(stride),
        },
        Inst {
            ty: i64t,
            op: Op::Bin(
                BinOp::Mul,
                ValRef::Val(InstId(base_id)),
                ValRef::Val(InstId(base_id + 1)),
            ),
        },
        Inst {
            ty: ptrt,
            op: Op::Bin(
                BinOp::Add,
                ValRef::Param(base_param),
                ValRef::Val(InstId(base_id + 2)),
            ),
        },
    ]
}

fn decoration_targets(
    m: &dr::Module,
    decoration: rspirv::spirv::Decoration,
) -> Vec<(u32, Vec<dr::Operand>)> {
    m.annotations
        .iter()
        .filter(|i| {
            i.class.opcode == SOp::Decorate
                && i.operands.get(1) == Some(&dr::Operand::Decoration(decoration))
        })
        .map(|i| {
            let target = match i.operands[0] {
                dr::Operand::IdRef(id) => id,
                _ => unreachable!(),
            };
            (target, i.operands[2..].to_vec())
        })
        .collect()
}

/// `OpMemberDecorate` targets — a distinct opcode from `OpDecorate` (see `decoration_targets`),
/// used for `Offset` (both storage-buffer and push-constant struct members).
fn member_decoration_targets(
    m: &dr::Module,
    decoration: rspirv::spirv::Decoration,
) -> Vec<(u32, u32, Vec<dr::Operand>)> {
    m.annotations
        .iter()
        .filter(|i| {
            i.class.opcode == SOp::MemberDecorate
                && i.operands.get(2) == Some(&dr::Operand::Decoration(decoration))
        })
        .map(|i| {
            let target = match i.operands[0] {
                dr::Operand::IdRef(id) => id,
                _ => unreachable!(),
            };
            let member = match i.operands[1] {
                dr::Operand::LiteralBit32(n) => n,
                _ => unreachable!(),
            };
            (target, member, i.operands[3..].to_vec())
        })
        .collect()
}

fn variables_with_storage_class(m: &dr::Module, class: rspirv::spirv::StorageClass) -> Vec<u32> {
    m.types_global_values
        .iter()
        .filter(|i| {
            i.class.opcode == SOp::Variable
                && i.operands.first() == Some(&dr::Operand::StorageClass(class))
        })
        .map(|i| i.result_id.unwrap())
        .collect()
}

#[test]
fn glcompute_variant_none_and_other_strings_keep_the_kernel_path_byte_identical() {
    let module = lower_vector_add();
    let kernel_bytes = emit_bytes(&module);
    let other_variant = Spirv
        .emit(
            &module,
            &EmitOpts {
                target_variant: Some("not-glcompute".to_string()),
                ..EmitOpts::default()
            },
        )
        .expect("emit succeeds")
        .as_bytes()
        .unwrap()
        .to_vec();
    assert_eq!(
        kernel_bytes, other_variant,
        "only target_variant == Some(\"glcompute\") may change this backend's output"
    );
}

#[test]
fn glcompute_module_declares_shader_capability_logical_addressing_and_glsl450() {
    let f = simple_fn("empty", vec![], vec![], Term::Ret(None));
    let module = emit_and_parse_glcompute(&wrap(f));
    let caps: Vec<_> = module
        .capabilities
        .iter()
        .map(|i| match i.operands[0] {
            dr::Operand::Capability(c) => c,
            _ => unreachable!(),
        })
        .collect();
    for want in [
        rspirv::spirv::Capability::Shader,
        rspirv::spirv::Capability::Int64,
        rspirv::spirv::Capability::Float64,
    ] {
        assert!(caps.contains(&want), "missing capability {want:?}");
    }
    assert!(
        !caps.contains(&rspirv::spirv::Capability::Kernel),
        "glcompute must not declare the Kernel-path capability"
    );
    let mm = module
        .memory_model
        .expect("memory model instruction present");
    assert_eq!(
        mm.operands[0],
        dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Logical)
    );
    assert_eq!(
        mm.operands[1],
        dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::GLSL450)
    );
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(
        module.entry_points[0].operands[0],
        dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::GLCompute)
    );
}

#[test]
fn glcompute_resource_binding_abi_binds_pointers_in_order_and_packs_the_scalar_push_constant() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    // params: (a: ptr, b: ptr, n: i32) — two storage-buffer bindings, one push-constant scalar.
    let mut insts = index_addr_insts(0, 0, 2, f32t);
    insts.push(Inst {
        ty: f32t,
        op: Op::Load {
            ptr: ValRef::Val(InstId(3)),
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    });
    insts.extend(index_addr_insts(5, 1, 2, f32t));
    insts.push(Inst {
        ty: f32t,
        op: Op::Load {
            ptr: ValRef::Val(InstId(8)),
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    });
    let f = simple_fn(
        "two_bindings",
        vec![ptrt, ptrt, i32t],
        insts,
        Term::Ret(None),
    );
    let module = emit_and_parse_glcompute(&wrap(f));

    let storage_vars = variables_with_storage_class(&module, rspirv::spirv::StorageClass::Uniform);
    assert_eq!(
        storage_vars.len(),
        2,
        "expected exactly two storage-buffer bindings"
    );
    let push_const_vars =
        variables_with_storage_class(&module, rspirv::spirv::StorageClass::PushConstant);
    assert_eq!(
        push_const_vars.len(),
        1,
        "expected exactly one push-constant block"
    );

    let descriptor_sets = decoration_targets(&module, rspirv::spirv::Decoration::DescriptorSet);
    for &var in &storage_vars {
        let (_, ops) = descriptor_sets
            .iter()
            .find(|&&(t, _)| t == var)
            .expect("every storage-buffer variable has a DescriptorSet decoration");
        assert_eq!(ops[0], dr::Operand::LiteralBit32(0));
    }
    let bindings = decoration_targets(&module, rspirv::spirv::Decoration::Binding);
    let mut binding_numbers: Vec<u32> = storage_vars
        .iter()
        .map(|&var| {
            let (_, ops) = bindings.iter().find(|&&(t, _)| t == var).unwrap();
            match ops[0] {
                dr::Operand::LiteralBit32(n) => n,
                _ => unreachable!(),
            }
        })
        .collect();
    binding_numbers.sort_unstable();
    assert_eq!(
        binding_numbers,
        vec![0, 1],
        "the two pointer parameters must be bound at 0 and 1, in declared order"
    );

    let array_strides = decoration_targets(&module, rspirv::spirv::Decoration::ArrayStride);
    assert!(
        array_strides
            .iter()
            .any(|(_, ops)| ops[0] == dr::Operand::LiteralBit32(4)),
        "the f32 buffer element must be declared ArrayStride 4"
    );
    let offsets = member_decoration_targets(&module, rspirv::spirv::Decoration::Offset);
    assert!(
        offsets
            .iter()
            .any(|&(_, member, ref ops)| member == 0 && ops[0] == dr::Operand::LiteralBit32(0)),
        "the push-constant scalar's one member must be packed at offset 0: {offsets:?}"
    );
}

#[test]
fn glcompute_recognizes_the_sema_index_shape_and_lowers_to_a_real_access_chain() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let mut insts = index_addr_insts(0, 0, 1, f32t);
    insts.push(Inst {
        ty: f32t,
        op: Op::Load {
            ptr: ValRef::Val(InstId(3)),
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    });
    let f = simple_fn("recognized_load", vec![ptrt, i32t], insts, Term::Ret(None));
    let module = wrap(f);
    assert!(
        Spirv.emit(&module, &glcompute_opts()).is_ok(),
        "the exact shape basalt-sema's lower_index_lvalue produces must be recognized"
    );
    let parsed = emit_and_parse_glcompute(&module);
    let ops = all_opcodes(only_fn(&parsed));
    assert!(ops.contains(&SOp::AccessChain), "missing {ops:?}");
    assert!(
        !ops.contains(&SOp::ConvertUToPtr) && !ops.contains(&SOp::ConvertPtrToU),
        "glcompute has no raw-address representation to convert to/from: {ops:?}"
    );
}

#[test]
fn glcompute_refuses_an_unrecognized_address_shape_not_a_guess() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let f32t = Ty::Scalar(Scalar::F32);
    // A bare parameter used directly as an address — no Bin::Add/Bin::Mul index computation at
    // all. Perfectly fine under the Kernel path's raw-integer pointer model (`supports()` still
    // reports this module as Supported); not a shape glcompute recognizes.
    let f = simple_fn(
        "bare_address",
        vec![ptrt],
        vec![Inst {
            ty: f32t,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Supported,
        "the Kernel path's own supports() is unaffected by glcompute-specific shape rules"
    );
    let err = Spirv
        .emit(&module, &glcompute_opts())
        .expect_err("an address that isn't the recognized Add(ptr_param, Mul(idx, stride)) shape must be refused, not guessed");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedFeature);
}

#[test]
fn glcompute_refuses_a_pointer_parameter_after_a_scalar_parameter() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn("bad_order", vec![i32t, ptrt], vec![], Term::Ret(None));
    let module = wrap(f);
    let err = Spirv
        .emit(&module, &glcompute_opts())
        .expect_err("a pointer parameter after a scalar parameter breaks this ABI's push-constant/binding split");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedFeature);
}

#[test]
fn glcompute_refuses_a_wrong_stride_not_a_guess() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    // Same shape as `index_addr_insts`, but the stride constant (8) does not match f32's real
    // size (4) — a mismatch this backend must catch rather than trust.
    let i64t = Ty::Scalar(Scalar::I64);
    let insts = vec![
        Inst {
            ty: i64t,
            op: Op::Cast(CastOp::Sext, i32t, ValRef::Param(1)),
        },
        Inst {
            ty: i64t,
            op: Op::ConstInt(8),
        },
        Inst {
            ty: i64t,
            op: Op::Bin(BinOp::Mul, ValRef::Val(InstId(0)), ValRef::Val(InstId(1))),
        },
        Inst {
            ty: ptrt,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Val(InstId(2))),
        },
        Inst {
            ty: f32t,
            op: Op::Load {
                ptr: ValRef::Val(InstId(3)),
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
    ];
    let f = simple_fn("wrong_stride", vec![ptrt, i32t], insts, Term::Ret(None));
    let module = wrap(f);
    let err = Spirv.emit(&module, &glcompute_opts()).expect_err(
        "a stride constant that does not match the accessed type's real size must be refused",
    );
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedFeature);
}

#[test]
fn magic_number_and_header_are_well_formed() {
    let f = simple_fn("empty", vec![], vec![], Term::Ret(None));
    let bytes = emit_bytes(&wrap(f));
    assert!(
        bytes.len() >= 20,
        "a SPIR-V module is at least a 5-word header"
    );
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic, 0x0723_0203, "wrong SPIR-V magic number");
    let module = dr::load_bytes(&bytes).expect("well-formed SPIR-V");
    let header = module.header.expect("a parsed module always has a header");
    assert_eq!(header.magic_number, 0x0723_0203);
    assert!(header.bound > 1, "bound must exceed every id actually used");
}

#[test]
fn declares_kernel_capability_addresses_int64_float64_and_physical64_opencl() {
    let f = simple_fn("empty", vec![], vec![], Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    let caps: Vec<_> = module
        .capabilities
        .iter()
        .map(|i| match i.operands[0] {
            dr::Operand::Capability(c) => c,
            _ => unreachable!(),
        })
        .collect();
    for want in [
        rspirv::spirv::Capability::Kernel,
        rspirv::spirv::Capability::Addresses,
        rspirv::spirv::Capability::Int64,
        rspirv::spirv::Capability::Float64,
    ] {
        assert!(caps.contains(&want), "missing capability {want:?}");
    }
    let mm = module
        .memory_model
        .expect("memory model instruction present");
    assert_eq!(mm.class.opcode, SOp::MemoryModel);
    assert_eq!(
        mm.operands[0],
        dr::Operand::AddressingModel(rspirv::spirv::AddressingModel::Physical64)
    );
    assert_eq!(
        mm.operands[1],
        dr::Operand::MemoryModel(rspirv::spirv::MemoryModel::OpenCL)
    );
}

#[test]
fn entry_point_and_local_size_execution_mode_are_present() {
    let f = simple_fn("my_kernel", vec![], vec![], Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    assert_eq!(module.entry_points.len(), 1);
    let ep = &module.entry_points[0];
    assert_eq!(
        ep.operands[0],
        dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::Kernel)
    );
    assert_eq!(
        ep.operands[2],
        dr::Operand::LiteralString("my_kernel".to_string())
    );

    assert_eq!(module.execution_modes.len(), 1);
    let em = &module.execution_modes[0];
    assert_eq!(
        em.operands[1],
        dr::Operand::ExecutionMode(rspirv::spirv::ExecutionMode::LocalSize)
    );
    assert_eq!(em.operands[2], dr::Operand::LiteralBit32(1));
    assert_eq!(em.operands[3], dr::Operand::LiteralBit32(1));
    assert_eq!(em.operands[4], dr::Operand::LiteralBit32(1));
}

#[test]
fn bin_arithmetic_emits_expected_opcodes() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Sub, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Mul, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Div, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Rem, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::And, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Shl, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Ashr, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Lshr, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FAdd, ValRef::Param(2), ValRef::Param(3)),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FDiv, ValRef::Param(2), ValRef::Param(3)),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FRem, ValRef::Param(2), ValRef::Param(3)),
        },
    ];
    let f = simple_fn(
        "arith",
        vec![i32t, i32t, f32t, f32t],
        insts,
        Term::Ret(None),
    );
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    for want in [
        SOp::IAdd,
        SOp::ISub,
        SOp::IMul,
        SOp::SDiv,
        SOp::SRem,
        SOp::BitwiseAnd,
        SOp::ShiftLeftLogical,
        SOp::ShiftRightArithmetic,
        SOp::ShiftRightLogical,
        SOp::FAdd,
        SOp::FDiv,
        SOp::FRem,
    ] {
        assert!(ops.contains(&want), "missing {want:?} in {ops:?}");
    }
}

#[test]
fn pointer_arithmetic_is_plain_64_bit_integer_arithmetic() {
    let i64t = Ty::Scalar(Scalar::I64);
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let insts = vec![Inst {
        ty: ptrt,
        op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
    }];
    let f = simple_fn("ptr_arith", vec![ptrt, i64t], insts, Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    // One `OpConvertPtrToU` (the parameter prologue) then a plain `OpIAdd` — no
    // `OpPtrAccessChain` anywhere, matching this backend's raw-integer pointer model.
    assert!(ops.contains(&SOp::ConvertPtrToU));
    assert!(ops.contains(&SOp::IAdd));
    assert!(!ops.contains(&SOp::PtrAccessChain));
}

#[test]
fn compares_emit_the_right_opcode_per_predicate() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(ICmpPred::Slt, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(ICmpPred::Ult, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::FCmp(FCmpPred::Olt, f32t, ValRef::Param(2), ValRef::Param(3)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::FCmp(FCmpPred::Uno, f32t, ValRef::Param(2), ValRef::Param(3)),
        },
    ];
    let f = simple_fn("cmp", vec![i32t, i32t, f32t, f32t], insts, Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::SLessThan));
    assert!(ops.contains(&SOp::ULessThan));
    assert!(ops.contains(&SOp::FOrdLessThan));
    assert!(ops.contains(&SOp::Unordered));
}

#[test]
fn bool_typed_compare_widens_via_select_for_ordered_predicates() {
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![Inst {
        ty: i1t,
        op: Op::ICmp(ICmpPred::Slt, i1t, ValRef::Param(0), ValRef::Param(1)),
    }];
    let f = simple_fn("boolcmp", vec![i1t, i1t], insts, Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::Select));
    assert!(ops.contains(&SOp::SLessThan));
}

#[test]
fn casts_emit_the_expected_conversion_opcodes() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i64t = Ty::Scalar(Scalar::I64);
    let f32t = Ty::Scalar(Scalar::F32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![
        Inst {
            ty: i1t,
            op: Op::Cast(CastOp::Trunc, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: i64t,
            op: Op::Cast(CastOp::Sext, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: i64t,
            op: Op::Cast(CastOp::Zext, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: f32t,
            op: Op::Cast(CastOp::SiToFp, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::FpToSi, f32t, ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::Bitcast, f32t, ValRef::Param(1)),
        },
    ];
    let f = simple_fn("casts", vec![i32t, f32t], insts, Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::BitwiseAnd)); // part of Trunc(i32->i1)
    assert!(ops.contains(&SOp::SConvert));
    assert!(ops.contains(&SOp::UConvert));
    assert!(ops.contains(&SOp::ConvertSToF));
    assert!(ops.contains(&SOp::ConvertFToS));
    assert!(ops.contains(&SOp::Bitcast));
}

#[test]
fn load_store_on_global_convert_the_raw_address_then_access() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: f32t,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
        Inst {
            ty: Ty::Void,
            op: Op::Store {
                ptr: ValRef::Param(0),
                val: ValRef::Val(InstId(0)),
                ty: f32t,
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
    ];
    let f = simple_fn("loadstore", vec![ptrt], insts, Term::Ret(None));
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::ConvertUToPtr));
    assert!(ops.contains(&SOp::Load));
    assert!(ops.contains(&SOp::Store));
}

#[test]
fn local_address_space_load_is_refused_not_guessed() {
    let ptrt = Ty::Ptr(AddrSpace::Local);
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn(
        "localload",
        vec![ptrt],
        vec![Inst {
            ty: i32t,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Local,
                align: 4,
                volatile: false,
            },
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedAddressSpace)
    );
    assert!(Spirv.emit(&module, &EmitOpts::default()).is_err());
}

#[test]
fn all_twelve_gpu_index_ops_read_the_right_builtin() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ops = [
        Op::TidX,
        Op::TidY,
        Op::TidZ,
        Op::BidX,
        Op::BidY,
        Op::BidZ,
        Op::BdimX,
        Op::BdimY,
        Op::BdimZ,
        Op::GdimX,
        Op::GdimY,
        Op::GdimZ,
    ];
    let insts = ops.into_iter().map(|op| Inst { ty: i32t, op }).collect();
    let f = simple_fn("indices", vec![], insts, Term::Ret(None));
    let module = emit_and_parse(&wrap(f));

    // Three real `Input` builtin variables (WorkgroupId/LocalInvocationId/NumWorkgroups), each
    // with exactly one BuiltIn decoration. `BdimX/Y/Z` touch no builtin at all (see the module
    // header's "GPU index op" section for why).
    let builtin_decorations: Vec<rspirv::spirv::BuiltIn> = module
        .annotations
        .iter()
        .filter_map(|i| {
            if i.operands.get(1)
                == Some(&dr::Operand::Decoration(rspirv::spirv::Decoration::BuiltIn))
            {
                match i.operands[2] {
                    dr::Operand::BuiltIn(b) => Some(b),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect();
    for want in [
        rspirv::spirv::BuiltIn::WorkgroupId,
        rspirv::spirv::BuiltIn::LocalInvocationId,
        rspirv::spirv::BuiltIn::NumWorkgroups,
    ] {
        assert!(
            builtin_decorations.contains(&want),
            "missing BuiltIn {want:?} decoration"
        );
    }
    assert!(!builtin_decorations.contains(&rspirv::spirv::BuiltIn::WorkgroupSize));

    let ops = all_opcodes(only_fn(&module));
    assert!(ops.iter().filter(|&&o| o == SOp::Load).count() >= 3); // one per Input builtin read
                                                                   // One `OpCompositeExtract` per `Tid`/`Bid`/`Gdim` axis (9), none for `Bdim` (a plain constant).
    assert_eq!(
        ops.iter().filter(|&&o| o == SOp::CompositeExtract).count(),
        9
    );

    // The entry point's interface lists exactly the three Input variables this kernel touched.
    let ep = &module.entry_points[0];
    assert_eq!(
        ep.operands.len() - 3,
        3,
        "expected exactly 3 interface variables"
    );
}

#[test]
fn barrier_emits_a_real_control_barrier() {
    let f = simple_fn(
        "barrier",
        vec![],
        vec![Inst {
            ty: Ty::Void,
            op: Op::Barrier,
        }],
        Term::Ret(None),
    );
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::ControlBarrier));
}

#[test]
fn if_then_emits_selection_merge_and_branch_conditional() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![
        Inst {
            ty: i1t,
            op: Op::ICmp(ICmpPred::Slt, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: Ty::Void,
            op: Op::Barrier,
        },
    ];
    let f = Function {
        is_kernel: true,
        name: "guarded".into(),
        params: vec![i32t, i32t],
        ret: Ty::Void,
        insts,
        blocks: vec![
            Block {
                insts: vec![InstId(0)],
                term: Term::CondBr(ValRef::Val(InstId(0)), BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![InstId(1)],
                term: Term::Br(BlockId(2)),
            },
            Block {
                insts: vec![],
                term: Term::Ret(None),
            },
        ],
    };
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::SelectionMerge));
    assert!(ops.contains(&SOp::BranchConditional));
    assert!(ops.contains(&SOp::Branch));
}

#[test]
fn if_else_phi_resolves_via_a_real_op_phi() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![
        Inst {
            ty: i1t,
            op: Op::ICmp(ICmpPred::Sgt, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::ConstInt(10),
        },
        Inst {
            ty: i32t,
            op: Op::ConstInt(20),
        },
        Inst {
            ty: i32t,
            op: Op::Phi(vec![
                (BlockId(1), ValRef::Val(InstId(1))),
                (BlockId(2), ValRef::Val(InstId(2))),
            ]),
        },
    ];
    let f = Function {
        is_kernel: true,
        name: "phi_fn".into(),
        params: vec![i32t, i32t],
        ret: Ty::Void,
        insts,
        blocks: vec![
            Block {
                insts: vec![InstId(0)],
                term: Term::CondBr(ValRef::Val(InstId(0)), BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![InstId(1)],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![InstId(2)],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![InstId(3)],
                term: Term::Ret(None),
            },
        ],
    };
    let module = emit_and_parse(&wrap(f));
    let ops = all_opcodes(only_fn(&module));
    assert!(ops.contains(&SOp::Phi));
    assert!(ops.contains(&SOp::SelectionMerge));
}

#[test]
fn switch_is_refused_not_guessed() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        is_kernel: true,
        name: "hasswitch".into(),
        params: vec![i32t],
        ret: Ty::Void,
        insts: vec![],
        blocks: vec![
            Block {
                insts: vec![],
                term: Term::Switch(ValRef::Param(0), BlockId(1), vec![(0, BlockId(1))]),
            },
            Block {
                insts: vec![],
                term: Term::Ret(None),
            },
        ],
    };
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
}

#[test]
fn a_back_edge_loop_is_refused_not_guessed() {
    let f = Function {
        is_kernel: true,
        name: "hasloop".into(),
        params: vec![],
        ret: Ty::Void,
        insts: vec![],
        blocks: vec![Block {
            insts: vec![],
            term: Term::Br(BlockId(0)),
        }],
    };
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
}

#[test]
fn non_reconverging_branch_arms_are_refused_not_guessed() {
    let i1t = Ty::Scalar(Scalar::I1);
    let f = Function {
        is_kernel: true,
        name: "divergent".into(),
        params: vec![i1t],
        ret: Ty::Void,
        insts: vec![],
        blocks: vec![
            Block {
                insts: vec![],
                term: Term::CondBr(ValRef::Param(0), BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![],
                term: Term::Ret(None),
            },
            Block {
                insts: vec![],
                term: Term::Ret(None),
            },
        ],
    };
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
}

#[test]
fn f16_is_refused_not_guessed() {
    let f16t = Ty::Scalar(Scalar::F16);
    let f = simple_fn(
        "f16_add",
        vec![f16t, f16t],
        vec![Inst {
            ty: f16t,
            op: Op::Bin(BinOp::FAdd, ValRef::Param(0), ValRef::Param(1)),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedType)
    );
    assert!(Spirv.emit(&module, &EmitOpts::default()).is_err());
}

#[test]
fn vector_types_are_refused_not_guessed() {
    let vecty = Ty::Vec(Scalar::F32, 4);
    let f = simple_fn(
        "vecadd",
        vec![vecty, vecty],
        vec![Inst {
            ty: vecty,
            op: Op::Bin(BinOp::FAdd, ValRef::Param(0), ValRef::Param(1)),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedType)
    );
}

#[test]
fn atomics_are_refused_not_guessed() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn(
        "atomicadd",
        vec![ptrt, i32t],
        vec![Inst {
            ty: i32t,
            op: Op::Atomic(
                basalt_bir::AtomicOp::Add,
                ValRef::Param(0),
                ValRef::Param(1),
                AddrSpace::Global,
            ),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
}

#[test]
fn shuffle_is_refused_not_guessed() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn(
        "shuf",
        vec![i32t, i32t],
        vec![Inst {
            ty: i32t,
            op: Op::Shuffle(
                basalt_bir::ShuffleKind::Idx,
                ValRef::Param(0),
                ValRef::Param(1),
            ),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
}

#[test]
fn mma_is_refused_not_guessed() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let f = simple_fn(
        "usesmma",
        vec![ptrt, ptrt, ptrt, ptrt],
        vec![Inst {
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
                layout_a: basalt_bir::MmaLayout::RowMajor,
                layout_b: basalt_bir::MmaLayout::RowMajor,
            },
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedOp)
    );
    let err = Spirv
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedOp);
}

/// P13-T1b's kernel-launch/CUDA-Runtime-API ops are sema-only today (see
/// `basalt_bir::Op::KernelLaunch`'s own doc comment) — every backend refuses them cleanly.
#[test]
fn kernel_launch_and_cuda_runtime_api_ops_are_refused_not_guessed() {
    let f = simple_fn(
        "launch_stub",
        vec![],
        vec![Inst {
            ty: Ty::Void,
            op: Op::CudaDeviceSynchronize,
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedOp)
    );
    let err = Spirv
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedOp);
}

/// `Op::Call` (P13-T-calls-i) has no lowering in this backend yet — refuse cleanly rather
/// than falling through to the scalar per-op emitters, which have no case for it.
#[test]
fn function_call_is_refused_not_guessed() {
    let f = simple_fn(
        "caller",
        vec![Ty::Scalar(Scalar::I32)],
        vec![Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Call {
                func: "callee".to_string(),
                args: vec![ValRef::Param(0)],
            },
        }],
        Term::Ret(Some(ValRef::Val(InstId(0)))),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedOp)
    );
    let err = Spirv
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedOp);
}

#[test]
fn non_kernel_function_is_refused_not_silently_emitted_as_an_entry_point() {
    // The live gap this test guards: every function in a module becomes its own
    // `OpEntryPoint` (see `emit_module`), so a non-kernel function must be refused rather
    // than silently miscompiled as a launchable one.
    let mut f = simple_fn("host_fn", vec![], vec![], Term::Ret(None));
    f.is_kernel = false;
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
    let err = Spirv
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedFeature);
}

#[test]
fn glcompute_refuses_a_non_kernel_function_too() {
    let mut f = simple_fn("host_fn", vec![], vec![], Term::Ret(None));
    f.is_kernel = false;
    let module = wrap(f);
    let err = Spirv
        .emit(&module, &glcompute_opts())
        .expect_err("glcompute must refuse a non-kernel function, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedFeature);
}

#[test]
fn pointer_bitcast_is_refused_not_guessed() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i64t = Ty::Scalar(Scalar::I64);
    let f = simple_fn(
        "ptrbitcast",
        vec![ptrt],
        vec![Inst {
            ty: i64t,
            op: Op::Cast(CastOp::Bitcast, ptrt, ValRef::Param(0)),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Spirv.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedType)
    );
}

#[test]
fn emit_is_deterministic() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn(
        "det",
        vec![i32t],
        vec![Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(0)),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    let a = emit_bytes(&module);
    let b = emit_bytes(&module);
    assert_eq!(a, b);
}

// ---- real pipeline: tests/kernels/vector_add.cu -------------------------------------------

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
fn vector_add_lowers_to_a_well_formed_spirv_module_via_the_real_pipeline() {
    let module = lower_vector_add();
    assert_eq!(Spirv.supports(&module), Support::Supported);
    let parsed = emit_and_parse(&module);

    assert_eq!(parsed.entry_points.len(), 1);
    assert_eq!(
        parsed.entry_points[0].operands[2],
        dr::Operand::LiteralString("vector_add".to_string())
    );
    let ops = all_opcodes(only_fn(&parsed));
    assert!(ops.contains(&SOp::Load));
    assert!(ops.contains(&SOp::Store));
    assert!(ops.contains(&SOp::IMul));
    assert!(ops.contains(&SOp::IAdd));
    assert!(ops.contains(&SOp::SLessThan));
    assert!(ops.contains(&SOp::BranchConditional));
    assert!(ops.contains(&SOp::FAdd));
}

#[test]
fn vector_add_emit_is_deterministic_through_the_real_pipeline() {
    let module = lower_vector_add();
    let a = emit_bytes(&module);
    let b = emit_bytes(&module);
    assert_eq!(a, b);
}

// ---- real pipeline through the glcompute path, mirroring the two tests above ---------------

#[test]
fn vector_add_lowers_to_a_well_formed_glcompute_spirv_module_via_the_real_pipeline() {
    let module = lower_vector_add();
    let parsed = emit_and_parse_glcompute(&module);

    assert_eq!(parsed.entry_points.len(), 1);
    assert_eq!(
        parsed.entry_points[0].operands[2],
        dr::Operand::LiteralString("vector_add".to_string())
    );
    assert_eq!(
        parsed.entry_points[0].operands[0],
        dr::Operand::ExecutionModel(rspirv::spirv::ExecutionModel::GLCompute)
    );

    let storage_vars = variables_with_storage_class(&parsed, rspirv::spirv::StorageClass::Uniform);
    assert_eq!(
        storage_vars.len(),
        3,
        "vector_add's a/b/c pointer parameters must become three storage-buffer bindings"
    );
    let push_const_vars =
        variables_with_storage_class(&parsed, rspirv::spirv::StorageClass::PushConstant);
    assert_eq!(
        push_const_vars.len(),
        1,
        "vector_add's n parameter must become the one push-constant block"
    );

    let ops = all_opcodes(only_fn(&parsed));
    assert!(ops.contains(&SOp::AccessChain));
    assert!(ops.contains(&SOp::Load));
    assert!(ops.contains(&SOp::Store));
    assert!(ops.contains(&SOp::IMul));
    assert!(ops.contains(&SOp::IAdd));
    assert!(ops.contains(&SOp::SLessThan));
    assert!(ops.contains(&SOp::BranchConditional));
    assert!(ops.contains(&SOp::FAdd));
    assert!(
        !ops.contains(&SOp::ConvertUToPtr) && !ops.contains(&SOp::ConvertPtrToU),
        "the glcompute path has no raw-address representation, unlike the Kernel path: {ops:?}"
    );
}

#[test]
fn vector_add_glcompute_emit_is_deterministic_through_the_real_pipeline() {
    let module = lower_vector_add();
    let a = emit_glcompute_bytes(&module);
    let b = emit_glcompute_bytes(&module);
    assert_eq!(a, b);
}
