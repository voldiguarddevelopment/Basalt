// The x86-64 regalloc backend: the CPU performance path sibling of `oracle.rs`. Same SysV
// calling convention, same per-thread native-loop model, same op/type/refusal surface — but
// values live in real general-purpose and XMM registers wherever `basalt_passes::regalloc`
// assigned one, falling back to a real stack spill slot only where it spilled. See
// `oracle.rs`'s own header for the parts of the design (the native thread loop, address-space
// handling, calling convention, signed div/rem, `select`-via-branch, f16 refusal) that are
// unchanged here; this header covers only what differs.
//
// # Pipeline
//
// `emit` runs `basalt_passes::ssa::construct_ssa` (promoting local/param/shared/constant
// memory traffic to direct SSA values and real `phi`s) then `basalt_passes::regalloc::allocate`
// (assigning every value a `Location`) before lowering. `global`-space memory and any slot
// `construct_ssa` declines to promote are still handled as real memory operations, exactly the
// way the oracle does, via this backend's own small `const_addr_disp` slot table.
//
// # Register budget
//
// This backend restricts itself to caller-saved GP/XMM registers only, same reasoning as the
// oracle: no callee-saved save/restore bookkeeping. The 9 caller-saved GP registers other than
// `rsp`/`rbp` (which are never candidates — `rsp` is the stack pointer, `rbp` the frame
// pointer) split three ways:
//
//   - `r11`, one register, reserved for the whole function as the thread-loop's own counter.
//     BIR has no notion of the wrapping loop (see `oracle.rs`'s header on why that's fine), so
//     nothing about it goes through `allocate` — it lives in `r11` for the entire function,
//     never spilled, never treated as scratch. This is a real improvement over the oracle
//     (which spills its loop counter to memory every iteration): `tid.x` is now a bare
//     register read.
//   - `rax`, `rcx`, `rdx`, `r10`: four scratch registers this backend's own instruction
//     templates freely clobber while reloading a spilled operand or computing an address —
//     the same role the oracle's `rax`/`rcx`/`rdx`/`r10` scratch set plays, with `rbx` (the
//     oracle's fifth scratch register, but callee-saved) replaced by reusing `rdx` for the
//     "stash old atomic value" role instead.
//   - `rsi`, `rdi`, `r8`, `r9`: the four registers handed to `basalt_passes::regalloc::allocate`
//     as the abstract integer register pool (`num_int_regs = 4`). These double as SysV integer
//     argument registers, which is safe: every incoming argument is staged to memory before
//     any of them is reused (see "function entry" below).
//
// All 8 XMM argument registers are caller-saved by convention (SysV has no callee-saved XMM
// registers at all), so the split there is: `xmm0..xmm2` scratch (mirroring the oracle's own
// float scratch set exactly, including `xmm2`'s role in `frem`/atomic-min-max/fp-to-ui/ui-to-fp
// emulation), `xmm3..xmm7` the abstract float register pool (`num_float_regs = 5`). `xmm8`
// upward are never touched, matching the encoder's own documented scope.
//
// # Function entry
//
// Moving an incoming SysV argument register directly into its allocator-assigned `Location`
// risks the classic parallel-move hazard (param A's incoming register is param B's assigned
// register and vice versa). This backend sidesteps it exactly like the oracle sidesteps its
// own version of the same shape of problem: every incoming parameter is first spilled to its
// own dedicated temporary stack slot (`Frame::param_stage`), then, in a second pass, loaded
// from that safe temporary into its real assigned `Location` (a register move, or a spill
// store). A few extra instructions, unconditionally correct.
//
// # Phi resolution
//
// `basalt_passes::regalloc::allocate` deliberately does not insert the copies that resolve a
// phi's operands into its own location (see that module's header) — this backend does it here,
// at the end of every predecessor block, right before the jump into the block containing the
// phi. The same parallel-move hazard from function entry recurs when a block has more than one
// phi: naively copying each phi's incoming value straight into that phi's own location, one at
// a time, can clobber a register that a *later* phi in the same copy list still needs to read
// from. This is solved the same way as function entry — stage every incoming value to its own
// temporary slot first (`Frame::phi_stage`, one slot per `Phi` instruction in the function),
// then copy from staging into every phi's real location in a second pass. Simple, and
// unconditionally correct regardless of what the allocator happened to assign.

use std::collections::HashMap;

use basalt_backend::{
    write_elf_object, Architecture, Artifact, ArtifactKind, Backend, ElfObjectSpec, EmitOpts,
    Endianness, Support,
};
use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, CastOp, FCmpPred, Function, ICmpPred, InstId, Module, Op, Scalar,
    Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};
use basalt_passes::{allocate, construct_ssa, Allocation, Location, RegClass, ValueId};

use crate::enc::{
    cc, AluOp, Enc, Rm, ShiftKind, SseArith, R10, R11, R8, R9, RAX, RBP, RCX, RDI, RDX, RSI, W,
};

/// The abstract integer register pool handed to `basalt_passes::regalloc::allocate`, in index
/// order (`allocate`'s `Reg(Int, i)` maps to `INT_POOL[i]`). See the module header.
const INT_POOL: [u8; 4] = [RSI, RDI, R8, R9];
/// The abstract float (XMM) register pool, same role as `INT_POOL` for `RegClass::Float`.
const FLOAT_POOL: [u8; 5] = [3, 4, 5, 6, 7];

/// The x86-64 regalloc backend: real registers wherever the allocator assigned one, real
/// stack spills otherwise. See the module header for the full design; `name()` returns
/// `"x86-regalloc"`.
#[derive(Debug, Default, Clone, Copy)]
pub struct X86Regalloc;

impl Backend for X86Regalloc {
    fn name(&self) -> &'static str {
        "x86-regalloc"
    }

    fn supports(&self, module: &Module) -> Support {
        match check_module(module) {
            Ok(_) => Support::Supported,
            Err(diag) => Support::Unsupported(diag.code),
        }
    }

    fn emit(&self, module: &Module, _opts: &EmitOpts) -> Result<Artifact, Diag> {
        check_module(module)?;
        let ssa_module = construct_ssa(module);
        let f = &ssa_module.funcs[0];
        let alloc = allocate(f, INT_POOL.len() as u32, FLOAT_POOL.len() as u32);
        let text = emit_function(f, &alloc)?;
        let spec = ElfObjectSpec::new(
            Architecture::X86_64,
            Endianness::Little,
            f.name.clone(),
            text,
        );
        let bytes = write_elf_object(&spec)?;
        Ok(Artifact::bytes(ArtifactKind::Object, bytes))
    }
}

/// Single source of truth for what this backend refuses, shared verbatim by `supports()` and
/// `emit()`. Mirrors `oracle.rs`'s own `check_module` exactly (same op/type/feature surface —
/// this matters for diffing the two backends against each other) but is a separate copy: it
/// runs on the module as originally handed in, before `construct_ssa`, and this backend must
/// never share code with, or otherwise touch, the oracle's own module.
fn check_module(module: &Module) -> Result<&Function, Diag> {
    if module.funcs.len() != 1 {
        return Err(Diag::new(ECode::UnsupportedFeature)
            .with_arg("multi-function module: ElfObjectSpec names exactly one symbol"));
    }
    let f = &module.funcs[0];

    if matches!(f.ret, Ty::Vec(..)) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed return value"));
    }
    if f.params.iter().any(|t| matches!(t, Ty::Vec(..))) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed parameter"));
    }
    if ty_is_f16(f.ret) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 return value needs F16C"));
    }
    if f.params.iter().any(|&t| ty_is_f16(t)) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 parameter needs F16C"));
    }
    if classify_params(&f.params).is_none() {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
            "argument classes overflow SysV's 6 integer / 8 SSE registers (incl. trailing nthreads)",
        ));
    }

    for inst in &f.insts {
        if matches!(inst.ty, Ty::Vec(..)) {
            return Err(
                Diag::new(ECode::UnsupportedType).with_arg("vector-typed instruction result")
            );
        }
        if ty_is_f16(inst.ty) {
            return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 arithmetic needs F16C"));
        }
        match &inst.op {
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                    "warp-collective op has no meaning under one-thread-at-a-time execution",
                ));
            }
            Op::Cast(_, sty, _) | Op::FCmp(_, sty, _, _) if ty_is_f16(*sty) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 operand needs F16C"));
            }
            Op::Store { ty: sty, .. } if ty_is_f16(*sty) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 store needs F16C"));
            }
            Op::Mma { .. } => {
                return Err(Diag::new(ECode::UnsupportedOp)
                    .with_arg("mma has no lowering in the regalloc-based x86-64 backend yet"));
            }
            _ => {}
        }
    }

    Ok(f)
}

fn ty_is_f16(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F16))
}

fn is_float(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F32) | Ty::Scalar(Scalar::F64))
}

fn is_f64(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F64))
}

/// See `oracle.rs`'s identically-named function: the exact byte width every memory access for
/// a value of this type uses.
fn width_of(ty: Ty) -> W {
    match ty {
        Ty::Scalar(Scalar::I1 | Scalar::I8) => W::B1,
        Ty::Scalar(Scalar::I16) => W::B2,
        Ty::Scalar(Scalar::I32 | Scalar::F32) => W::B4,
        Ty::Scalar(Scalar::I64 | Scalar::F64) | Ty::Ptr(_) => W::B8,
        Ty::Scalar(Scalar::F16) | Ty::Vec(..) | Ty::Void => {
            unreachable!("width_of called on a type check_module should have refused")
        }
    }
}

fn local_like(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Shared | AddrSpace::Constant
    )
}

fn space_tag(space: AddrSpace) -> u8 {
    match space {
        AddrSpace::Global => 0,
        AddrSpace::Shared => 1,
        AddrSpace::Constant => 2,
        AddrSpace::Local => 3,
        AddrSpace::Param => 4,
    }
}

fn block_label(id: u32) -> String {
    format!("bb{id}")
}

#[derive(Clone, Copy)]
enum ArgLoc {
    Int(u8),
    Sse(u8),
}

/// SysV integer-class argument registers, in passing order. A private duplicate of
/// `oracle.rs`'s `INT_ARG_REGS`/`SSE_ARG_REGS`/`classify_params` (down to the algorithm and
/// doc comment) rather than a shared import, so this backend never has to touch `oracle.rs` to
/// stay in sync — see the module header.
const INT_ARG_REGS: [u8; 6] = [RDI, RSI, RDX, RCX, R8, R9];
const SSE_ARG_REGS: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// Classifies `params` into the SysV integer/SSE argument sequence and returns the location
/// the trailing `nthreads` parameter would land in. `None` means the signature overflows the
/// register-passing convention.
fn classify_params(params: &[Ty]) -> Option<(Vec<ArgLoc>, ArgLoc)> {
    let mut int_idx = 0usize;
    let mut sse_idx = 0usize;
    let mut locs = Vec::with_capacity(params.len());
    for &ty in params {
        if is_float(ty) {
            if sse_idx >= SSE_ARG_REGS.len() {
                return None;
            }
            locs.push(ArgLoc::Sse(SSE_ARG_REGS[sse_idx]));
            sse_idx += 1;
        } else {
            if int_idx >= INT_ARG_REGS.len() {
                return None;
            }
            locs.push(ArgLoc::Int(INT_ARG_REGS[int_idx]));
            int_idx += 1;
        }
    }
    if int_idx >= INT_ARG_REGS.len() {
        return None;
    }
    Some((locs, ArgLoc::Int(INT_ARG_REGS[int_idx])))
}

/// This function's real native stack frame. Unlike the oracle's (one slot per instruction,
/// unconditionally), most SSA values here never touch memory at all — this frame only reserves
/// space for what genuinely needs a stack home: the temporary parameter/phi staging slots (see
/// the module header), `nthreads`'s home (never promoted to a register — only `tid.x`'s loop
/// counter earns that treatment, see the header), the non-void return value's home, one slot
/// per spilled value in each register class, and the same synthesized local/param/shared/
/// constant address table the oracle uses for any slot `construct_ssa` declined to promote.
struct Frame {
    param_stage: Vec<i32>,
    nthreads_home: i32,
    retval_home: Option<i32>,
    int_spill: Vec<i32>,
    float_spill: Vec<i32>,
    phi_stage: HashMap<u32, i32>,
    const_addr_disp: HashMap<(u8, i64), i32>,
    frame_size: i32,
}

fn next_slot(offset: &mut i32) -> i32 {
    *offset += 8;
    -*offset
}

impl Frame {
    fn build(f: &Function, alloc: &Allocation) -> Frame {
        let mut offset: i32 = 0;

        let param_stage: Vec<i32> = f.params.iter().map(|_| next_slot(&mut offset)).collect();
        let nthreads_home = next_slot(&mut offset);
        let retval_home = if matches!(f.ret, Ty::Void) {
            None
        } else {
            Some(next_slot(&mut offset))
        };
        let int_spill: Vec<i32> = (0..alloc.num_int_spills)
            .map(|_| next_slot(&mut offset))
            .collect();
        let float_spill: Vec<i32> = (0..alloc.num_float_spills)
            .map(|_| next_slot(&mut offset))
            .collect();

        let mut phi_stage: HashMap<u32, i32> = HashMap::new();
        for (idx, inst) in f.insts.iter().enumerate() {
            if matches!(inst.op, Op::Phi(_)) {
                phi_stage.insert(idx as u32, next_slot(&mut offset));
            }
        }

        let mut const_addr_disp = HashMap::new();
        for inst in &f.insts {
            if let (Op::ConstInt(n), Ty::Ptr(space)) = (&inst.op, inst.ty) {
                if local_like(space) {
                    let key = (space_tag(space), *n);
                    const_addr_disp
                        .entry(key)
                        .or_insert_with(|| next_slot(&mut offset));
                }
            }
        }

        let frame_size = (offset + 15) & !15;
        Frame {
            param_stage,
            nthreads_home,
            retval_home,
            int_spill,
            float_spill,
            phi_stage,
            const_addr_disp,
            frame_size,
        }
    }
}

/// Per-edge phi lowering: for each `(from_block, to_block)` edge, every phi instruction in
/// `to_block` that needs a copy from `from_block`'s own incoming value, before the jump that
/// takes that edge. Unlike `oracle.rs`'s version of this table (which stores a destination
/// slot directly), this stores the phi's own `InstId` — the copy is emitted through the same
/// `copy_value` helper `Select` uses, since "copy a value into instruction X's own result
/// location" is exactly the same operation either way.
type PhiCopies = HashMap<(u32, u32), Vec<(InstId, ValRef, Ty)>>;

fn build_phi_copies(f: &Function) -> PhiCopies {
    let mut map: PhiCopies = HashMap::new();
    for (bidx, block) in f.blocks.iter().enumerate() {
        for &inst_id in &block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            if let Op::Phi(preds) = &inst.op {
                for &(pred_block, val) in preds {
                    map.entry((pred_block.0, bidx as u32))
                        .or_default()
                        .push((inst_id, val, inst.ty));
                }
            }
        }
    }
    map
}

/// Where one value's operand/result actually lives at the machine level: a real register, or
/// an address in this function's own frame.
#[derive(Clone, Copy)]
enum OpLoc {
    Reg(u8),
    Mem(i32),
}

struct CodeGen<'a> {
    f: &'a Function,
    frame: Frame,
    alloc: &'a Allocation,
    enc: Enc,
    label_counter: u32,
    phi_copies: PhiCopies,
}

fn emit_function(f: &Function, alloc: &Allocation) -> Result<Vec<u8>, Diag> {
    let (param_locs, nthreads_loc) =
        classify_params(&f.params).expect("check_module already validated the signature");

    let frame = Frame::build(f, alloc);
    let phi_copies = build_phi_copies(f);
    let mut cg = CodeGen {
        f,
        frame,
        alloc,
        enc: Enc::new(),
        label_counter: 0,
        phi_copies,
    };

    cg.enc.push_reg(RBP);
    cg.enc.mov_rbp_rsp();
    cg.enc.sub_rsp_imm(cg.frame.frame_size);

    // Phase 1: every incoming argument register (and `nthreads`) is staged to its own
    // temporary slot, never touched again after phase 2 — see the module header.
    for (i, loc) in param_locs.iter().enumerate() {
        let disp = cg.frame.param_stage[i];
        let ty = f.params[i];
        match *loc {
            ArgLoc::Int(r) => cg.enc.mov_rbp_reg(width_of(ty), disp, r),
            ArgLoc::Sse(r) => {
                if is_f64(ty) {
                    cg.enc.movsd_store(Rm::RbpDisp(disp), r);
                } else {
                    cg.enc.movss_store(Rm::RbpDisp(disp), r);
                }
            }
        }
    }
    if let ArgLoc::Int(r) = nthreads_loc {
        cg.enc.mov_rbp_reg(W::B8, cg.frame.nthreads_home, r);
    }

    // The thread-loop counter lives in `r11` for the whole function — see the module header.
    cg.enc.mov_reg_imm32(W::B8, R11, 0);

    cg.enc.label("__loop_check");
    cg.enc.mov_reg_rbp(W::B8, RAX, cg.frame.nthreads_home);
    cg.enc.alu_reg_reg(AluOp::Cmp, W::B8, R11, RAX);
    cg.enc.jcc(cc::GE, "__loop_end");

    // Phase 2: move each staged parameter into its real allocator-assigned location. This
    // must run at the top of *every* iteration, not once before the loop: a param's register
    // (or spill slot) is only guaranteed to hold its value up to that param's own last use —
    // `allocate` freely lets a later, unrelated value in the same body reuse that location
    // once the param is dead, which is correct for one logical execution but would otherwise
    // leave a param's location holding leftover garbage from the *previous* iteration's tail
    // by the time this iteration's body reads it. Re-staging every iteration undoes that.
    cg.place_params();

    for (bidx, block) in f.blocks.iter().enumerate() {
        cg.enc.label(&block_label(bidx as u32));
        for &inst_id in &block.insts {
            cg.lower_inst(inst_id);
        }
        cg.lower_term(bidx as u32, &block.term);
    }

    cg.enc.label("__loop_incr");
    cg.enc.alu_reg_imm32(AluOp::Add, W::B8, R11, 1);
    cg.enc.jmp("__loop_check");

    cg.enc.label("__loop_end");
    if !matches!(f.ret, Ty::Void) {
        let disp = cg
            .frame
            .retval_home
            .expect("non-void return always gets a retval home");
        if is_float(f.ret) {
            if is_f64(f.ret) {
                cg.enc.movsd_load(0, Rm::RbpDisp(disp));
            } else {
                cg.enc.movss_load(0, Rm::RbpDisp(disp));
            }
        } else {
            cg.enc.mov_reg_rbp(width_of(f.ret), RAX, disp);
        }
    }
    cg.enc.mov_rsp_rbp();
    cg.enc.pop_reg(RBP);
    cg.enc.ret();

    Ok(cg.enc.finish())
}

impl<'a> CodeGen<'a> {
    fn fresh_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("__{prefix}_{}", self.label_counter)
    }

    /// Moves every parameter from its permanent staging slot (`Frame::param_stage`, written
    /// once by the real SysV argument registers before the loop even starts) into its real
    /// allocator-assigned location. Called at the top of every loop iteration — see the call
    /// site in `emit_function` for why once, in the prologue, is not enough.
    fn place_params(&mut self) {
        for (i, &ty) in self.f.params.iter().enumerate() {
            let disp = self.frame.param_stage[i];
            let vid = ValueId::Param(i as u32);
            if is_float(ty) {
                let f64_ = is_f64(ty);
                if f64_ {
                    self.enc.movsd_load(0, Rm::RbpDisp(disp));
                } else {
                    self.enc.movss_load(0, Rm::RbpDisp(disp));
                }
                self.place_xmm_vid(vid, 0, f64_);
            } else {
                let w = width_of(ty);
                self.enc.mov_reg_rbp(w, RAX, disp);
                self.place_gpr_vid(vid, RAX, w);
            }
        }
    }

    fn valref_ty(&self, v: ValRef) -> Ty {
        match v {
            ValRef::Param(i) => self.f.params[i as usize],
            ValRef::Val(id) => self.f.insts[id.0 as usize].ty,
        }
    }

    fn gpr_loc(&self, vid: ValueId) -> OpLoc {
        match self.alloc.locations[&vid] {
            Location::Reg(RegClass::Int, idx) => OpLoc::Reg(INT_POOL[idx as usize]),
            Location::Spill(RegClass::Int, idx) => OpLoc::Mem(self.frame.int_spill[idx as usize]),
            Location::Reg(RegClass::Float, _) | Location::Spill(RegClass::Float, _) => {
                unreachable!("gpr_loc called on a float-class value")
            }
        }
    }

    fn xmm_loc(&self, vid: ValueId) -> OpLoc {
        match self.alloc.locations[&vid] {
            Location::Reg(RegClass::Float, idx) => OpLoc::Reg(FLOAT_POOL[idx as usize]),
            Location::Spill(RegClass::Float, idx) => {
                OpLoc::Mem(self.frame.float_spill[idx as usize])
            }
            Location::Reg(RegClass::Int, _) | Location::Spill(RegClass::Int, _) => {
                unreachable!("xmm_loc called on an int-class value")
            }
        }
    }

    // ---- operand reads ----------------------------------------------------------------

    fn load_gpr(&mut self, v: ValRef, dst: u8, w: W) {
        match self.gpr_loc(ValueId::from(v)) {
            OpLoc::Reg(r) => {
                if r != dst {
                    self.enc.mov_reg_reg(w, dst, r);
                }
            }
            OpLoc::Mem(disp) => self.enc.mov_reg_rbp(w, dst, disp),
        }
    }

    fn load_gpr_zx(&mut self, v: ValRef, dst: u8, dst_w: W, src_w: W) {
        match self.gpr_loc(ValueId::from(v)) {
            OpLoc::Reg(r) => match src_w {
                W::B1 | W::B2 => self.enc.movzx(dst_w, src_w, dst, Rm::Direct(r)),
                // A plain write already zero-extends the upper 32 bits of a 64-bit register
                // natively; no movzx opcode exists for this case (see `oracle.rs`).
                W::B4 | W::B8 => self.enc.mov_reg_reg(src_w, dst, r),
            },
            OpLoc::Mem(disp) => match src_w {
                W::B1 | W::B2 => self.enc.movzx(dst_w, src_w, dst, Rm::RbpDisp(disp)),
                W::B4 | W::B8 => self.enc.mov_reg_rbp(src_w, dst, disp),
            },
        }
    }

    fn load_gpr_sx(&mut self, v: ValRef, dst: u8, dst_w: W, src_w: W) {
        match self.gpr_loc(ValueId::from(v)) {
            OpLoc::Reg(r) => match src_w {
                W::B1 | W::B2 => self.enc.movsx(dst_w, src_w, dst, Rm::Direct(r)),
                W::B4 => self.enc.movsx(W::B8, W::B4, dst, Rm::Direct(r)),
                W::B8 => self.enc.mov_reg_reg(W::B8, dst, r),
            },
            OpLoc::Mem(disp) => match src_w {
                W::B1 | W::B2 => self.enc.movsx(dst_w, src_w, dst, Rm::RbpDisp(disp)),
                W::B4 => self.enc.movsx(W::B8, W::B4, dst, Rm::RbpDisp(disp)),
                W::B8 => self.enc.mov_reg_rbp(W::B8, dst, disp),
            },
        }
    }

    fn load_xmm(&mut self, v: ValRef, dst_xmm: u8, f64_: bool) {
        match self.xmm_loc(ValueId::from(v)) {
            OpLoc::Reg(r) => {
                if r != dst_xmm {
                    self.enc.sse_move(dst_xmm, r, f64_);
                }
            }
            OpLoc::Mem(disp) => {
                if f64_ {
                    self.enc.movsd_load(dst_xmm, Rm::RbpDisp(disp));
                } else {
                    self.enc.movss_load(dst_xmm, Rm::RbpDisp(disp));
                }
            }
        }
    }

    // ---- result placement ---------------------------------------------------------------

    fn place_gpr_vid(&mut self, vid: ValueId, src: u8, w: W) {
        match self.gpr_loc(vid) {
            OpLoc::Reg(r) => {
                if r != src {
                    self.enc.mov_reg_reg(w, r, src);
                }
            }
            OpLoc::Mem(disp) => self.enc.mov_rbp_reg(w, disp, src),
        }
    }

    fn place_gpr(&mut self, id: InstId, src: u8, w: W) {
        self.place_gpr_vid(ValueId::Val(id.0), src, w);
    }

    fn place_xmm_vid(&mut self, vid: ValueId, src_xmm: u8, f64_: bool) {
        match self.xmm_loc(vid) {
            OpLoc::Reg(r) => {
                if r != src_xmm {
                    self.enc.sse_move(r, src_xmm, f64_);
                }
            }
            OpLoc::Mem(disp) => {
                if f64_ {
                    self.enc.movsd_store(Rm::RbpDisp(disp), src_xmm);
                } else {
                    self.enc.movss_store(Rm::RbpDisp(disp), src_xmm);
                }
            }
        }
    }

    fn place_xmm(&mut self, id: InstId, src_xmm: u8, f64_: bool) {
        self.place_xmm_vid(ValueId::Val(id.0), src_xmm, f64_);
    }

    /// Copies `v` (of type `ty`) straight into instruction `id`'s own result location — used
    /// by `select`'s two arms and, via `emit_phi_copies`'s staging, by phi resolution.
    fn copy_value(&mut self, v: ValRef, id: InstId, ty: Ty) {
        if is_float(ty) {
            let f64_ = is_f64(ty);
            self.load_xmm(v, 0, f64_);
            self.place_xmm(id, 0, f64_);
        } else {
            let w = width_of(ty);
            self.load_gpr(v, RAX, w);
            self.place_gpr(id, RAX, w);
        }
    }

    fn lower_inst(&mut self, id: InstId) {
        let f = self.f;
        let inst = &f.insts[id.0 as usize];
        let ty = inst.ty;
        match &inst.op {
            Op::ConstInt(n) => {
                let n = *n;
                if let Ty::Ptr(space) = ty {
                    if local_like(space) {
                        let key = (space_tag(space), n);
                        let disp = *self
                            .frame
                            .const_addr_disp
                            .get(&key)
                            .expect("Frame::build pre-scans every local-slot constant");
                        self.enc.lea_rbp(RAX, disp);
                        self.place_gpr(id, RAX, W::B8);
                        return;
                    }
                }
                self.enc.movabs(RAX, n);
                self.place_gpr(id, RAX, width_of(ty));
            }
            Op::ConstFloat(v) => {
                let v = *v;
                if is_f64(ty) {
                    self.enc.movabs(RAX, v.to_bits() as i64);
                    self.enc.movd_to_xmm(W::B8, 0, Rm::Direct(RAX));
                    self.place_xmm(id, 0, true);
                } else {
                    let bits = (v as f32).to_bits();
                    self.enc.mov_reg_imm32(W::B4, RAX, bits as i32);
                    self.enc.movd_to_xmm(W::B4, 0, Rm::Direct(RAX));
                    self.place_xmm(id, 0, false);
                }
            }
            Op::Bin(op, a, b) => {
                let (op, a, b) = (*op, *a, *b);
                self.lower_bin(id, op, a, b, ty);
            }
            Op::ICmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                self.lower_icmp(id, pred, cty, a, b);
            }
            Op::FCmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                self.lower_fcmp(id, pred, cty, a, b);
            }
            Op::Select(c, a, b) => {
                let (c, a, b) = (*c, *a, *b);
                self.load_gpr(c, RAX, W::B1);
                self.enc.test_reg_reg(W::B1, RAX);
                let else_label = self.fresh_label("select_false");
                let end_label = self.fresh_label("select_end");
                self.enc.jcc(cc::E, &else_label);
                self.copy_value(a, id, ty);
                self.enc.jmp(&end_label);
                self.enc.label(&else_label);
                self.copy_value(b, id, ty);
                self.enc.label(&end_label);
            }
            Op::Cast(cop, sty, v) => {
                let (cop, sty, v) = (*cop, *sty, *v);
                self.lower_cast(id, cop, sty, v, ty);
            }
            Op::Load { ptr, .. } => {
                let ptr = *ptr;
                self.lower_load(id, ptr, ty);
            }
            Op::Store {
                ptr, val, ty: sty, ..
            } => {
                let (ptr, val, sty) = (*ptr, *val, *sty);
                self.lower_store(ptr, val, sty);
            }
            Op::Phi(_) => {
                // Nothing to do at the definition site: every predecessor writes this phi's
                // own result location before jumping here — see `emit_phi_copies`.
            }
            Op::TidX => {
                // A bare register read: the loop counter already lives in `r11` for the
                // whole function (see the module header) — no memory reload needed.
                self.place_gpr(id, R11, width_of(ty));
            }
            Op::BdimX => {
                self.enc.mov_reg_rbp(W::B8, RAX, self.frame.nthreads_home);
                self.place_gpr(id, RAX, width_of(ty));
            }
            Op::TidY | Op::TidZ | Op::BidX | Op::BidY | Op::BidZ => {
                self.enc.movabs(RAX, 0);
                self.place_gpr(id, RAX, width_of(ty));
            }
            Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ => {
                self.enc.movabs(RAX, 1);
                self.place_gpr(id, RAX, width_of(ty));
            }
            Op::Barrier => self.enc.nop(),
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
                unreachable!("check_module refuses these before codegen starts")
            }
            Op::Atomic(aop, ptr, val, _space) => {
                let (aop, ptr, val) = (*aop, *ptr, *val);
                self.lower_atomic(id, aop, ptr, val, ty);
            }
            Op::AtomicCas(ptr, cmp, newv, _space) => {
                let (ptr, cmp, newv) = (*ptr, *cmp, *newv);
                self.lower_atomic_cas(id, ptr, cmp, newv, ty);
            }
            Op::Mma { .. } => {
                unreachable!("check_module refuses mma before codegen starts")
            }
        }
    }

    fn lower_bin(&mut self, id: InstId, op: BinOp, a: ValRef, b: ValRef, ty: Ty) {
        if is_float(ty) {
            let f64_ = is_f64(ty);
            self.load_xmm(a, 0, f64_);
            self.load_xmm(b, 1, f64_);
            match op {
                BinOp::FAdd => self.enc.sse_arith(SseArith::Add, f64_, 0, Rm::Direct(1)),
                BinOp::FSub => self.enc.sse_arith(SseArith::Sub, f64_, 0, Rm::Direct(1)),
                BinOp::FMul => self.enc.sse_arith(SseArith::Mul, f64_, 0, Rm::Direct(1)),
                BinOp::FDiv => self.enc.sse_arith(SseArith::Div, f64_, 0, Rm::Direct(1)),
                BinOp::FRem => self.lower_frem(f64_),
                _ => unreachable!("float-typed Bin with a non-float BinOp"),
            }
            self.place_xmm(id, 0, f64_);
            return;
        }

        let w = width_of(ty);
        match op {
            BinOp::Add | BinOp::Sub | BinOp::And | BinOp::Or | BinOp::Xor => {
                self.load_gpr(a, RAX, w);
                self.load_gpr(b, RCX, w);
                let aop = match op {
                    BinOp::Add => AluOp::Add,
                    BinOp::Sub => AluOp::Sub,
                    BinOp::And => AluOp::And,
                    BinOp::Or => AluOp::Or,
                    BinOp::Xor => AluOp::Xor,
                    _ => unreachable!(),
                };
                self.enc.alu_reg_reg(aop, w, RAX, RCX);
                self.place_gpr(id, RAX, w);
            }
            BinOp::Mul => {
                if w == W::B1 {
                    self.load_gpr_zx(a, RAX, W::B4, W::B1);
                    self.load_gpr_zx(b, RCX, W::B4, W::B1);
                    self.enc.imul_reg_reg(W::B4, RAX, RCX);
                } else {
                    self.load_gpr(a, RAX, w);
                    self.load_gpr(b, RCX, w);
                    self.enc.imul_reg_reg(w, RAX, RCX);
                }
                self.place_gpr(id, RAX, w);
            }
            BinOp::Div | BinOp::Rem => {
                let dw = if w == W::B1 { W::B4 } else { w };
                if w == W::B1 {
                    self.load_gpr_sx(a, RAX, W::B4, W::B1);
                    self.load_gpr_sx(b, R10, W::B4, W::B1);
                } else {
                    self.load_gpr(a, RAX, w);
                    self.load_gpr(b, R10, w);
                }
                self.enc.cdq(dw);
                self.enc.idiv_reg(dw, R10);
                let result_reg = if matches!(op, BinOp::Div) { RAX } else { RDX };
                self.place_gpr(id, result_reg, w);
            }
            BinOp::Shl | BinOp::Lshr | BinOp::Ashr => {
                self.load_gpr(a, RAX, w);
                self.load_gpr(b, RCX, w);
                let kind = match op {
                    BinOp::Shl => ShiftKind::Shl,
                    BinOp::Lshr => ShiftKind::Shr,
                    BinOp::Ashr => ShiftKind::Sar,
                    _ => unreachable!(),
                };
                self.enc.shift_cl(kind, w, RAX);
                self.place_gpr(id, RAX, w);
            }
            _ => unreachable!("integer-typed Bin with a float BinOp"),
        }
    }

    /// Software float remainder, identical technique to `oracle.rs`'s `lower_frem`: entered
    /// with `xmm0 = a`, `xmm1 = b`; leaves the result in `xmm0`.
    fn lower_frem(&mut self, f64_: bool) {
        self.enc.sse_move(2, 0, f64_);
        self.enc.sse_arith(SseArith::Div, f64_, 0, Rm::Direct(1));
        self.enc.cvtt_to_si(f64_, W::B8, RAX, Rm::Direct(0));
        self.enc.cvt_si_to(f64_, W::B8, 0, Rm::Direct(RAX));
        self.enc.sse_arith(SseArith::Mul, f64_, 0, Rm::Direct(1));
        self.enc.sse_move(1, 2, f64_);
        self.enc.sse_arith(SseArith::Sub, f64_, 1, Rm::Direct(0));
        self.enc.sse_move(0, 1, f64_);
    }

    fn lower_icmp(&mut self, id: InstId, pred: ICmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let w = width_of(cty);
        self.load_gpr(a, RAX, w);
        self.load_gpr(b, RCX, w);
        self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
        let code = match pred {
            ICmpPred::Eq => cc::E,
            ICmpPred::Ne => cc::NE,
            ICmpPred::Slt => cc::L,
            ICmpPred::Sle => cc::LE,
            ICmpPred::Sgt => cc::G,
            ICmpPred::Sge => cc::GE,
            ICmpPred::Ult => cc::B,
            ICmpPred::Ule => cc::BE,
            ICmpPred::Ugt => cc::A,
            ICmpPred::Uge => cc::AE,
        };
        self.enc.setcc(code, RAX);
        self.place_gpr(id, RAX, W::B1);
    }

    /// See `oracle.rs`'s identically-named function for the `ucomiss`/`ucomisd` flag-
    /// combination derivation this mirrors exactly.
    fn lower_fcmp(&mut self, id: InstId, pred: FCmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let f64_ = is_f64(cty);
        self.load_xmm(a, 0, f64_);
        self.load_xmm(b, 1, f64_);
        if f64_ {
            self.enc.ucomisd(0, Rm::Direct(1));
        } else {
            self.enc.ucomiss(0, Rm::Direct(1));
        }
        match pred {
            FCmpPred::Ogt => self.enc.setcc(cc::A, RAX),
            FCmpPred::Oge => self.enc.setcc(cc::AE, RAX),
            FCmpPred::Ord => self.enc.setcc(cc::NP, RAX),
            FCmpPred::Uno => self.enc.setcc(cc::P, RAX),
            FCmpPred::Olt => {
                self.enc.setcc(cc::B, RAX);
                self.enc.setcc(cc::NE, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
            FCmpPred::Ole => {
                self.enc.setcc(cc::BE, RAX);
                self.enc.setcc(cc::NP, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
            FCmpPred::Oeq => {
                self.enc.setcc(cc::E, RAX);
                self.enc.setcc(cc::NP, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
            FCmpPred::One => {
                self.enc.setcc(cc::NE, RAX);
                self.enc.setcc(cc::NP, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
        }
        self.place_gpr(id, RAX, W::B1);
    }

    fn lower_cast(&mut self, id: InstId, cop: CastOp, sty: Ty, v: ValRef, dty: Ty) {
        match cop {
            CastOp::Trunc => {
                let dw = width_of(dty);
                self.load_gpr(v, RAX, dw);
                self.place_gpr(id, RAX, dw);
            }
            CastOp::Zext => {
                let sw = width_of(sty);
                let dw = width_of(dty);
                self.load_gpr_zx(v, RAX, dw, sw);
                self.place_gpr(id, RAX, dw);
            }
            CastOp::Sext => {
                let sw = width_of(sty);
                let dw = width_of(dty);
                self.load_gpr_sx(v, RAX, dw, sw);
                self.place_gpr(id, RAX, dw);
            }
            CastOp::FpTrunc => {
                self.load_xmm(v, 0, true);
                self.enc.cvtsd2ss(0, Rm::Direct(0));
                self.place_xmm(id, 0, false);
            }
            CastOp::FpExt => {
                self.load_xmm(v, 0, false);
                self.enc.cvtss2sd(0, Rm::Direct(0));
                self.place_xmm(id, 0, true);
            }
            CastOp::FpToSi => {
                let src_f64 = is_f64(sty);
                self.load_xmm(v, 0, src_f64);
                let gpr_w = if width_of(dty) == W::B8 { W::B8 } else { W::B4 };
                self.enc.cvtt_to_si(src_f64, gpr_w, RAX, Rm::Direct(0));
                self.place_gpr(id, RAX, width_of(dty));
            }
            CastOp::FpToUi => self.lower_fp_to_ui(id, sty, v, dty),
            CastOp::SiToFp => {
                let dst_f64 = is_f64(dty);
                let sw = width_of(sty);
                if sw == W::B1 || sw == W::B2 {
                    self.load_gpr_sx(v, RAX, W::B4, sw);
                } else {
                    self.load_gpr(v, RAX, sw);
                }
                let gpr_w = if sw == W::B8 { W::B8 } else { W::B4 };
                self.enc.cvt_si_to(dst_f64, gpr_w, 0, Rm::Direct(RAX));
                self.place_xmm(id, 0, dst_f64);
            }
            CastOp::UiToFp => self.lower_ui_to_fp(id, sty, v, dty),
            CastOp::Bitcast => {
                let src_float = is_float(sty);
                let dst_float = is_float(dty);
                let w = width_of(dty);
                match (src_float, dst_float) {
                    (false, true) => {
                        self.load_gpr(v, RAX, w);
                        self.enc.movd_to_xmm(w, 0, Rm::Direct(RAX));
                        self.place_xmm(id, 0, w == W::B8);
                    }
                    (true, false) => {
                        self.load_xmm(v, 0, is_f64(sty));
                        self.enc.movd_from_xmm(w, Rm::Direct(RAX), 0);
                        self.place_gpr(id, RAX, w);
                    }
                    _ => {
                        self.load_gpr(v, RAX, w);
                        self.place_gpr(id, RAX, w);
                    }
                }
            }
        }
    }

    /// See `oracle.rs`'s identically-named function for the two-path derivation this mirrors.
    fn lower_fp_to_ui(&mut self, id: InstId, sty: Ty, v: ValRef, dty: Ty) {
        let src_f64 = is_f64(sty);
        self.load_xmm(v, 0, src_f64);
        let dw = width_of(dty);
        if dw != W::B8 {
            self.enc.cvtt_to_si(src_f64, W::B8, RAX, Rm::Direct(0));
            self.place_gpr(id, RAX, dw);
            return;
        }

        let two_pow_63_bits: i64 = if src_f64 {
            9223372036854775808.0_f64.to_bits() as i64
        } else {
            9223372036854775808.0_f32.to_bits() as i64
        };
        self.enc.movabs(RCX, two_pow_63_bits);
        self.enc
            .movd_to_xmm(if src_f64 { W::B8 } else { W::B4 }, 1, Rm::Direct(RCX));
        if src_f64 {
            self.enc.ucomisd(0, Rm::Direct(1));
        } else {
            self.enc.ucomiss(0, Rm::Direct(1));
        }
        let hi_label = self.fresh_label("fptoui_hi");
        let done_label = self.fresh_label("fptoui_done");
        self.enc.jcc(cc::AE, &hi_label);
        self.enc.cvtt_to_si(src_f64, W::B8, RAX, Rm::Direct(0));
        self.enc.jmp(&done_label);
        self.enc.label(&hi_label);
        self.enc.sse_arith(SseArith::Sub, src_f64, 0, Rm::Direct(1));
        self.enc.cvtt_to_si(src_f64, W::B8, RAX, Rm::Direct(0));
        self.enc.movabs(RCX, i64::MIN);
        self.enc.alu_reg_reg(AluOp::Xor, W::B8, RAX, RCX);
        self.enc.label(&done_label);
        self.place_gpr(id, RAX, W::B8);
    }

    /// See `oracle.rs`'s identically-named function for the two-path derivation this mirrors.
    fn lower_ui_to_fp(&mut self, id: InstId, sty: Ty, v: ValRef, dty: Ty) {
        let dst_f64 = is_f64(dty);
        let sw = width_of(sty);
        if sw != W::B8 {
            match sw {
                W::B1 | W::B2 => self.load_gpr_zx(v, RAX, W::B8, sw),
                W::B4 => self.load_gpr(v, RAX, W::B4), // zero-extends to 64 bits natively
                W::B8 => unreachable!(),
            }
            self.enc.cvt_si_to(dst_f64, W::B8, 0, Rm::Direct(RAX));
            self.place_xmm(id, 0, dst_f64);
            return;
        }

        self.load_gpr(v, RAX, W::B8);
        self.enc.test_reg_reg(W::B8, RAX);
        let hi_label = self.fresh_label("uitofp_hi");
        let done_label = self.fresh_label("uitofp_done");
        self.enc.jcc(cc::S, &hi_label);
        self.enc.cvt_si_to(dst_f64, W::B8, 0, Rm::Direct(RAX));
        self.enc.jmp(&done_label);
        self.enc.label(&hi_label);
        self.enc.mov_reg_reg(W::B8, RCX, RAX);
        self.enc.alu_reg_imm32(AluOp::And, W::B8, RCX, 1);
        self.enc.shift1(ShiftKind::Shr, W::B8, RAX);
        self.enc.alu_reg_reg(AluOp::Or, W::B8, RAX, RCX);
        self.enc.cvt_si_to(dst_f64, W::B8, 0, Rm::Direct(RAX));
        self.enc.sse_arith(SseArith::Add, dst_f64, 0, Rm::Direct(0));
        self.enc.label(&done_label);
        self.place_xmm(id, 0, dst_f64);
    }

    /// Every address space this backend touches is, by the time a value reaches here, a
    /// genuine usable stack or heap address — same as `oracle.rs`.
    fn lower_load(&mut self, id: InstId, ptr: ValRef, ty: Ty) {
        self.load_gpr(ptr, R10, W::B8);
        if is_float(ty) {
            let f64_ = is_f64(ty);
            if f64_ {
                self.enc.movsd_load(0, Rm::IndBase(R10));
            } else {
                self.enc.movss_load(0, Rm::IndBase(R10));
            }
            self.place_xmm(id, 0, f64_);
        } else {
            let w = width_of(ty);
            self.enc.mov_reg_ind(w, RAX, R10);
            self.place_gpr(id, RAX, w);
        }
    }

    fn lower_store(&mut self, ptr: ValRef, val: ValRef, ty: Ty) {
        self.load_gpr(ptr, R10, W::B8);
        if is_float(ty) {
            let f64_ = is_f64(ty);
            self.load_xmm(val, 0, f64_);
            if f64_ {
                self.enc.movsd_store(Rm::IndBase(R10), 0);
            } else {
                self.enc.movss_store(Rm::IndBase(R10), 0);
            }
        } else {
            let w = width_of(ty);
            self.load_gpr(val, RAX, w);
            self.enc.mov_ind_reg(w, R10, RAX);
        }
    }

    /// Ordinary (non-`lock`-prefixed) load-compute-store, exactly like `oracle.rs`'s own
    /// atomic lowering and for the same reason (one thread ever executes at a time). The old
    /// value is stashed in `rdx` rather than the oracle's `rbx`, since this backend's scratch
    /// set is caller-saved-only (see the module header) and `rbx` is callee-saved.
    fn lower_atomic(&mut self, id: InstId, op: AtomicOp, ptr: ValRef, val: ValRef, ty: Ty) {
        self.load_gpr(ptr, R10, W::B8);

        if is_float(ty) {
            let f64_ = is_f64(ty);
            if f64_ {
                self.enc.movsd_load(0, Rm::IndBase(R10));
            } else {
                self.enc.movss_load(0, Rm::IndBase(R10));
            }
            self.enc.sse_move(2, 0, f64_); // xmm2 = old, kept for the return value
            self.load_xmm(val, 1, f64_);
            match op {
                AtomicOp::Add => self.enc.sse_arith(SseArith::Add, f64_, 0, Rm::Direct(1)),
                AtomicOp::Sub => self.enc.sse_arith(SseArith::Sub, f64_, 0, Rm::Direct(1)),
                AtomicOp::Exch => self.enc.sse_move(0, 1, f64_),
                AtomicOp::Min | AtomicOp::Max => {
                    if f64_ {
                        self.enc.ucomisd(0, Rm::Direct(1));
                    } else {
                        self.enc.ucomiss(0, Rm::Direct(1));
                    }
                    let skip = self.fresh_label("atomic_minmax_skip");
                    let skip_cc = if matches!(op, AtomicOp::Min) {
                        cc::BE
                    } else {
                        cc::AE
                    };
                    self.enc.jcc(skip_cc, &skip);
                    self.enc.sse_move(0, 1, f64_);
                    self.enc.label(&skip);
                }
                AtomicOp::And | AtomicOp::Or | AtomicOp::Xor => {
                    let w = if f64_ { W::B8 } else { W::B4 };
                    self.enc.movd_from_xmm(w, Rm::Direct(RAX), 0);
                    self.enc.movd_from_xmm(w, Rm::Direct(RCX), 1);
                    let aop = match op {
                        AtomicOp::And => AluOp::And,
                        AtomicOp::Or => AluOp::Or,
                        AtomicOp::Xor => AluOp::Xor,
                        _ => unreachable!(),
                    };
                    self.enc.alu_reg_reg(aop, w, RAX, RCX);
                    self.enc.movd_to_xmm(w, 0, Rm::Direct(RAX));
                }
            }
            if f64_ {
                self.enc.movsd_store(Rm::IndBase(R10), 0);
            } else {
                self.enc.movss_store(Rm::IndBase(R10), 0);
            }
            self.place_xmm(id, 2, f64_);
            return;
        }

        let w = width_of(ty);
        self.enc.mov_reg_ind(w, RAX, R10); // old
        self.enc.mov_reg_reg(w, RDX, RAX); // stashed for the return value
        self.load_gpr(val, RCX, w);
        match op {
            AtomicOp::Add => self.enc.alu_reg_reg(AluOp::Add, w, RAX, RCX),
            AtomicOp::Sub => self.enc.alu_reg_reg(AluOp::Sub, w, RAX, RCX),
            AtomicOp::And => self.enc.alu_reg_reg(AluOp::And, w, RAX, RCX),
            AtomicOp::Or => self.enc.alu_reg_reg(AluOp::Or, w, RAX, RCX),
            AtomicOp::Xor => self.enc.alu_reg_reg(AluOp::Xor, w, RAX, RCX),
            AtomicOp::Exch => self.enc.mov_reg_reg(w, RAX, RCX),
            AtomicOp::Min | AtomicOp::Max => {
                self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
                let skip = self.fresh_label("atomic_minmax_skip");
                let skip_cc = if matches!(op, AtomicOp::Min) {
                    cc::LE
                } else {
                    cc::GE
                };
                self.enc.jcc(skip_cc, &skip);
                self.enc.mov_reg_reg(w, RAX, RCX);
                self.enc.label(&skip);
            }
        }
        self.enc.mov_ind_reg(w, R10, RAX); // store new
        self.place_gpr(id, RDX, w); // return old
    }

    /// `atomicCAS`: compares and swaps the raw bit pattern regardless of `ty` — see
    /// `oracle.rs`'s identically-named function.
    fn lower_atomic_cas(&mut self, id: InstId, ptr: ValRef, cmp: ValRef, newv: ValRef, ty: Ty) {
        let w = width_of(ty);
        self.load_gpr(ptr, R10, W::B8);
        self.enc.mov_reg_ind(w, RAX, R10);
        self.load_gpr(cmp, RCX, w);
        self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
        let mismatch = self.fresh_label("cas_mismatch");
        self.enc.jcc(cc::NE, &mismatch);
        self.load_gpr(newv, RDX, w);
        self.enc.mov_ind_reg(w, R10, RDX);
        self.enc.label(&mismatch);
        self.place_gpr(id, RAX, w);
    }

    /// Resolves every phi in `to_block` that needs a copy from the `from_block -> to_block`
    /// edge. Two phases to correctly handle multiple simultaneous phis in the same target
    /// block — see the module header on why a naive single-pass copy is unsound here (it
    /// wasn't for the oracle, where every phi destination is a distinct memory slot that can
    /// never collide with another phi's source location; it can be here, since two different
    /// phis' locations can coincide with each other's source registers).
    fn emit_phi_copies(&mut self, from: u32, to: u32) {
        let Some(copies) = self.phi_copies.get(&(from, to)).cloned() else {
            return;
        };

        // Phase 1: capture every incoming value into its own phi's staging slot before any
        // phi's real location is touched.
        for &(phi_id, val, ty) in &copies {
            let stage_disp = self.frame.phi_stage[&phi_id.0];
            if is_float(ty) {
                let f64_ = is_f64(ty);
                self.load_xmm(val, 0, f64_);
                if f64_ {
                    self.enc.movsd_store(Rm::RbpDisp(stage_disp), 0);
                } else {
                    self.enc.movss_store(Rm::RbpDisp(stage_disp), 0);
                }
            } else {
                let w = width_of(ty);
                self.load_gpr(val, RAX, w);
                self.enc.mov_rbp_reg(w, stage_disp, RAX);
            }
        }

        // Phase 2: move every captured value from staging into its phi's real location.
        for &(phi_id, _val, ty) in &copies {
            let stage_disp = self.frame.phi_stage[&phi_id.0];
            if is_float(ty) {
                let f64_ = is_f64(ty);
                if f64_ {
                    self.enc.movsd_load(0, Rm::RbpDisp(stage_disp));
                } else {
                    self.enc.movss_load(0, Rm::RbpDisp(stage_disp));
                }
                self.place_xmm(phi_id, 0, f64_);
            } else {
                let w = width_of(ty);
                self.enc.mov_reg_rbp(w, RAX, stage_disp);
                self.place_gpr(phi_id, RAX, w);
            }
        }
    }

    fn lower_term(&mut self, from_block: u32, term: &Term) {
        match term {
            Term::Br(target) => {
                self.emit_phi_copies(from_block, target.0);
                self.enc.jmp(&block_label(target.0));
            }
            Term::CondBr(cond, t, f) => {
                self.load_gpr(*cond, RAX, W::B1);
                self.enc.test_reg_reg(W::B1, RAX);
                let false_prep = self.fresh_label("condbr_false");
                self.enc.jcc(cc::E, &false_prep);
                self.emit_phi_copies(from_block, t.0);
                self.enc.jmp(&block_label(t.0));
                self.enc.label(&false_prep);
                self.emit_phi_copies(from_block, f.0);
                self.enc.jmp(&block_label(f.0));
            }
            Term::Switch(scrut, default, cases) => {
                let ty = self.valref_ty(*scrut);
                let w = width_of(ty);
                self.load_gpr(*scrut, RAX, w);
                for &(case_val, target) in cases {
                    self.enc.movabs(RCX, case_val);
                    self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
                    let skip = self.fresh_label("switch_skip");
                    self.enc.jcc(cc::NE, &skip);
                    self.emit_phi_copies(from_block, target.0);
                    self.enc.jmp(&block_label(target.0));
                    self.enc.label(&skip);
                }
                self.emit_phi_copies(from_block, default.0);
                self.enc.jmp(&block_label(default.0));
            }
            Term::Ret(v) => {
                if let Some(val) = v {
                    let rty = self.f.ret;
                    let disp = self
                        .frame
                        .retval_home
                        .expect("non-void Ret always has a retval home");
                    if is_float(rty) {
                        let f64_ = is_f64(rty);
                        self.load_xmm(*val, 0, f64_);
                        if f64_ {
                            self.enc.movsd_store(Rm::RbpDisp(disp), 0);
                        } else {
                            self.enc.movss_store(Rm::RbpDisp(disp), 0);
                        }
                    } else {
                        let w = width_of(rty);
                        self.load_gpr(*val, RAX, w);
                        self.enc.mov_rbp_reg(w, disp, RAX);
                    }
                }
                self.enc.jmp("__loop_incr");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{Block, BlockId, Inst, LaunchBounds, MmaLayout};
    use object::read::{Object as ReadObject, ObjectSection, ObjectSymbol};

    fn wrap(f: Function) -> Module {
        Module {
            funcs: vec![f],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    fn parses_as_elf_with_symbol<'a>(bytes: &'a [u8], symbol: &str) -> object::read::File<'a> {
        let file = object::read::File::parse(bytes).expect("parses as an object file");
        assert_eq!(file.format(), object::BinaryFormat::Elf);
        assert_eq!(file.architecture(), object::Architecture::X86_64);
        let text = file
            .section_by_name(".text")
            .expect(".text section present");
        assert!(!text.data().unwrap().is_empty(), ".text must not be empty");
        let sym = file
            .symbols()
            .find(|s| s.name() == Ok(symbol))
            .unwrap_or_else(|| panic!("symbol `{symbol}` present"));
        assert_eq!(sym.size(), text.data().unwrap().len() as u64);
        file
    }

    // ---- fixtures, mirroring oracle.rs's own (same shapes, same doc comments removed to
    // avoid duplicating that module's ASCII sketches of oracle-specific memory layout, which
    // doesn't apply here) --------------------------------------------------------------------

    fn func_ret_const() -> Function {
        Function {
            name: "ret_const".into(),
            params: vec![],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::ConstInt(42),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    fn func_add_i32() -> Function {
        Function {
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
        }
    }

    fn func_max_i32() -> Function {
        Function {
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
        }
    }

    fn func_write_idx() -> Function {
        Function {
            name: "write_idx".into(),
            params: vec![Ty::Ptr(AddrSpace::Global)],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::TidX,
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I64),
                    op: Op::Cast(
                        CastOp::Zext,
                        Ty::Scalar(Scalar::I32),
                        ValRef::Val(InstId(0)),
                    ),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I64),
                    op: Op::ConstInt(4),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I64),
                    op: Op::Bin(BinOp::Mul, ValRef::Val(InstId(1)), ValRef::Val(InstId(2))),
                },
                Inst {
                    ty: Ty::Ptr(AddrSpace::Global),
                    op: Op::Cast(
                        CastOp::Bitcast,
                        Ty::Scalar(Scalar::I64),
                        ValRef::Val(InstId(3)),
                    ),
                },
                Inst {
                    ty: Ty::Ptr(AddrSpace::Global),
                    op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Val(InstId(4))),
                },
                Inst {
                    ty: Ty::Void,
                    op: Op::Store {
                        ptr: ValRef::Val(InstId(5)),
                        val: ValRef::Val(InstId(0)),
                        ty: Ty::Scalar(Scalar::I32),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                },
            ],
            blocks: vec![Block {
                insts: (0..7).map(InstId).collect(),
                term: Term::Ret(None),
            }],
        }
    }

    // ---- supports() --------------------------------------------------------------------

    #[test]
    fn supports_a_module_using_only_implemented_ops() {
        assert_eq!(
            X86Regalloc.supports(&wrap(func_ret_const())),
            Support::Supported
        );
        assert_eq!(
            X86Regalloc.supports(&wrap(func_add_i32())),
            Support::Supported
        );
        assert_eq!(
            X86Regalloc.supports(&wrap(func_max_i32())),
            Support::Supported
        );
        assert_eq!(
            X86Regalloc.supports(&wrap(func_write_idx())),
            Support::Supported
        );
    }

    #[test]
    fn refuses_shuffle_with_e090() {
        let f = Function {
            name: "shuf".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Shuffle(
                    basalt_bir::ShuffleKind::Idx,
                    ValRef::Param(0),
                    ValRef::Param(0),
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Regalloc.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    #[test]
    fn refuses_mma_with_e090() {
        let ptr_global = Ty::Ptr(AddrSpace::Global);
        let f = Function {
            name: "usesmma".into(),
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
        assert_eq!(
            X86Regalloc.supports(&wrap(f.clone())),
            Support::Unsupported(ECode::UnsupportedOp)
        );
        let err = X86Regalloc
            .emit(&wrap(f), &EmitOpts::default())
            .expect_err("emit must refuse what supports() refuses, not guess");
        assert_eq!(err.code, ECode::UnsupportedOp);
    }

    #[test]
    fn refuses_ballot_vote_with_e090() {
        for op in [
            Op::Ballot(ValRef::Param(0)),
            Op::VoteAny(ValRef::Param(0)),
            Op::VoteAll(ValRef::Param(0)),
        ] {
            let f = Function {
                name: "vote".into(),
                params: vec![Ty::Scalar(Scalar::I1)],
                ret: Ty::Void,
                insts: vec![Inst {
                    ty: Ty::Scalar(Scalar::I1),
                    op,
                }],
                blocks: vec![Block {
                    insts: vec![InstId(0)],
                    term: Term::Ret(None),
                }],
            };
            assert_eq!(
                X86Regalloc.supports(&wrap(f)),
                Support::Unsupported(ECode::UnsupportedOp)
            );
        }
    }

    #[test]
    fn refuses_vector_result_with_e091() {
        let f = Function {
            name: "vecty".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Vec(Scalar::F32, 4),
                op: Op::ConstFloat(1.0),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Regalloc.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_vector_return_with_e091() {
        let f = Function {
            name: "vecret".into(),
            params: vec![],
            ret: Ty::Vec(Scalar::I32, 4),
            insts: vec![],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Regalloc.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_multi_function_module_with_e093() {
        let module = Module {
            funcs: vec![func_ret_const(), func_add_i32()],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        };
        assert_eq!(
            X86Regalloc.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_too_many_integer_params_with_e093() {
        let f = Function {
            name: "toomany".into(),
            params: vec![Ty::Scalar(Scalar::I32); 6],
            ret: Ty::Void,
            insts: vec![],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Regalloc.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_f16_arithmetic_with_e091() {
        let f = Function {
            name: "halfty".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::F16),
                op: Op::ConstFloat(1.0),
            }],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Regalloc.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    // ---- emit() -------------------------------------------------------------------------

    #[test]
    fn emits_valid_elf_for_ret_const() {
        let artifact = X86Regalloc
            .emit(&wrap(func_ret_const()), &EmitOpts::default())
            .expect("emit succeeds");
        assert_eq!(artifact.kind, ArtifactKind::Object);
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "ret_const");
    }

    #[test]
    fn emits_valid_elf_for_add_i32() {
        let artifact = X86Regalloc
            .emit(&wrap(func_add_i32()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "add_i32");
    }

    #[test]
    fn emits_valid_elf_for_condbr_with_phi() {
        let artifact = X86Regalloc
            .emit(&wrap(func_max_i32()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "max_i32");
    }

    #[test]
    fn emits_valid_elf_for_thread_index_loop() {
        let artifact = X86Regalloc
            .emit(&wrap(func_write_idx()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "write_idx");
    }

    #[test]
    fn emit_refuses_what_supports_refuses() {
        let f = Function {
            name: "shuf".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Shuffle(
                    basalt_bir::ShuffleKind::Idx,
                    ValRef::Param(0),
                    ValRef::Param(0),
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        let err = X86Regalloc
            .emit(&wrap(f), &EmitOpts::default())
            .expect_err("must refuse, not guess");
        assert_eq!(err.code, ECode::UnsupportedOp);
    }

    #[test]
    fn emit_is_deterministic() {
        let module = wrap(func_write_idx());
        let a = X86Regalloc.emit(&module, &EmitOpts::default()).unwrap();
        let b = X86Regalloc.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(
            a, b,
            "same module in must yield byte-identical artifact out"
        );
    }

    #[test]
    fn emit_is_deterministic_with_spills_and_phis() {
        // Exercises the spill and phi-copy paths together, not just the straight-line case.
        let module = wrap(func_max_i32());
        let a = X86Regalloc.emit(&module, &EmitOpts::default()).unwrap();
        let b = X86Regalloc.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(X86Regalloc.name(), "x86-regalloc");
    }

    /// Same op/type/feature refusal surface as `X86Oracle` — the two backends must agree on
    /// what a kernel suite can and cannot run through, since a later differential harness diffs
    /// them against each other on the same kernels.
    #[test]
    fn agrees_with_oracle_on_supported_and_refused_modules() {
        use crate::X86Oracle;

        let supported = [
            wrap(func_ret_const()),
            wrap(func_add_i32()),
            wrap(func_max_i32()),
            wrap(func_write_idx()),
        ];
        for module in &supported {
            assert_eq!(X86Oracle.supports(module), X86Regalloc.supports(module));
            assert_eq!(X86Oracle.supports(module), Support::Supported);
        }

        let f16 = Function {
            name: "halfty".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::F16),
                op: Op::ConstFloat(1.0),
            }],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        let module = wrap(f16);
        assert_eq!(X86Oracle.supports(&module), X86Regalloc.supports(&module));
    }

    /// Sanity check that the fixtures above are not vacuously trivial: a module can also carry
    /// the metadata BIR allows (launch bounds), which this backend simply ignores.
    #[test]
    fn ignores_launch_bounds_metadata() {
        let mut module = wrap(func_ret_const());
        module.launch_bounds = Some(LaunchBounds {
            max_threads: 128,
            min_blocks: 2,
        });
        assert_eq!(X86Regalloc.supports(&module), Support::Supported);
    }
}
