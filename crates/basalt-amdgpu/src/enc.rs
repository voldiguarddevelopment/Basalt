// Hand-rolled RDNA3 (gfx1100, wave32) instruction encoder. Same identity as
// `basalt-x86/src/enc.rs`: every function here builds bytes directly from the ISA's own
// bit-field layout, and the layouts were pinned down empirically (assembled with LLVM's own
// MC layer for this exact target, byte-for-byte, one instruction at a time) rather than
// recalled from memory — every field boundary and opcode number in this file has a matching
// hard-coded test at the bottom that reproduces the same derivation. This is deliberately not
// a general assembler: only the instruction shapes a BIR-to-AMDGCN lowering pass would
// plausibly need are implemented.
//
// # Wave size
//
// gfx1100 defaults to wave32. `vcc`/`exec` are therefore single 32-bit scalar registers
// (`vcc_lo`/`exec_lo`), not register pairs — there is no wave64 support anywhere in this file.
//
// # Register numbering
//
// SGPRs are numbered 0-105 directly. VGPRs are numbered 0-255 directly. A handful of scalar
// registers have fixed numeric codes shared by every scalar-operand field (SOP*/SMEM's
// SDATA/SBASE/SOFFSET/VOP3's SDST): `VCC_LO`=106, `VCC_HI`=107, `NULL`=124 (the "no register"
// sentinel — what a real assembler emits when, e.g., an SMEM instruction has no SGPR offset
// operand at all), `M0`=125, `EXEC_LO`=126, `EXEC_HI`=127. A 64-bit register *pair* (`s[4:5]`,
// `v[0:1]`) is always addressed by its even base register number; nothing here re-derives an
// pair's second half, callers just know their own register allocation.
//
// # Operand encoding (the part every format shares)
//
// Every scalar-ALU source operand (`SOP2`/`SOP1`/`SOPC`'s 8-bit SSRC fields) and every
// vector-ALU source operand (`VOP1`/`VOP2`/`VOP3`/`VOPC`'s 9-bit SRC fields) share one
// encoding scheme:
//   - 0-127: a scalar register (SGPR 0-105, or one of the special codes above).
//   - 128-192: inline integer constant 0-64 (value = code - 128). Vector-only: 256-511 is a
//     VGPR (value = code - 256) — the two ranges are disjoint by construction, which is
//     exactly why the vector field is 9 bits (0-511) while the scalar field is 8 (0-255).
//   - 193-208: inline integer constant -1 to -16 (value = -(code - 192)).
//   - 240-247: inline float constant {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0}.
//   - 255: literal constant — the operand is not encodable inline, so a raw 32-bit value
//     (int bit pattern, float bit pattern, or an arbitrary bit pattern for a `bitcast`
//     constant) follows immediately after the instruction's own words. Hardware allows at
//     most one literal per instruction; `Src`/`VSrc` encoding never produces two, and every
//     multi-operand builder below asserts that at most one of its operands actually needed
//     one.
// `Imm`/`Src`/`VSrc` below model exactly this; `encode_src`/`encode_vsrc` are the one place
// the 128-64-16-8-constant table lives.
//
// # Branch offsets
//
// `SOPP`'s branch instructions (`s_branch`/`s_cbranch_*`) carry a 16-bit signed *word* offset
// relative to the address of the next instruction (this instruction's own address + 4),
// exactly like x86's rel32 jumps but word- rather than byte-granular. Resolving a real branch
// target into that offset is a lowering-pass concern (basic block layout isn't known here);
// these functions take the already-computed word offset directly.

/// `vcc_lo` — the wave32 vector-compare-result / carry-out register.
pub const VCC_LO: u8 = 106;
/// `vcc_hi` — unused in wave32 (no compare ever writes here) but a valid scalar operand code.
pub const VCC_HI: u8 = 107;
/// The "no register" sentinel: what a real assembler emits for an optional scalar-register
/// operand (e.g. SMEM's SOFFSET) that the source simply didn't specify.
pub const NULL: u8 = 124;
/// `m0` — LDS/GDS size limit register (also repurposed by some ops as a general 32-bit scalar
/// scratch operand).
pub const M0: u8 = 125;
pub const EXEC_LO: u8 = 126;
pub const EXEC_HI: u8 = 127;

/// An immediate value carried by a scalar or vector source operand. `Int`/`F32` are encoded
/// inline when they match one of the ISA's fixed constant codes (see the module header) and
/// fall back to a trailing literal otherwise; `Raw` always forces a trailing literal (used for
/// `bitcast` constants and other bit patterns with no arithmetic meaning as int or float).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Imm {
    Int(i32),
    F32(f32),
    Raw(u32),
}

fn imm_inline_code(imm: Imm) -> Option<u8> {
    match imm {
        Imm::Int(v) => {
            if (0..=64).contains(&v) {
                Some(128 + v as u8)
            } else if (-16..=-1).contains(&v) {
                Some((192 + (-v)) as u8)
            } else {
                None
            }
        }
        Imm::F32(v) => {
            let bits = v.to_bits();
            [
                (0.5f32, 240u8),
                (-0.5f32, 241),
                (1.0f32, 242),
                (-1.0f32, 243),
                (2.0f32, 244),
                (-2.0f32, 245),
                (4.0f32, 246),
                (-4.0f32, 247),
            ]
            .iter()
            .find(|(c, _)| c.to_bits() == bits)
            .map(|(_, code)| *code)
        }
        Imm::Raw(_) => None,
    }
}

fn imm_literal_bits(imm: Imm) -> u32 {
    match imm {
        Imm::Int(v) => v as u32,
        Imm::F32(v) => v.to_bits(),
        Imm::Raw(b) => b,
    }
}

/// A scalar-ALU source operand (`SOP2`/`SOP1`/`SOPC`'s 8-bit SSRC fields): a scalar register
/// or an immediate. No VGPR variant exists because the scalar ALU cannot read one.
#[derive(Clone, Copy)]
pub enum Src {
    Sgpr(u8),
    Imm(Imm),
}

fn encode_src(src: Src) -> (u8, Option<u32>) {
    match src {
        Src::Sgpr(r) => {
            debug_assert!(r <= 127, "scalar operand field is 7 bits wide");
            (r, None)
        }
        Src::Imm(imm) => match imm_inline_code(imm) {
            Some(code) => (code, None),
            None => (0xFF, Some(imm_literal_bits(imm))),
        },
    }
}

/// A vector-ALU source operand (`VOP1`/`VOP2`/`VOP3`/`VOPC`'s 9-bit SRC fields): a scalar
/// register, a VGPR, or an immediate.
#[derive(Clone, Copy)]
pub enum VSrc {
    Sgpr(u8),
    Vgpr(u8),
    Imm(Imm),
}

fn encode_vsrc(src: VSrc) -> (u16, Option<u32>) {
    match src {
        VSrc::Sgpr(r) => {
            debug_assert!(r <= 127, "scalar operand field is 7 bits wide");
            (r as u16, None)
        }
        VSrc::Vgpr(r) => (256 + r as u16, None),
        VSrc::Imm(imm) => match imm_inline_code(imm) {
            Some(code) => (code as u16, None),
            None => (0xFF, Some(imm_literal_bits(imm))),
        },
    }
}

fn push_lit(code: &mut Vec<u8>, lit: Option<u32>) {
    if let Some(l) = lit {
        code.extend_from_slice(&l.to_le_bytes());
    }
}

fn at_most_one_literal(lits: &[Option<u32>]) {
    debug_assert!(
        lits.iter().filter(|l| l.is_some()).count() <= 1,
        "hardware allows at most one literal-constant operand per instruction"
    );
}

// ---- SOP2: scalar ALU, 2 operands, writes SCC -------------------------------------------
//
// `[31:30]=0b10` (fixed) `[29:23]=OP(7)` `[22:16]=SDST(7)` `[15:8]=SSRC1(8)` `[7:0]=SSRC0(8)`.
// Opcodes cover the integer arithmetic/bitwise/shift/min-max ops a scalar bookkeeping path
// (loop counters, address arithmetic derived from kernarg values) needs, plus a scalar select
// (`s_cselect_b32`, `dst := scc ? src0 : src1`) for uniform control-flow-free choices.

#[derive(Clone, Copy)]
pub enum Sop2Op {
    AddU32,
    SubU32,
    LshlB32,
    LshrB32,
    AshrI32,
    AndB32,
    OrB32,
    XorB32,
    MinI32,
    MinU32,
    MaxI32,
    MaxU32,
    MulI32,
    CselectB32,
}

impl Sop2Op {
    fn opcode(self) -> u32 {
        match self {
            Sop2Op::AddU32 => 0,
            Sop2Op::SubU32 => 1,
            Sop2Op::LshlB32 => 8,
            Sop2Op::LshrB32 => 10,
            Sop2Op::AshrI32 => 12,
            Sop2Op::MinI32 => 18,
            Sop2Op::MinU32 => 19,
            Sop2Op::MaxI32 => 20,
            Sop2Op::MaxU32 => 21,
            Sop2Op::AndB32 => 22,
            Sop2Op::OrB32 => 24,
            Sop2Op::XorB32 => 26,
            Sop2Op::MulI32 => 44,
            Sop2Op::CselectB32 => 48,
        }
    }
}

pub fn sop2(op: Sop2Op, sdst: u8, ssrc0: Src, ssrc1: Src) -> Vec<u8> {
    debug_assert!(sdst <= 127);
    let (s0, lit0) = encode_src(ssrc0);
    let (s1, lit1) = encode_src(ssrc1);
    at_most_one_literal(&[lit0, lit1]);
    let word = (0b10u32 << 30)
        | ((op.opcode() & 0x7F) << 23)
        | ((sdst as u32) << 16)
        | ((s1 as u32) << 8)
        | s0 as u32;
    let mut code = word.to_le_bytes().to_vec();
    push_lit(&mut code, lit0.or(lit1));
    code
}

// ---- SOP1: scalar ALU, 1 operand ---------------------------------------------------------
//
// `[31:23]=0b1_0111_1101` (fixed, 9 bits) `[22:16]=SDST(7)` `[15:8]=OP(8)` `[7:0]=SSRC0(8)`.

const SOP1_FIXED: u32 = 0b1_0111_1101;

#[derive(Clone, Copy)]
pub enum Sop1Op {
    MovB32,
    MovB64,
    NotB32,
    AbsI32,
    Bcnt0I32B32,
    Bcnt1I32B32,
    CtzI32B32,
    ClzI32U32,
    SextI32I8,
    SextI32I16,
}

impl Sop1Op {
    fn opcode(self) -> u32 {
        match self {
            Sop1Op::MovB32 => 0,
            Sop1Op::MovB64 => 1,
            Sop1Op::CtzI32B32 => 8,
            Sop1Op::ClzI32U32 => 10,
            Sop1Op::SextI32I8 => 14,
            Sop1Op::SextI32I16 => 15,
            Sop1Op::AbsI32 => 21,
            Sop1Op::Bcnt0I32B32 => 22,
            Sop1Op::Bcnt1I32B32 => 24,
            Sop1Op::NotB32 => 30,
        }
    }
}

pub fn sop1(op: Sop1Op, sdst: u8, ssrc0: Src) -> Vec<u8> {
    debug_assert!(sdst <= 127);
    let (s0, lit0) = encode_src(ssrc0);
    let word = (SOP1_FIXED << 23) | ((sdst as u32) << 16) | ((op.opcode() & 0xFF) << 8) | s0 as u32;
    let mut code = word.to_le_bytes().to_vec();
    push_lit(&mut code, lit0);
    code
}

// ---- SOPK: scalar ALU with a 16-bit immediate -------------------------------------------
//
// `[31:28]=0b1011` (fixed) `[27:23]=OP(5)` `[22:16]=SDST(7)` `[15:0]=SIMM16`. Included for
// `s_movk_i32` (a cheap way to materialize a 16-bit-or-less scalar constant without spending
// a trailing literal dword) and `s_addk_i32` (in-place accumulate, useful for a loop-bound
// bump); the rest of SOPK (shift/mul-by-16-bit-immediate, etc.) is not implemented since
// nothing in the covered op set needs it yet.

#[derive(Clone, Copy)]
pub enum SopkOp {
    MovkI32,
    AddkI32,
}

impl SopkOp {
    fn opcode(self) -> u32 {
        match self {
            SopkOp::MovkI32 => 0,
            SopkOp::AddkI32 => 15,
        }
    }
}

pub fn sopk(op: SopkOp, sdst: u8, simm16: i16) -> Vec<u8> {
    debug_assert!(sdst <= 127);
    let word = (0b1011u32 << 28)
        | ((op.opcode() & 0x1F) << 23)
        | ((sdst as u32) << 16)
        | (simm16 as u16 as u32);
    word.to_le_bytes().to_vec()
}

// ---- SOPC: scalar compare, writes SCC ----------------------------------------------------
//
// `[31:23]=0b1_0111_1110` (fixed, 9 bits) `[22:16]=OP(7)` `[15:8]=SSRC1(8)` `[7:0]=SSRC0(8)`.

const SOPC_FIXED: u32 = 0b1_0111_1110;

#[derive(Clone, Copy)]
pub enum SopcOp {
    EqI32,
    LgI32,
    GtI32,
    GeI32,
    LtI32,
    LeI32,
    EqU32,
    LgU32,
    GtU32,
    GeU32,
    LtU32,
    LeU32,
    Bitcmp0B32,
    Bitcmp1B32,
}

impl SopcOp {
    fn opcode(self) -> u32 {
        match self {
            SopcOp::EqI32 => 0,
            SopcOp::LgI32 => 1,
            SopcOp::GtI32 => 2,
            SopcOp::GeI32 => 3,
            SopcOp::LtI32 => 4,
            SopcOp::LeI32 => 5,
            SopcOp::EqU32 => 6,
            SopcOp::LgU32 => 7,
            SopcOp::GtU32 => 8,
            SopcOp::GeU32 => 9,
            SopcOp::LtU32 => 10,
            SopcOp::LeU32 => 11,
            SopcOp::Bitcmp0B32 => 12,
            SopcOp::Bitcmp1B32 => 13,
        }
    }
}

pub fn sopc(op: SopcOp, ssrc0: Src, ssrc1: Src) -> Vec<u8> {
    let (s0, lit0) = encode_src(ssrc0);
    let (s1, lit1) = encode_src(ssrc1);
    at_most_one_literal(&[lit0, lit1]);
    let word = (SOPC_FIXED << 23) | ((op.opcode() & 0x7F) << 16) | ((s1 as u32) << 8) | s0 as u32;
    let mut code = word.to_le_bytes().to_vec();
    push_lit(&mut code, lit0.or(lit1));
    code
}

// ---- SOPP: scalar program control --------------------------------------------------------
//
// `[31:23]=0b1_0111_1111` (fixed, 9 bits) `[22:16]=OP(7)` `[15:0]=SIMM16`. Covers the
// program-termination, synchronization, and branch primitives every kernel needs.

const SOPP_FIXED: u32 = 0b1_0111_1111;

fn sopp(op: u32, simm16: u16) -> Vec<u8> {
    let word = (SOPP_FIXED << 23) | ((op & 0x7F) << 16) | simm16 as u32;
    word.to_le_bytes().to_vec()
}

pub fn s_endpgm() -> Vec<u8> {
    sopp(48, 0)
}

/// `wait_states` is the number of extra cycles to idle beyond the one this instruction itself
/// takes (`s_nop 0` is the bare one-cycle nop).
pub fn s_nop(wait_states: u8) -> Vec<u8> {
    sopp(0, wait_states as u16)
}

pub fn s_barrier() -> Vec<u8> {
    sopp(61, 0)
}

/// `s_waitcnt`. Each counter saturates hardware-side at its field width; out-of-range counts
/// are a codegen bug, not a user error, hence the assert rather than a clamp.
pub fn s_waitcnt(vmcnt: u8, expcnt: u8, lgkmcnt: u8) -> Vec<u8> {
    debug_assert!(vmcnt <= 0x3F && expcnt <= 0x7 && lgkmcnt <= 0x3F);
    let simm16 =
        ((vmcnt as u16 & 0x3F) << 10) | ((lgkmcnt as u16 & 0x3F) << 4) | (expcnt as u16 & 0x7);
    sopp(9, simm16)
}

/// Unconditional branch. `offset_words` is the already-resolved word-granular PC-relative
/// offset (see module header) — block layout is a lowering-pass concern, not this crate's.
pub fn s_branch(offset_words: i16) -> Vec<u8> {
    sopp(32, offset_words as u16)
}

#[derive(Clone, Copy)]
pub enum BrCc {
    Scc0,
    Scc1,
    Vccz,
    Vccnz,
    Execz,
    Execnz,
}

impl BrCc {
    fn opcode(self) -> u32 {
        match self {
            BrCc::Scc0 => 33,
            BrCc::Scc1 => 34,
            BrCc::Vccz => 35,
            BrCc::Vccnz => 36,
            BrCc::Execz => 37,
            BrCc::Execnz => 38,
        }
    }
}

pub fn s_cbranch(cc: BrCc, offset_words: i16) -> Vec<u8> {
    sopp(cc.opcode(), offset_words as u16)
}

// ---- VOP2: vector ALU, 2 operands ---------------------------------------------------------
//
// `[31]=0` (fixed) `[30:25]=OP(6)` `[24:17]=VDST(8)` `[16:9]=VSRC1(8, always a VGPR)`
// `[8:0]=SRC0(9)`. `CndmaskB32` (vector select) reads/writes `vcc_lo` implicitly — it is not
// an encoded operand, matching hardware — so it fits this same 2-operand shape. `AddCoCiU32`
// (carry-in *and* carry-out, both implicit through `vcc_lo`, just like `CndmaskB32`'s implicit
// `vcc_lo` read) is this same shape too: `dst := src0 + vsrc1 + vcc_lo`, `vcc_lo := carry-out`.
// It is the second half of a 64-bit add (see `vop3_carry`'s own header for the pairing).

#[derive(Clone, Copy)]
pub enum Vop2Op {
    CndmaskB32,
    AddF32,
    SubF32,
    SubrevF32,
    MulF32,
    MinF32,
    MaxF32,
    MinI32,
    MaxI32,
    MinU32,
    MaxU32,
    LshlrevB32,
    LshrrevB32,
    AshrrevI32,
    AndB32,
    OrB32,
    XorB32,
    AddNcU32,
    SubNcU32,
    AddCoCiU32,
}

impl Vop2Op {
    fn opcode(self) -> u32 {
        match self {
            Vop2Op::CndmaskB32 => 1,
            Vop2Op::AddF32 => 3,
            Vop2Op::SubF32 => 4,
            Vop2Op::SubrevF32 => 5,
            Vop2Op::MulF32 => 8,
            Vop2Op::MinF32 => 15,
            Vop2Op::MaxF32 => 16,
            Vop2Op::MinI32 => 17,
            Vop2Op::MaxI32 => 18,
            Vop2Op::MinU32 => 19,
            Vop2Op::MaxU32 => 20,
            Vop2Op::LshlrevB32 => 24,
            Vop2Op::LshrrevB32 => 25,
            Vop2Op::AshrrevI32 => 26,
            Vop2Op::AndB32 => 27,
            Vop2Op::OrB32 => 28,
            Vop2Op::XorB32 => 29,
            Vop2Op::AddNcU32 => 37,
            Vop2Op::SubNcU32 => 38,
            Vop2Op::AddCoCiU32 => 32,
        }
    }
}

pub fn vop2(op: Vop2Op, vdst: u8, src0: VSrc, vsrc1: u8) -> Vec<u8> {
    let (s0, lit0) = encode_vsrc(src0);
    let word =
        ((op.opcode() & 0x3F) << 25) | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | s0 as u32;
    let mut code = word.to_le_bytes().to_vec();
    push_lit(&mut code, lit0);
    code
}

// ---- VOP1: vector ALU, 1 operand ---------------------------------------------------------
//
// `[31:25]=0b011_1111` (fixed, 7 bits) `[24:17]=VDST(8)` `[16:9]=OP(8)` `[8:0]=SRC0(9)`.
// `ReadfirstlaneB32` writes an SGPR (`VDST` is reused as the SDST field) rather than a VGPR —
// same encoding shape, different destination register class; the caller just passes an SGPR
// number.

const VOP1_FIXED: u32 = 0b011_1111;

#[derive(Clone, Copy)]
pub enum Vop1Op {
    Nop,
    MovB32,
    ReadfirstlaneB32,
    CvtF32I32,
    CvtF32U32,
    CvtU32F32,
    CvtI32F32,
    CvtF32F64,
    CvtF64F32,
    RcpF32,
    SqrtF32,
    NotB32,
}

impl Vop1Op {
    fn opcode(self) -> u32 {
        match self {
            Vop1Op::Nop => 0,
            Vop1Op::MovB32 => 1,
            Vop1Op::ReadfirstlaneB32 => 2,
            Vop1Op::CvtF32I32 => 5,
            Vop1Op::CvtF32U32 => 6,
            Vop1Op::CvtU32F32 => 7,
            Vop1Op::CvtI32F32 => 8,
            Vop1Op::CvtF32F64 => 15,
            Vop1Op::CvtF64F32 => 16,
            Vop1Op::RcpF32 => 42,
            Vop1Op::SqrtF32 => 51,
            Vop1Op::NotB32 => 55,
        }
    }
}

pub fn vop1(op: Vop1Op, vdst: u8, src0: VSrc) -> Vec<u8> {
    let (s0, lit0) = encode_vsrc(src0);
    let word = (VOP1_FIXED << 25) | ((vdst as u32) << 17) | ((op.opcode() & 0xFF) << 9) | s0 as u32;
    let mut code = word.to_le_bytes().to_vec();
    push_lit(&mut code, lit0);
    code
}

pub fn v_nop() -> Vec<u8> {
    vop1(Vop1Op::Nop, 0, VSrc::Sgpr(0))
}

// ---- VOP3: extended 64-bit encoding -------------------------------------------------------
//
// Two dwords. Word0: `[31:26]=0b11_0101` (fixed) `[25:16]=OP(10)` `[15]=CLAMP`
// `[10:8]=ABS(one bit per src0/src1/src2)` `[7:0]=VDST(8)`. Word1: `[31:29]=NEG(one bit per
// src0/src1/src2)` `[28:27]=OMOD(2, 0=none/1=*2/2=*4/3=/2)` `[26:18]=SRC2(9)` `[17:9]=SRC1(9)`
// `[8:0]=SRC0(9)`.
//
// This is the only encoding that gives every operand full VGPR/SGPR/inline-constant
// addressing (VOP2's SRC1 is VGPR-only) plus output modifiers, and the only encoding for
// genuinely 3-operand ops. Opcodes below are exactly the numbers gfx1100 assembles to — for
// the ops that also have a VOP2 form the VOP3 opcode happens to equal `256 + the VOP2 opcode`
// (confirmed per-op below, not assumed), but `FmaF32`/`MulLoU32`/`MulHiU32` have no VOP2 form at
// all and sit in their own part of the opcode space, which is why every value here is a literal
// rather than a computed `256 + vop2_op`.

#[derive(Clone, Copy)]
pub enum Vop3Op {
    CndmaskB32,
    AddF32,
    SubF32,
    SubrevF32,
    MulF32,
    MinF32,
    MaxF32,
    MinI32,
    MaxI32,
    MinU32,
    MaxU32,
    LshlrevB32,
    LshrrevB32,
    AshrrevI32,
    AndB32,
    OrB32,
    XorB32,
    AddNcU32,
    SubNcU32,
    FmaF32,
    MulLoU32,
    MulHiU32,
}

impl Vop3Op {
    fn opcode(self) -> u32 {
        match self {
            Vop3Op::CndmaskB32 => 257,
            Vop3Op::AddF32 => 259,
            Vop3Op::SubF32 => 260,
            Vop3Op::SubrevF32 => 261,
            Vop3Op::MulF32 => 264,
            Vop3Op::MinF32 => 271,
            Vop3Op::MaxF32 => 272,
            Vop3Op::MinI32 => 273,
            Vop3Op::MaxI32 => 274,
            Vop3Op::MinU32 => 275,
            Vop3Op::MaxU32 => 276,
            Vop3Op::LshlrevB32 => 280,
            Vop3Op::LshrrevB32 => 281,
            Vop3Op::AshrrevI32 => 282,
            Vop3Op::AndB32 => 283,
            Vop3Op::OrB32 => 284,
            Vop3Op::XorB32 => 285,
            Vop3Op::AddNcU32 => 293,
            Vop3Op::SubNcU32 => 294,
            Vop3Op::FmaF32 => 531,
            Vop3Op::MulLoU32 => 812,
            Vop3Op::MulHiU32 => 813,
        }
    }
}

/// Output/input modifiers shared by every VOP3 instruction. `neg`/`abs` are indexed by operand
/// (0=src0, 1=src1, 2=src2); an op with fewer than 3 real operands simply leaves the unused
/// slots at their `Default` (`false`).
#[derive(Clone, Copy, Default)]
pub struct Vop3Mods {
    pub neg: [bool; 3],
    pub abs: [bool; 3],
    pub clamp: bool,
    /// 0=none, 1=`*2`, 2=`*4`, 3=`/2`.
    pub omod: u8,
}

/// A VOP3 instruction with an unused source slot (e.g. `src2` on a 2-operand op) takes
/// `VSrc::Sgpr(0)` for that slot — the exact encoding a real assembler produces for an operand
/// position hardware never reads.
pub fn vop3(op: Vop3Op, vdst: u8, src0: VSrc, src1: VSrc, src2: VSrc, mods: Vop3Mods) -> Vec<u8> {
    let (s0, lit0) = encode_vsrc(src0);
    let (s1, lit1) = encode_vsrc(src1);
    let (s2, lit2) = encode_vsrc(src2);
    at_most_one_literal(&[lit0, lit1, lit2]);
    debug_assert!(mods.omod <= 3);
    let abs = (mods.abs[0] as u32) | ((mods.abs[1] as u32) << 1) | ((mods.abs[2] as u32) << 2);
    let word0 = (0b11_0101u32 << 26)
        | ((op.opcode() & 0x3FF) << 16)
        | ((mods.clamp as u32) << 15)
        | (abs << 8)
        | vdst as u32;
    let neg = (mods.neg[0] as u32) | ((mods.neg[1] as u32) << 1) | ((mods.neg[2] as u32) << 2);
    let word1 = (neg << 29)
        | ((mods.omod as u32 & 0x3) << 27)
        | ((s2 as u32) << 18)
        | ((s1 as u32) << 9)
        | s0 as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    push_lit(&mut code, lit0.or(lit1).or(lit2));
    code
}

// ---- VOPC: vector compare -----------------------------------------------------------------
//
// `e32` form (writes `vcc_lo` implicitly): `[31:25]=0b011_1110` (fixed, 7 bits)
// `[24:17]=OP(8)` `[16:9]=VSRC1(8, VGPR)` `[8:0]=SRC0(9)`.
//
// `e64` form (arbitrary SGPR destination, full VOP3 operand addressing on both sources): the
// VOP3 encoding above with `OP` numerically identical to the `e32` opcode (confirmed per-op
// below — VOPC's opcode space is 0-255 and needs no `+256` offset the way a VOP2-promoted
// arithmetic op does) and `VDST` reinterpreted as `SDST`.
//
// Covers `icmp`/`fcmp`'s signed/unsigned-integer and ordered/unordered-float predicates.

#[derive(Clone, Copy)]
pub enum VCmpOp {
    LtF32,
    EqF32,
    LeF32,
    GtF32,
    LgF32,
    GeF32,
    OF32,
    UF32,
    LtI32,
    EqI32,
    LeI32,
    GtI32,
    NeI32,
    GeI32,
    LtU32,
    EqU32,
    LeU32,
    GtU32,
    NeU32,
    GeU32,
}

impl VCmpOp {
    fn opcode(self) -> u32 {
        match self {
            VCmpOp::LtF32 => 17,
            VCmpOp::EqF32 => 18,
            VCmpOp::LeF32 => 19,
            VCmpOp::GtF32 => 20,
            VCmpOp::LgF32 => 21,
            VCmpOp::GeF32 => 22,
            VCmpOp::OF32 => 23,
            VCmpOp::UF32 => 24,
            VCmpOp::LtI32 => 65,
            VCmpOp::EqI32 => 66,
            VCmpOp::LeI32 => 67,
            VCmpOp::GtI32 => 68,
            VCmpOp::NeI32 => 69,
            VCmpOp::GeI32 => 70,
            VCmpOp::LtU32 => 73,
            VCmpOp::EqU32 => 74,
            VCmpOp::LeU32 => 75,
            VCmpOp::GtU32 => 76,
            VCmpOp::NeU32 => 77,
            VCmpOp::GeU32 => 78,
        }
    }
}

pub fn vopc_e32(op: VCmpOp, src0: VSrc, vsrc1: u8) -> Vec<u8> {
    let (s0, lit0) = encode_vsrc(src0);
    let word =
        (0b011_1110u32 << 25) | ((op.opcode() & 0xFF) << 17) | ((vsrc1 as u32) << 9) | s0 as u32;
    let mut code = word.to_le_bytes().to_vec();
    push_lit(&mut code, lit0);
    code
}

pub fn vopc_e64(op: VCmpOp, sdst: u8, src0: VSrc, src1: VSrc) -> Vec<u8> {
    let (s0, lit0) = encode_vsrc(src0);
    let (s1, lit1) = encode_vsrc(src1);
    at_most_one_literal(&[lit0, lit1]);
    let word0 = (0b11_0101u32 << 26) | ((op.opcode() & 0x3FF) << 16) | sdst as u32;
    let word1 = ((s1 as u32) << 9) | s0 as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    push_lit(&mut code, lit0.or(lit1));
    code
}

// ---- VOP3SD: extended encoding with a scalar-register destination alongside VDST -----------
//
// Two dwords, same fixed `[31:26]=0b11_0101` prefix and OP/SRC0/SRC1/SRC2 field positions as
// plain VOP3 (`vop3` above), but word0's bits `[15:8]` (CLAMP+ABS in a normal VOP3 arithmetic
// op) are reinterpreted as a second, 8-bit-wide scalar destination (`SDST`) — there is no
// clamp/abs for these ops, so the ISA reuses the bits. This is the "carry" form of 64-bit
// integer add: `v_add_co_u32` (`VDST := SRC0 + SRC1`, `SDST := carry-out`, no carry-in — the
// low word of a 64-bit add) and `v_add_co_ci_u32`'s `_e64` form (`VDST := SRC0 + SRC1 +
// SRC2-as-carry-in`, `SDST := carry-out` — the same op as `Vop2Op::AddCoCiU32` above but with
// an arbitrary SGPR carry-in/out instead of always `vcc_lo`; not used by this crate's own
// lowering, which always chains through `vcc_lo`, but included since it is the same
// instruction family and free to expose). Confirmed against `llvm-mc-18` (`v_add_co_u32 v0,
// s0, v1, v2` / `v_add_co_ci_u32_e64 v0, s0, v1, v2, s3`) — see the `tests` module.

#[derive(Clone, Copy)]
pub enum Vop3CarryOp {
    AddCoU32,
    AddCoCiU32,
}

impl Vop3CarryOp {
    fn opcode(self) -> u32 {
        match self {
            Vop3CarryOp::AddCoU32 => 768,
            Vop3CarryOp::AddCoCiU32 => 288,
        }
    }
}

/// `sdst` carries the op's carry-out (and, for `AddCoCiU32`, `src2` is the carry-in — pass
/// `VSrc::Sgpr(0)` for `AddCoU32`, which has no carry-in operand, matching the filler
/// convention `vop3` itself already uses for an unused source slot).
pub fn vop3_carry(
    op: Vop3CarryOp,
    vdst: u8,
    sdst: u8,
    src0: VSrc,
    src1: VSrc,
    src2: VSrc,
) -> Vec<u8> {
    debug_assert!(sdst <= 127, "VOP3SD's SDST field is 7 bits wide");
    let (s0, lit0) = encode_vsrc(src0);
    let (s1, lit1) = encode_vsrc(src1);
    let (s2, lit2) = encode_vsrc(src2);
    at_most_one_literal(&[lit0, lit1, lit2]);
    let word0 =
        (0b11_0101u32 << 26) | ((op.opcode() & 0x3FF) << 16) | ((sdst as u32) << 8) | vdst as u32;
    let word1 = ((s2 as u32) << 18) | ((s1 as u32) << 9) | s0 as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    push_lit(&mut code, lit0.or(lit1).or(lit2));
    code
}

// ---- SMEM: scalar memory (kernarg/constant reads) -----------------------------------------
//
// Two dwords. Word0: `[31:26]=0b11_1101` (fixed) `[25:18]=OP(8)` `[14]=GLC` `[13]=DLC`
// `[12:6]=SDATA(7)` `[5:0]=SBASE(6, base register of an aligned SGPR pair, halved)`. Word1:
// `[31:25]=SOFFSET(7, an SGPR; `NULL` means "no extra register offset")` `[20:0]=OFFSET(21,
// signed immediate, added to the pointer in SBASE and to SOFFSET's value if present)`.
//
// Only the plain `s_load_bN` forms are implemented (dwordx1/2/4/8/16), which is exactly what
// reading scalar values — kernel arguments, a `Constant`-address-space scalar — out of the
// kernarg segment through a pointer sitting in an SGPR pair needs. `s_buffer_load` (resource
// descriptor-based constant reads) is not implemented: it needs a 128-bit buffer descriptor
// this crate has no other reason to model yet.

#[derive(Clone, Copy)]
pub enum SmemOp {
    LoadB32,
    LoadB64,
    LoadB128,
    LoadB256,
    LoadB512,
}

impl SmemOp {
    fn opcode(self) -> u32 {
        match self {
            SmemOp::LoadB32 => 0,
            SmemOp::LoadB64 => 1,
            SmemOp::LoadB128 => 2,
            SmemOp::LoadB256 => 3,
            SmemOp::LoadB512 => 4,
        }
    }
}

/// `sbase` is the base register number of the aligned SGPR pair holding the 64-bit source
/// pointer (e.g. `4` for `s[4:5]`), `sdata` is the destination SGPR (or SGPR-tuple base),
/// `offset` is a 21-bit signed byte immediate, and `soffset` is an optional extra SGPR whose
/// value is added to the address (`None` encodes `NULL`, matching what a real assembler emits
/// for a plain-immediate-offset load).
pub fn smem_load(
    op: SmemOp,
    sdata: u8,
    sbase: u8,
    offset: i32,
    soffset: Option<u8>,
    glc: bool,
    dlc: bool,
) -> Vec<u8> {
    debug_assert!(
        sbase.is_multiple_of(2),
        "SBASE must be an aligned register pair"
    );
    debug_assert!(
        (-(1 << 20)..(1 << 20)).contains(&offset),
        "SMEM offset is a 21-bit signed immediate"
    );
    let soff = soffset.unwrap_or(NULL);
    let word0 = (0b11_1101u32 << 26)
        | ((op.opcode() & 0xFF) << 18)
        | ((glc as u32) << 14)
        | ((dlc as u32) << 13)
        | (((sdata as u32) & 0x7F) << 6)
        | (((sbase / 2) as u32) & 0x3F);
    let word1 = (((soff as u32) & 0x7F) << 25) | (offset as u32 & 0x1F_FFFF);
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    code
}

// ---- DS: LDS (shared/local address space) access ------------------------------------------
//
// Two dwords. Word0: `[31:26]=0b11_0110` (fixed) `[25:18]=OP(8)` `[17]=GDS(always 0 here —
// LDS, not GDS)` `[16:8]=OFFSET1(9, unused/0 for the single-address ops below)`
// `[7:0]=OFFSET0(8)`. Word1: `[31:24]=VDST(8)` `[23:16]=DATA1(8, unused/0 here)`
// `[15:8]=DATA0(8)` `[7:0]=ADDR(8)`.
//
// Covers plain (single-address) loads at every width a `Load`/cast pair needs (`b8`/`u8`
// zero-extending, `i8` sign-extending, `u16`/`i16` likewise, `b32`/`b64`/`b128`) and the
// matching stores. Two-address forms (`ds_read2_*`, bank-conflict-avoiding double loads) and
// LDS atomics are not implemented — nothing in the covered op set needs them yet.

#[derive(Clone, Copy)]
pub enum DsLoadOp {
    U8,
    I8,
    U16,
    I16,
    B32,
    B64,
    B128,
}

impl DsLoadOp {
    fn opcode(self) -> u32 {
        match self {
            DsLoadOp::I8 => 57,
            DsLoadOp::U8 => 58,
            DsLoadOp::I16 => 59,
            DsLoadOp::U16 => 60,
            DsLoadOp::B32 => 54,
            DsLoadOp::B64 => 118,
            DsLoadOp::B128 => 255,
        }
    }
}

#[derive(Clone, Copy)]
pub enum DsStoreOp {
    B8,
    B16,
    B32,
    B64,
    B128,
}

impl DsStoreOp {
    fn opcode(self) -> u32 {
        match self {
            DsStoreOp::B8 => 30,
            DsStoreOp::B16 => 31,
            DsStoreOp::B32 => 13,
            DsStoreOp::B64 => 77,
            DsStoreOp::B128 => 223,
        }
    }
}

pub fn ds_load(op: DsLoadOp, vdst: u8, addr: u8, offset0: u8) -> Vec<u8> {
    let word0 = (0b11_0110u32 << 26) | ((op.opcode() & 0xFF) << 18) | offset0 as u32;
    let word1 = (vdst as u32) << 24 | addr as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    code
}

pub fn ds_store(op: DsStoreOp, addr: u8, data0: u8, offset0: u8) -> Vec<u8> {
    let word0 = (0b11_0110u32 << 26) | ((op.opcode() & 0xFF) << 18) | offset0 as u32;
    let word1 = ((data0 as u32) << 8) | addr as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    code
}

// ---- FLAT/GLOBAL: generic/global memory access and atomics --------------------------------
//
// Two dwords, one shared layout for FLAT/SCRATCH/GLOBAL, selected by a 2-bit segment field.
// Word0: `[31:26]=0b11_0111` (fixed) `[25:18]=OP(8)` `[17:16]=SEG(2: 0=flat,1=scratch,
// 2=global)` `[14]=GLC (also "return the pre-atomic value" for atomics)` `[12:0]=OFFSET(13,
// signed)`. Word1: `[31:24]=VDST(8)` `[23:16]=SADDR(8, an SGPR-pair base; `NULL` means the
// address is a full 64-bit VGPR pointer with no SGPR part — the form this crate always uses)`
// `[15:8]=DATA(8)` `[7:0]=ADDR(8, VGPR — base of the 64-bit-pointer pair for `Flat`/`Global`)`.
//
// Only `Global` is exercised by the tests (BIR's `Global` address space); `Flat`/`Scratch` are
// exposed since the encoding is identical and free. Atomic opcodes cover every `AtomicOp`
// variant (`Add`/`Sub`/`Exch`/`Min`/`Max`/`And`/`Or`/`Xor`) plus `cmpswap` for a CAS lowering,
// with separate signed/unsigned min/max primitives since BIR's `Min`/`Max` don't distinguish
// signedness themselves — that choice is a lowering-pass concern, resolved from the element
// type, not this encoder's.

#[derive(Clone, Copy)]
pub enum Seg {
    Flat,
    Scratch,
    Global,
}

impl Seg {
    fn code(self) -> u32 {
        match self {
            Seg::Flat => 0,
            Seg::Scratch => 1,
            Seg::Global => 2,
        }
    }
}

#[derive(Clone, Copy)]
pub enum FlatOp {
    LoadU8,
    LoadI8,
    LoadU16,
    LoadI16,
    LoadB32,
    LoadB64,
    LoadB128,
    StoreB8,
    StoreB16,
    StoreB32,
    StoreB64,
    StoreB128,
    AtomicSwapB32,
    AtomicCmpswapB32,
    AtomicAddU32,
    AtomicSubU32,
    AtomicSminI32,
    AtomicUminU32,
    AtomicSmaxI32,
    AtomicUmaxU32,
    AtomicAndB32,
    AtomicOrB32,
    AtomicXorB32,
}

impl FlatOp {
    fn opcode(self) -> u32 {
        match self {
            FlatOp::LoadU8 => 16,
            FlatOp::LoadI8 => 17,
            FlatOp::LoadU16 => 18,
            FlatOp::LoadI16 => 19,
            FlatOp::LoadB32 => 20,
            FlatOp::LoadB64 => 21,
            FlatOp::LoadB128 => 23,
            FlatOp::StoreB8 => 24,
            FlatOp::StoreB16 => 25,
            FlatOp::StoreB32 => 26,
            FlatOp::StoreB64 => 27,
            FlatOp::StoreB128 => 29,
            FlatOp::AtomicSwapB32 => 51,
            FlatOp::AtomicCmpswapB32 => 52,
            FlatOp::AtomicAddU32 => 53,
            FlatOp::AtomicSubU32 => 54,
            FlatOp::AtomicSminI32 => 56,
            FlatOp::AtomicUminU32 => 57,
            FlatOp::AtomicSmaxI32 => 58,
            FlatOp::AtomicUmaxU32 => 59,
            FlatOp::AtomicAndB32 => 60,
            FlatOp::AtomicOrB32 => 61,
            FlatOp::AtomicXorB32 => 62,
        }
    }
}

fn flat_offset_check(offset: i16) {
    debug_assert!(
        (-4096..4096).contains(&offset),
        "FLAT/GLOBAL offset is a 13-bit signed immediate"
    );
}

/// `addr` is the base VGPR of the 64-bit address pair; `saddr` is an optional SGPR-pair base
/// added to it (`None` encodes `NULL`, i.e. "the VGPR pair alone is the full address" — the
/// form used for a plain global pointer).
pub fn flat_load(
    seg: Seg,
    op: FlatOp,
    vdst: u8,
    addr: u8,
    saddr: Option<u8>,
    offset: i16,
    glc: bool,
) -> Vec<u8> {
    flat_offset_check(offset);
    let sa = saddr.unwrap_or(NULL);
    let word0 = (0b11_0111u32 << 26)
        | ((op.opcode() & 0xFF) << 18)
        | ((seg.code() & 0x3) << 16)
        | ((glc as u32) << 14)
        | (offset as u16 as u32 & 0x1FFF);
    let word1 = ((vdst as u32) << 24) | ((sa as u32) << 16) | addr as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    code
}

pub fn flat_store(
    seg: Seg,
    op: FlatOp,
    addr: u8,
    data: u8,
    saddr: Option<u8>,
    offset: i16,
    glc: bool,
) -> Vec<u8> {
    flat_offset_check(offset);
    let sa = saddr.unwrap_or(NULL);
    let word0 = (0b11_0111u32 << 26)
        | ((op.opcode() & 0xFF) << 18)
        | ((seg.code() & 0x3) << 16)
        | ((glc as u32) << 14)
        | (offset as u16 as u32 & 0x1FFF);
    let word1 = ((sa as u32) << 16) | ((data as u32) << 8) | addr as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    code
}

/// `vdst`: `Some(reg)` requests the pre-atomic-op value back (sets `GLC`, matching hardware —
/// GLC on an atomic means "return the old value"); `None` fires the atomic without a return.
pub fn flat_atomic(
    seg: Seg,
    op: FlatOp,
    vdst: Option<u8>,
    addr: u8,
    data: u8,
    saddr: Option<u8>,
    offset: i16,
) -> Vec<u8> {
    flat_offset_check(offset);
    let sa = saddr.unwrap_or(NULL);
    let glc = vdst.is_some();
    let word0 = (0b11_0111u32 << 26)
        | ((op.opcode() & 0xFF) << 18)
        | ((seg.code() & 0x3) << 16)
        | ((glc as u32) << 14)
        | (offset as u16 as u32 & 0x1FFF);
    let word1 = ((vdst.unwrap_or(0) as u32) << 24)
        | ((sa as u32) << 16)
        | ((data as u32) << 8)
        | addr as u32;
    let mut code = word0.to_le_bytes().to_vec();
    code.extend_from_slice(&word1.to_le_bytes());
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected byte sequence below was derived the same way: write the assembly line(s)
    // shown in the test's own comment to a file and run
    //   llvm-mc-18 -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -show-encoding -filetype=asm <file>
    // LLVM's `-show-encoding` prints each instruction's exact bytes as a trailing comment; that
    // printed byte sequence is copied verbatim into the assertion. Branch-offset resolution
    // (`s_branch`/`s_cbranch`) additionally needed `-filetype=obj` + `llvm-objdump-18 -d` to see
    // the fixup resolved against a concrete label distance, noted on that test specifically.

    // ---- SOP2 ----

    #[test]
    fn sop2_opcodes() {
        // s_add_u32 s0, s1, s2      ; encoding: [0x01,0x02,0x00,0x80]
        // s_sub_u32 s3, s4, s5      ; encoding: [0x04,0x05,0x83,0x80]
        // s_lshl_b32 s0, s1, s2     ; encoding: [0x01,0x02,0x00,0x84]
        // s_lshr_b32 s0, s1, s2     ; encoding: [0x01,0x02,0x00,0x85]
        // s_ashr_i32 s0, s1, s2     ; encoding: [0x01,0x02,0x00,0x86]
        // s_and_b32 s6, s7, s8      ; encoding: [0x07,0x08,0x06,0x8b]
        // s_or_b32 s0, s1, s2       ; encoding: [0x01,0x02,0x00,0x8c]
        // s_xor_b32 s0, s1, s2      ; encoding: [0x01,0x02,0x00,0x8d]
        // s_min_i32 s0, s1, s2      ; encoding: [0x01,0x02,0x00,0x89]
        // s_min_u32 s0, s1, s2      ; encoding: [0x01,0x02,0x80,0x89]
        // s_max_i32 s0, s1, s2      ; encoding: [0x01,0x02,0x00,0x8a]
        // s_max_u32 s0, s1, s2      ; encoding: [0x01,0x02,0x80,0x8a]
        // s_mul_i32 s0, s1, s2      ; encoding: [0x01,0x02,0x00,0x96]
        // s_cselect_b32 s0, s1, s2  ; encoding: [0x01,0x02,0x00,0x98]
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x80]
        );
        assert_eq!(
            sop2(Sop2Op::SubU32, 3, Src::Sgpr(4), Src::Sgpr(5)),
            [0x04, 0x05, 0x83, 0x80]
        );
        assert_eq!(
            sop2(Sop2Op::LshlB32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x84]
        );
        assert_eq!(
            sop2(Sop2Op::LshrB32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x85]
        );
        assert_eq!(
            sop2(Sop2Op::AshrI32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x86]
        );
        assert_eq!(
            sop2(Sop2Op::AndB32, 6, Src::Sgpr(7), Src::Sgpr(8)),
            [0x07, 0x08, 0x06, 0x8b]
        );
        assert_eq!(
            sop2(Sop2Op::OrB32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x8c]
        );
        assert_eq!(
            sop2(Sop2Op::XorB32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x8d]
        );
        assert_eq!(
            sop2(Sop2Op::MinI32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x89]
        );
        assert_eq!(
            sop2(Sop2Op::MinU32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x80, 0x89]
        );
        assert_eq!(
            sop2(Sop2Op::MaxI32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x8a]
        );
        assert_eq!(
            sop2(Sop2Op::MaxU32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x80, 0x8a]
        );
        assert_eq!(
            sop2(Sop2Op::MulI32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x96]
        );
        assert_eq!(
            sop2(Sop2Op::CselectB32, 0, Src::Sgpr(1), Src::Sgpr(2)),
            [0x01, 0x02, 0x00, 0x98]
        );
    }

    #[test]
    fn sop2_inline_int_immediate() {
        // s_add_u32 s0, s1, 42 ; encoding: [0x01,0xaa,0x00,0x80]
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Imm(Imm::Int(42))),
            [0x01, 0xaa, 0x00, 0x80]
        );
        // s_add_u32 s0, s1, 64 ; encoding: [0x01,0xc0,0x00,0x80]
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Imm(Imm::Int(64))),
            [0x01, 0xc0, 0x00, 0x80]
        );
        // s_add_u32 s0, s1, -1  ; encoding: [0x01,0xc1,0x00,0x80]
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Imm(Imm::Int(-1))),
            [0x01, 0xc1, 0x00, 0x80]
        );
        // s_add_u32 s0, s1, -16 ; encoding: [0x01,0xd0,0x00,0x80]
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Imm(Imm::Int(-16))),
            [0x01, 0xd0, 0x00, 0x80]
        );
    }

    #[test]
    fn sop2_literal_spill() {
        // s_add_u32 s0, s1, 65 ; encoding: [0x01,0xff,0x00,0x80,0x41,0x00,0x00,0x00]
        // (65 is one past the inline range 0..=64, so it spills to a trailing literal dword)
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Imm(Imm::Int(65))),
            [0x01, 0xff, 0x00, 0x80, 0x41, 0x00, 0x00, 0x00]
        );
        // s_add_u32 s0, s1, -17 ; encoding: [0x01,0xff,0x00,0x80,0xef,0xff,0xff,0xff]
        assert_eq!(
            sop2(Sop2Op::AddU32, 0, Src::Sgpr(1), Src::Imm(Imm::Int(-17))),
            [0x01, 0xff, 0x00, 0x80, 0xef, 0xff, 0xff, 0xff]
        );
        // s_add_u32 s0, s1, 0x12345678 ; encoding: [0x01,0xff,0x00,0x80,0x78,0x56,0x34,0x12]
        assert_eq!(
            sop2(
                Sop2Op::AddU32,
                0,
                Src::Sgpr(1),
                Src::Imm(Imm::Int(0x1234_5678))
            ),
            [0x01, 0xff, 0x00, 0x80, 0x78, 0x56, 0x34, 0x12]
        );
    }

    // ---- SOP1 ----

    #[test]
    fn sop1_opcodes() {
        // s_mov_b32 s0, s1          ; encoding: [0x01,0x00,0x80,0xbe]
        // s_mov_b64 s[0:1], s[2:3]  ; encoding: [0x02,0x01,0x80,0xbe]
        // s_not_b32 s0, s1          ; encoding: [0x01,0x1e,0x80,0xbe]
        // s_abs_i32 s0, s1          ; encoding: [0x01,0x15,0x80,0xbe]
        // s_bcnt0_i32_b32 s0, s1    ; encoding: [0x01,0x16,0x80,0xbe]
        // s_bcnt1_i32_b32 s0, s1    ; encoding: [0x01,0x18,0x80,0xbe]
        // s_ctz_i32_b32 s0, s1      ; encoding: [0x01,0x08,0x80,0xbe]  (llvm-mc's spelling of
        //                             the classic `s_ff1_i32_b32` mnemonic on gfx11)
        // s_clz_i32_u32 s0, s1      ; encoding: [0x01,0x0a,0x80,0xbe]  (`s_flbit_i32_b32`)
        // s_sext_i32_i8 s0, s1      ; encoding: [0x01,0x0e,0x80,0xbe]
        // s_sext_i32_i16 s0, s1     ; encoding: [0x01,0x0f,0x80,0xbe]
        assert_eq!(
            sop1(Sop1Op::MovB32, 0, Src::Sgpr(1)),
            [0x01, 0x00, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB64, 0, Src::Sgpr(2)),
            [0x02, 0x01, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::NotB32, 0, Src::Sgpr(1)),
            [0x01, 0x1e, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::AbsI32, 0, Src::Sgpr(1)),
            [0x01, 0x15, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::Bcnt0I32B32, 0, Src::Sgpr(1)),
            [0x01, 0x16, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::Bcnt1I32B32, 0, Src::Sgpr(1)),
            [0x01, 0x18, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::CtzI32B32, 0, Src::Sgpr(1)),
            [0x01, 0x08, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::ClzI32U32, 0, Src::Sgpr(1)),
            [0x01, 0x0a, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::SextI32I8, 0, Src::Sgpr(1)),
            [0x01, 0x0e, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::SextI32I16, 0, Src::Sgpr(1)),
            [0x01, 0x0f, 0x80, 0xbe]
        );
    }

    #[test]
    fn sop1_special_registers() {
        // s_mov_b32 vcc_lo, s0   ; encoding: [0x00,0x00,0xea,0xbe]
        // s_mov_b32 vcc_hi, s0   ; encoding: [0x00,0x00,0xeb,0xbe]
        // s_mov_b32 exec_lo, s0  ; encoding: [0x00,0x00,0xfe,0xbe]
        // s_mov_b32 exec_hi, s0  ; encoding: [0x00,0x00,0xff,0xbe]
        // s_mov_b32 m0, s0       ; encoding: [0x00,0x00,0xfd,0xbe]
        // s_mov_b32 null, s0     ; encoding: [0x00,0x00,0xfc,0xbe]
        // s_mov_b32 s105, s0     ; encoding: [0x00,0x00,0xe9,0xbe]
        // s_mov_b32 s0, vcc_lo   ; encoding: [0x6a,0x00,0x80,0xbe]
        // s_mov_b32 s0, exec_lo  ; encoding: [0x7e,0x00,0x80,0xbe]
        // s_mov_b32 s0, m0       ; encoding: [0x7d,0x00,0x80,0xbe]
        // s_mov_b32 s0, null     ; encoding: [0x7c,0x00,0x80,0xbe]
        assert_eq!(
            sop1(Sop1Op::MovB32, VCC_LO, Src::Sgpr(0)),
            [0x00, 0x00, 0xea, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, VCC_HI, Src::Sgpr(0)),
            [0x00, 0x00, 0xeb, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, EXEC_LO, Src::Sgpr(0)),
            [0x00, 0x00, 0xfe, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, EXEC_HI, Src::Sgpr(0)),
            [0x00, 0x00, 0xff, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, M0, Src::Sgpr(0)),
            [0x00, 0x00, 0xfd, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, NULL, Src::Sgpr(0)),
            [0x00, 0x00, 0xfc, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, 105, Src::Sgpr(0)),
            [0x00, 0x00, 0xe9, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, 0, Src::Sgpr(VCC_LO)),
            [0x6a, 0x00, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, 0, Src::Sgpr(EXEC_LO)),
            [0x7e, 0x00, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, 0, Src::Sgpr(M0)),
            [0x7d, 0x00, 0x80, 0xbe]
        );
        assert_eq!(
            sop1(Sop1Op::MovB32, 0, Src::Sgpr(NULL)),
            [0x7c, 0x00, 0x80, 0xbe]
        );
    }

    // ---- SOPK ----

    #[test]
    fn sopk_opcodes() {
        // s_movk_i32 s0, 0x1234 ; encoding: [0x34,0x12,0x00,0xb0]
        // s_movk_i32 s0, 0xffff ; encoding: [0xff,0xff,0x00,0xb0]  (`s_movk_i32 s0, -1`)
        // s_addk_i32 s0, 0x10   ; encoding: [0x10,0x00,0x80,0xb7]
        assert_eq!(sopk(SopkOp::MovkI32, 0, 0x1234), [0x34, 0x12, 0x00, 0xb0]);
        assert_eq!(sopk(SopkOp::MovkI32, 0, -1), [0xff, 0xff, 0x00, 0xb0]);
        assert_eq!(sopk(SopkOp::AddkI32, 0, 0x10), [0x10, 0x00, 0x80, 0xb7]);
    }

    // ---- SOPC ----

    #[test]
    fn sopc_opcodes() {
        // s_cmp_eq_i32 s0, s1    ; encoding: [0x00,0x01,0x00,0xbf]
        // s_cmp_lg_i32 s0, s1    ; encoding: [0x00,0x01,0x01,0xbf]
        // s_cmp_gt_i32 s0, s1    ; encoding: [0x00,0x01,0x02,0xbf]
        // s_cmp_ge_i32 s0, s1    ; encoding: [0x00,0x01,0x03,0xbf]
        // s_cmp_lt_i32 s0, s1    ; encoding: [0x00,0x01,0x04,0xbf]
        // s_cmp_le_i32 s0, s1    ; encoding: [0x00,0x01,0x05,0xbf]
        // s_cmp_eq_u32 s0, s1    ; encoding: [0x00,0x01,0x06,0xbf]
        // s_cmp_lg_u32 s0, s1    ; encoding: [0x00,0x01,0x07,0xbf]
        // s_cmp_gt_u32 s0, s1    ; encoding: [0x00,0x01,0x08,0xbf]
        // s_cmp_ge_u32 s0, s1    ; encoding: [0x00,0x01,0x09,0xbf]
        // s_cmp_lt_u32 s0, s1    ; encoding: [0x00,0x01,0x0a,0xbf]
        // s_cmp_le_u32 s0, s1    ; encoding: [0x00,0x01,0x0b,0xbf]
        // s_bitcmp0_b32 s0, s1   ; encoding: [0x00,0x01,0x0c,0xbf]
        // s_bitcmp1_b32 s0, s1   ; encoding: [0x00,0x01,0x0d,0xbf]
        // s_cmp_eq_i32 s0, 42    ; encoding: [0x00,0xaa,0x00,0xbf]
        assert_eq!(
            sopc(SopcOp::EqI32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x00, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::LgI32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x01, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::GtI32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x02, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::GeI32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x03, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::LtI32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x04, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::LeI32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x05, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::EqU32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x06, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::LgU32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x07, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::GtU32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x08, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::GeU32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x09, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::LtU32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x0a, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::LeU32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x0b, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::Bitcmp0B32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x0c, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::Bitcmp1B32, Src::Sgpr(0), Src::Sgpr(1)),
            [0x00, 0x01, 0x0d, 0xbf]
        );
        assert_eq!(
            sopc(SopcOp::EqI32, Src::Sgpr(0), Src::Imm(Imm::Int(42))),
            [0x00, 0xaa, 0x00, 0xbf]
        );
    }

    // ---- SOPP ----

    #[test]
    fn sopp_fixed_form_opcodes() {
        // s_endpgm  ; encoding: [0x00,0x00,0xb0,0xbf]
        // s_nop 0   ; encoding: [0x00,0x00,0x80,0xbf]
        // s_nop 3   ; encoding: [0x03,0x00,0x80,0xbf]
        // s_barrier ; encoding: [0x00,0x00,0xbd,0xbf]
        assert_eq!(s_endpgm(), [0x00, 0x00, 0xb0, 0xbf]);
        assert_eq!(s_nop(0), [0x00, 0x00, 0x80, 0xbf]);
        assert_eq!(s_nop(3), [0x03, 0x00, 0x80, 0xbf]);
        assert_eq!(s_barrier(), [0x00, 0x00, 0xbd, 0xbf]);
    }

    #[test]
    fn sopp_waitcnt_bitfield() {
        // s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)   ; encoding: [0x00,0x00,0x89,0xbf]
        // s_waitcnt vmcnt(1) expcnt(0) lgkmcnt(0)   ; encoding: [0x00,0x04,0x89,0xbf]
        // s_waitcnt vmcnt(0) expcnt(1) lgkmcnt(0)   ; encoding: [0x01,0x00,0x89,0xbf]
        // s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(1)   ; encoding: [0x10,0x00,0x89,0xbf]
        // s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(63) ; encoding: [0xf7,0xff,0x89,0xbf]
        assert_eq!(s_waitcnt(0, 0, 0), [0x00, 0x00, 0x89, 0xbf]);
        assert_eq!(s_waitcnt(1, 0, 0), [0x00, 0x04, 0x89, 0xbf]);
        assert_eq!(s_waitcnt(0, 1, 0), [0x01, 0x00, 0x89, 0xbf]);
        assert_eq!(s_waitcnt(0, 0, 1), [0x10, 0x00, 0x89, 0xbf]);
        assert_eq!(s_waitcnt(63, 7, 63), [0xf7, 0xff, 0x89, 0xbf]);
    }

    #[test]
    fn sopp_branch_word_offset() {
        // Assembled and linked (`llvm-mc-18 -filetype=obj` + `llvm-objdump-18 -d`) rather than
        // read off `-show-encoding`, since a forward/backward branch's simm16 only exists once
        // the fixup against a real label is resolved:
        //   s_branch target      s_nop 0
        //   s_nop 0        ->    s_nop 0
        //   s_nop 0              s_branch back
        //   target:              (back: is the first s_nop)
        //   s_nop 0
        // objdump: forward branch -> BFA00002 (simm16=2, two words to `target`)
        //          backward branch -> BFA0FFFD (simm16=-3, three words back to `back`)
        assert_eq!(s_branch(2), [0x02, 0x00, 0xa0, 0xbf]);
        assert_eq!(s_branch(-3), [0xfd, 0xff, 0xa0, 0xbf]);
    }

    #[test]
    fn sopp_cbranch_opcodes() {
        // s_cbranch_scc0 lbl   ; encoding: [..,..,0xa1,0xbf]
        // s_cbranch_scc1 lbl   ; encoding: [..,..,0xa2,0xbf]
        // s_cbranch_vccz lbl   ; encoding: [..,..,0xa3,0xbf]
        // s_cbranch_vccnz lbl  ; encoding: [..,..,0xa4,0xbf]
        // s_cbranch_execz lbl  ; encoding: [..,..,0xa5,0xbf]
        // s_cbranch_execnz lbl ; encoding: [..,..,0xa6,0xbf]
        assert_eq!(s_cbranch(BrCc::Scc0, 0), [0x00, 0x00, 0xa1, 0xbf]);
        assert_eq!(s_cbranch(BrCc::Scc1, 0), [0x00, 0x00, 0xa2, 0xbf]);
        assert_eq!(s_cbranch(BrCc::Vccz, 0), [0x00, 0x00, 0xa3, 0xbf]);
        assert_eq!(s_cbranch(BrCc::Vccnz, 0), [0x00, 0x00, 0xa4, 0xbf]);
        assert_eq!(s_cbranch(BrCc::Execz, 0), [0x00, 0x00, 0xa5, 0xbf]);
        assert_eq!(s_cbranch(BrCc::Execnz, 0), [0x00, 0x00, 0xa6, 0xbf]);
    }

    // ---- VOP2 ----

    #[test]
    fn vop2_opcodes() {
        // v_add_f32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x06]
        // v_sub_f32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x08]
        // v_subrev_f32_e32 v0, v1, v2    ; encoding: [0x01,0x05,0x00,0x0a]
        // v_mul_f32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x10]
        // v_min_f32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x1e]
        // v_max_f32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x20]
        // v_min_i32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x22]
        // v_max_i32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x24]
        // v_min_u32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x26]
        // v_max_u32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x28]
        // v_lshlrev_b32_e32 v0, v1, v2   ; encoding: [0x01,0x05,0x00,0x30]
        // v_lshrrev_b32_e32 v0, v1, v2   ; encoding: [0x01,0x05,0x00,0x32]
        // v_ashrrev_i32_e32 v0, v1, v2   ; encoding: [0x01,0x05,0x00,0x34]
        // v_and_b32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x36]
        // v_or_b32_e32 v0, v1, v2        ; encoding: [0x01,0x05,0x00,0x38]
        // v_xor_b32_e32 v0, v1, v2       ; encoding: [0x01,0x05,0x00,0x3a]
        // v_add_nc_u32_e32 v0, v1, v2    ; encoding: [0x01,0x05,0x00,0x4a]
        // v_sub_nc_u32_e32 v0, v1, v2    ; encoding: [0x01,0x05,0x00,0x4c]
        // v_cndmask_b32_e32 v0, v1, v2, vcc_lo ; encoding: [0x01,0x05,0x00,0x02]
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::SubF32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x08]
        );
        assert_eq!(
            vop2(Vop2Op::SubrevF32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x0a]
        );
        assert_eq!(
            vop2(Vop2Op::MulF32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x10]
        );
        assert_eq!(
            vop2(Vop2Op::MinF32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x1e]
        );
        assert_eq!(
            vop2(Vop2Op::MaxF32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x20]
        );
        assert_eq!(
            vop2(Vop2Op::MinI32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x22]
        );
        assert_eq!(
            vop2(Vop2Op::MaxI32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x24]
        );
        assert_eq!(
            vop2(Vop2Op::MinU32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x26]
        );
        assert_eq!(
            vop2(Vop2Op::MaxU32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x28]
        );
        assert_eq!(
            vop2(Vop2Op::LshlrevB32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x30]
        );
        assert_eq!(
            vop2(Vop2Op::LshrrevB32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x32]
        );
        assert_eq!(
            vop2(Vop2Op::AshrrevI32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x34]
        );
        assert_eq!(
            vop2(Vop2Op::AndB32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x36]
        );
        assert_eq!(
            vop2(Vop2Op::OrB32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x38]
        );
        assert_eq!(
            vop2(Vop2Op::XorB32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x3a]
        );
        assert_eq!(
            vop2(Vop2Op::AddNcU32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x4a]
        );
        assert_eq!(
            vop2(Vop2Op::SubNcU32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x4c]
        );
        assert_eq!(
            vop2(Vop2Op::CndmaskB32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x02]
        );
    }

    #[test]
    fn vop2_sgpr_and_inline_float_src0() {
        // v_add_f32_e32 v0, s1, v2   ; encoding: [0x01,0x04,0x00,0x06]
        // v_add_f32_e32 v0, 1.0, v2  ; encoding: [0xf2,0x04,0x00,0x06]
        // v_add_f32_e32 v0, 0.5, v2  ; encoding: [0xf0,0x04,0x00,0x06]
        // v_add_f32_e32 v0, -0.5, v2 ; encoding: [0xf1,0x04,0x00,0x06]
        // v_add_f32_e32 v0, 2.0, v2  ; encoding: [0xf4,0x04,0x00,0x06]
        // v_add_f32_e32 v0, 4.0, v2  ; encoding: [0xf6,0x04,0x00,0x06]
        // v_add_f32_e32 v0, -4.0, v2 ; encoding: [0xf7,0x04,0x00,0x06]
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Sgpr(1), 2),
            [0x01, 0x04, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::F32(1.0)), 2),
            [0xf2, 0x04, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::F32(0.5)), 2),
            [0xf0, 0x04, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::F32(-0.5)), 2),
            [0xf1, 0x04, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::F32(2.0)), 2),
            [0xf4, 0x04, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::F32(4.0)), 2),
            [0xf6, 0x04, 0x00, 0x06]
        );
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::F32(-4.0)), 2),
            [0xf7, 0x04, 0x00, 0x06]
        );
    }

    #[test]
    fn vop2_literal_spill() {
        // v_add_f32_e32 v0, 0x12345678, v2 ; encoding: [0xff,0x04,0x00,0x06,0x78,0x56,0x34,0x12]
        assert_eq!(
            vop2(Vop2Op::AddF32, 0, VSrc::Imm(Imm::Raw(0x1234_5678)), 2),
            [0xff, 0x04, 0x00, 0x06, 0x78, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn vop2_add_co_ci_u32() {
        // v_add_co_ci_u32_e32 v0, vcc_lo, v1, v2, vcc_lo ; encoding: [0x01,0x05,0x00,0x40]
        assert_eq!(
            vop2(Vop2Op::AddCoCiU32, 0, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x00, 0x40]
        );
    }

    // ---- VOP1 ----

    #[test]
    fn vop1_opcodes() {
        // v_nop                    ; encoding: [0x00,0x00,0x00,0x7e]
        // v_mov_b32_e32 v0, v1     ; encoding: [0x01,0x03,0x00,0x7e]
        // v_readfirstlane_b32 s0, v1 ; encoding: [0x01,0x05,0x00,0x7e]
        // v_cvt_f32_i32_e32 v0, v1 ; encoding: [0x01,0x0b,0x00,0x7e]
        // v_cvt_f32_u32_e32 v0, v1 ; encoding: [0x01,0x0d,0x00,0x7e]
        // v_cvt_u32_f32_e32 v0, v1 ; encoding: [0x01,0x0f,0x00,0x7e]
        // v_cvt_i32_f32_e32 v0, v1 ; encoding: [0x01,0x11,0x00,0x7e]
        // v_cvt_f32_f64_e32 v0, v[1:2] ; encoding: [0x01,0x1f,0x00,0x7e]
        // v_cvt_f64_f32_e32 v[0:1], v2 ; encoding: [0x02,0x21,0x00,0x7e]
        // v_rcp_f32_e32 v0, v1     ; encoding: [0x01,0x55,0x00,0x7e]
        // v_sqrt_f32_e32 v0, v1    ; encoding: [0x01,0x67,0x00,0x7e]
        // v_not_b32_e32 v0, v1     ; encoding: [0x01,0x6f,0x00,0x7e]
        assert_eq!(v_nop(), [0x00, 0x00, 0x00, 0x7e]);
        assert_eq!(
            vop1(Vop1Op::MovB32, 0, VSrc::Vgpr(1)),
            [0x01, 0x03, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::ReadfirstlaneB32, 0, VSrc::Vgpr(1)),
            [0x01, 0x05, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::CvtF32I32, 0, VSrc::Vgpr(1)),
            [0x01, 0x0b, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::CvtF32U32, 0, VSrc::Vgpr(1)),
            [0x01, 0x0d, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::CvtU32F32, 0, VSrc::Vgpr(1)),
            [0x01, 0x0f, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::CvtI32F32, 0, VSrc::Vgpr(1)),
            [0x01, 0x11, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::CvtF32F64, 0, VSrc::Vgpr(1)),
            [0x01, 0x1f, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::CvtF64F32, 0, VSrc::Vgpr(2)),
            [0x02, 0x21, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::RcpF32, 0, VSrc::Vgpr(1)),
            [0x01, 0x55, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::SqrtF32, 0, VSrc::Vgpr(1)),
            [0x01, 0x67, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::NotB32, 0, VSrc::Vgpr(1)),
            [0x01, 0x6f, 0x00, 0x7e]
        );
    }

    #[test]
    fn vop1_inline_immediate_and_high_vgpr() {
        // v_mov_b32_e32 v0, 42  ; encoding: [0xaa,0x02,0x00,0x7e]
        // v_mov_b32_e32 v255, v1 ; encoding: [0x01,0x03,0xfe,0x7f]
        assert_eq!(
            vop1(Vop1Op::MovB32, 0, VSrc::Imm(Imm::Int(42))),
            [0xaa, 0x02, 0x00, 0x7e]
        );
        assert_eq!(
            vop1(Vop1Op::MovB32, 255, VSrc::Vgpr(1)),
            [0x01, 0x03, 0xfe, 0x7f]
        );
    }

    // ---- VOP3 ----

    #[test]
    fn vop3_promoted_arithmetic_opcodes() {
        // v_add_f32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x03,0xd5,0x01,0x05,0x02,0x00]
        // v_sub_f32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x04,0xd5,0x01,0x05,0x02,0x00]
        // v_subrev_f32_e64 v0, v1, v2   ; encoding: [0x00,0x00,0x05,0xd5,0x01,0x05,0x02,0x00]
        // v_mul_f32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x08,0xd5,0x01,0x05,0x02,0x00]
        // v_min_f32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x0f,0xd5,0x01,0x05,0x02,0x00]
        // v_max_f32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x10,0xd5,0x01,0x05,0x02,0x00]
        // v_min_i32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x11,0xd5,0x01,0x05,0x02,0x00]
        // v_max_i32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x12,0xd5,0x01,0x05,0x02,0x00]
        // v_min_u32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x13,0xd5,0x01,0x05,0x02,0x00]
        // v_max_u32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x14,0xd5,0x01,0x05,0x02,0x00]
        // v_lshlrev_b32_e64 v0, v1, v2  ; encoding: [0x00,0x00,0x18,0xd5,0x01,0x05,0x02,0x00]
        // v_lshrrev_b32_e64 v0, v1, v2  ; encoding: [0x00,0x00,0x19,0xd5,0x01,0x05,0x02,0x00]
        // v_ashrrev_i32_e64 v0, v1, v2  ; encoding: [0x00,0x00,0x1a,0xd5,0x01,0x05,0x02,0x00]
        // v_and_b32_e64 v0, s1, s2      ; encoding: [0x00,0x00,0x1b,0xd5,0x01,0x04,0x00,0x00]
        // v_or_b32_e64 v0, v1, v2       ; encoding: [0x00,0x00,0x1c,0xd5,0x01,0x05,0x02,0x00]
        // v_xor_b32_e64 v0, v1, v2      ; encoding: [0x00,0x00,0x1d,0xd5,0x01,0x05,0x02,0x00]
        // v_add_nc_u32_e64 v0, v1, v2   ; encoding: [0x00,0x00,0x25,0xd5,0x01,0x05,0x02,0x00]
        // v_sub_nc_u32_e64 v0, v1, v2   ; encoding: [0x00,0x00,0x26,0xd5,0x01,0x05,0x02,0x00]
        // v_cndmask_b32_e64 v0, v1, v2, vcc_lo ; encoding: [0x00,0x00,0x01,0xd5,0x01,0x05,0xaa,0x01]
        let m = Vop3Mods::default();
        let z = VSrc::Sgpr(0);
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::SubF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x04, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::SubrevF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x05, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MulF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x08, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MinF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x0f, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MaxF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x10, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MinI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x11, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MaxI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x12, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MinU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x13, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::MaxU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x14, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::LshlrevB32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x18, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::LshrrevB32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x19, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::AshrrevI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x1a, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::AndB32, 0, VSrc::Sgpr(1), VSrc::Sgpr(2), z, m),
            [0x00, 0x00, 0x1b, 0xd5, 0x01, 0x04, 0x00, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::OrB32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x1c, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::XorB32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x1d, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::AddNcU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x25, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(Vop3Op::SubNcU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x26, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(
                Vop3Op::CndmaskB32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(VCC_LO),
                m
            ),
            [0x00, 0x00, 0x01, 0xd5, 0x01, 0x05, 0xaa, 0x01]
        );
    }

    #[test]
    fn vop3_only_opcodes() {
        // v_fma_f32 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x13,0xd6,0x01,0x05,0x0e,0x04]
        // v_mul_lo_u32 v0, v1, v2  ; encoding: [0x00,0x00,0x2c,0xd7,0x01,0x05,0x02,0x00]
        let m = Vop3Mods::default();
        assert_eq!(
            vop3(
                Vop3Op::FmaF32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Vgpr(3),
                m
            ),
            [0x00, 0x00, 0x13, 0xd6, 0x01, 0x05, 0x0e, 0x04]
        );
        assert_eq!(
            vop3(
                Vop3Op::MulLoU32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(0),
                m
            ),
            [0x00, 0x00, 0x2c, 0xd7, 0x01, 0x05, 0x02, 0x00]
        );
    }

    #[test]
    fn vop3_mul_hi_u32() {
        // v_mul_hi_u32 v0, v1, v2 ; encoding: [0x00,0x00,0x2d,0xd7,0x01,0x05,0x02,0x00]
        // v_mul_hi_u32 v3, s0, v2 ; encoding: [0x03,0x00,0x2d,0xd7,0x00,0x04,0x02,0x00]
        // v_mul_hi_u32 v0, v9, v2 ; encoding: [0x00,0x00,0x2d,0xd7,0x09,0x05,0x02,0x00]
        let m = Vop3Mods::default();
        assert_eq!(
            vop3(
                Vop3Op::MulHiU32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(0),
                m
            ),
            [0x00, 0x00, 0x2d, 0xd7, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3(
                Vop3Op::MulHiU32,
                3,
                VSrc::Sgpr(0),
                VSrc::Vgpr(2),
                VSrc::Sgpr(0),
                m
            ),
            [0x03, 0x00, 0x2d, 0xd7, 0x00, 0x04, 0x02, 0x00]
        );
        assert_eq!(
            vop3(
                Vop3Op::MulHiU32,
                0,
                VSrc::Vgpr(9),
                VSrc::Vgpr(2),
                VSrc::Sgpr(0),
                m
            ),
            [0x00, 0x00, 0x2d, 0xd7, 0x09, 0x05, 0x02, 0x00]
        );
    }

    #[test]
    fn vop3_modifiers() {
        // v_add_f32_e64 v0, -v1, v2      ; encoding: [0x00,0x00,0x03,0xd5,0x01,0x05,0x02,0x20]
        // v_add_f32_e64 v0, v1, -v2      ; encoding: [0x00,0x00,0x03,0xd5,0x01,0x05,0x02,0x40]
        // v_add_f32_e64 v0, |v1|, v2     ; encoding: [0x00,0x01,0x03,0xd5,0x01,0x05,0x02,0x00]
        // v_add_f32_e64 v0, v1, v2 clamp ; encoding: [0x00,0x80,0x03,0xd5,0x01,0x05,0x02,0x00]
        // v_add_f32_e64 v0, v1, v2 mul:2 ; encoding: [0x00,0x00,0x03,0xd5,0x01,0x05,0x02,0x08]
        // v_add_f32_e64 v0, v1, v2 mul:4 ; encoding: [0x00,0x00,0x03,0xd5,0x01,0x05,0x02,0x10]
        // v_add_f32_e64 v0, v1, v2 div:2 ; encoding: [0x00,0x00,0x03,0xd5,0x01,0x05,0x02,0x18]
        // v_fma_f32 v0, -v1, v2, v3      ; encoding: [0x00,0x00,0x13,0xd6,0x01,0x05,0x0e,0x24]
        // v_fma_f32 v0, v1, v2, -v3      ; encoding: [0x00,0x00,0x13,0xd6,0x01,0x05,0x0e,0x84]
        let z = VSrc::Sgpr(0);
        let mut m = Vop3Mods {
            neg: [true, false, false],
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x20]
        );
        m = Vop3Mods {
            neg: [false, true, false],
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x40]
        );
        m = Vop3Mods {
            abs: [true, false, false],
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x01, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        m = Vop3Mods {
            clamp: true,
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x80, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
        m = Vop3Mods {
            omod: 1,
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x08]
        );
        m = Vop3Mods {
            omod: 2,
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x10]
        );
        m = Vop3Mods {
            omod: 3,
            ..Default::default()
        };
        assert_eq!(
            vop3(Vop3Op::AddF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z, m),
            [0x00, 0x00, 0x03, 0xd5, 0x01, 0x05, 0x02, 0x18]
        );
        m = Vop3Mods {
            neg: [true, false, false],
            ..Default::default()
        };
        assert_eq!(
            vop3(
                Vop3Op::FmaF32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Vgpr(3),
                m
            ),
            [0x00, 0x00, 0x13, 0xd6, 0x01, 0x05, 0x0e, 0x24]
        );
        m = Vop3Mods {
            neg: [false, false, true],
            ..Default::default()
        };
        assert_eq!(
            vop3(
                Vop3Op::FmaF32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Vgpr(3),
                m
            ),
            [0x00, 0x00, 0x13, 0xd6, 0x01, 0x05, 0x0e, 0x84]
        );
    }

    // ---- VOPC ----

    #[test]
    fn vopc_e32_opcodes() {
        // v_cmp_eq_f32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x24,0x7c]
        // v_cmp_lt_f32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x22,0x7c]
        // v_cmp_le_f32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x26,0x7c]
        // v_cmp_gt_f32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x28,0x7c]
        // v_cmp_ge_f32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x2c,0x7c]
        // v_cmp_lg_f32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x2a,0x7c]
        // v_cmp_o_f32_e32 vcc_lo, v1, v2  ; encoding: [0x01,0x05,0x2e,0x7c]
        // v_cmp_u_f32_e32 vcc_lo, v1, v2  ; encoding: [0x01,0x05,0x30,0x7c]
        // v_cmp_eq_i32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x84,0x7c]
        // v_cmp_ne_i32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x8a,0x7c]
        // v_cmp_lt_i32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x82,0x7c]
        // v_cmp_le_i32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x86,0x7c]
        // v_cmp_gt_i32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x88,0x7c]
        // v_cmp_ge_i32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x8c,0x7c]
        // v_cmp_lt_u32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x92,0x7c]
        // v_cmp_le_u32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x96,0x7c]
        // v_cmp_gt_u32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x98,0x7c]
        // v_cmp_ge_u32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x9c,0x7c]
        // v_cmp_eq_u32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x94,0x7c]
        // v_cmp_ne_u32_e32 vcc_lo, v1, v2 ; encoding: [0x01,0x05,0x9a,0x7c]
        assert_eq!(
            vopc_e32(VCmpOp::EqF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x24, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LtF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x22, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LeF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x26, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::GtF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x28, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::GeF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x2c, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LgF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x2a, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::OF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x2e, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::UF32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x30, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::EqI32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x84, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::NeI32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x8a, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LtI32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x82, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LeI32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x86, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::GtI32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x88, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::GeI32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x8c, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LtU32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x92, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::LeU32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x96, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::GtU32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x98, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::GeU32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x9c, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::EqU32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x94, 0x7c]
        );
        assert_eq!(
            vopc_e32(VCmpOp::NeU32, VSrc::Vgpr(1), 2),
            [0x01, 0x05, 0x9a, 0x7c]
        );
    }

    #[test]
    fn vopc_e64_opcodes() {
        // v_cmp_eq_f32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x12,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_lt_f32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x11,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_le_f32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x13,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_gt_f32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x14,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_ge_f32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x16,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_lg_f32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x15,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_o_f32_e64 s0, v1, v2  ; encoding: [0x00,0x00,0x17,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_u_f32_e64 s0, v1, v2  ; encoding: [0x00,0x00,0x18,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_lt_i32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x41,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_ne_i32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x45,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_le_i32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x43,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_gt_i32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x44,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_ge_i32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x46,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_lt_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x49,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_le_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x4b,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_gt_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x4c,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_ge_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x4e,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_eq_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x4a,0xd4,0x01,0x05,0x02,0x00]
        // v_cmp_ne_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0x4d,0xd4,0x01,0x05,0x02,0x00]
        assert_eq!(
            vopc_e64(VCmpOp::EqF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x12, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LtF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x11, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LeF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x13, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::GtF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x14, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::GeF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x16, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LgF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x15, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::OF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x17, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::UF32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x18, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LtI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x41, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::NeI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x45, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LeI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x43, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::GtI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x44, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::GeI32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x46, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LtU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x49, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::LeU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x4b, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::GtU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x4c, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::GeU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x4e, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::EqU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x4a, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vopc_e64(VCmpOp::NeU32, 0, VSrc::Vgpr(1), VSrc::Vgpr(2)),
            [0x00, 0x00, 0x4d, 0xd4, 0x01, 0x05, 0x02, 0x00]
        );
    }

    #[test]
    fn vopc_e64_vcc_lo_from_e32() {
        // v_cndmask_b32_e64 v0, v1, v2, vcc_lo -- the e64 form above already covers a real
        // SGPR sdst (s0); this test instead pins the e32 form's fixed vcc_lo destination is
        // never actually encoded (vopc_e32 has no dst parameter at all), and separately checks
        // encoding an explicit SGPR destination equal to vcc_lo through the e64 path:
        // v_cndmask_b32_e64 v0, v1, v2, s0 ; encoding: [0x00,0x00,0x01,0xd5,0x01,0x05,0x02,0x00]
        let m = Vop3Mods::default();
        assert_eq!(
            vop3(
                Vop3Op::CndmaskB32,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(0),
                m
            ),
            [0x00, 0x00, 0x01, 0xd5, 0x01, 0x05, 0x02, 0x00]
        );
    }

    // ---- VOP3SD ----

    #[test]
    fn vop3_carry_add_co_u32() {
        // v_add_co_u32 v0, s0, v1, v2 ; encoding: [0x00,0x00,0x00,0xd7,0x01,0x05,0x02,0x00]
        // v_add_co_u32 v3, s0, v1, v2 ; encoding: [0x03,0x00,0x00,0xd7,0x01,0x05,0x02,0x00]
        // v_add_co_u32 v0, s4, v1, v2 ; encoding: [0x00,0x04,0x00,0xd7,0x01,0x05,0x02,0x00]
        // v_add_co_u32 v0, vcc_lo, v1, v2 ; encoding: [0x00,0x6a,0x00,0xd7,0x01,0x05,0x02,0x00]
        let z = VSrc::Sgpr(0);
        assert_eq!(
            vop3_carry(Vop3CarryOp::AddCoU32, 0, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z),
            [0x00, 0x00, 0x00, 0xd7, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3_carry(Vop3CarryOp::AddCoU32, 3, 0, VSrc::Vgpr(1), VSrc::Vgpr(2), z),
            [0x03, 0x00, 0x00, 0xd7, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3_carry(Vop3CarryOp::AddCoU32, 0, 4, VSrc::Vgpr(1), VSrc::Vgpr(2), z),
            [0x00, 0x04, 0x00, 0xd7, 0x01, 0x05, 0x02, 0x00]
        );
        assert_eq!(
            vop3_carry(
                Vop3CarryOp::AddCoU32,
                0,
                VCC_LO,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                z
            ),
            [0x00, 0x6a, 0x00, 0xd7, 0x01, 0x05, 0x02, 0x00]
        );
    }

    #[test]
    fn vop3_carry_add_co_ci_u32() {
        // v_add_co_ci_u32_e64 v0, s0, v1, v2, s3       ; encoding: [0x00,0x00,0x20,0xd5,0x01,0x05,0x0e,0x00]
        // v_add_co_ci_u32_e64 v0, s4, v1, v2, s3        ; encoding: [0x00,0x04,0x20,0xd5,0x01,0x05,0x0e,0x00]
        // v_add_co_ci_u32_e64 v7, s0, v1, v2, s3        ; encoding: [0x07,0x00,0x20,0xd5,0x01,0x05,0x0e,0x00]
        // v_add_co_ci_u32_e64 v0, s0, v9, v2, s3        ; encoding: [0x00,0x00,0x20,0xd5,0x09,0x05,0x0e,0x00]
        // v_add_co_ci_u32_e64 v0, s0, v1, v2, vcc_lo    ; encoding: [0x00,0x00,0x20,0xd5,0x01,0x05,0xaa,0x01]
        assert_eq!(
            vop3_carry(
                Vop3CarryOp::AddCoCiU32,
                0,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(3)
            ),
            [0x00, 0x00, 0x20, 0xd5, 0x01, 0x05, 0x0e, 0x00]
        );
        assert_eq!(
            vop3_carry(
                Vop3CarryOp::AddCoCiU32,
                0,
                4,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(3)
            ),
            [0x00, 0x04, 0x20, 0xd5, 0x01, 0x05, 0x0e, 0x00]
        );
        assert_eq!(
            vop3_carry(
                Vop3CarryOp::AddCoCiU32,
                7,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(3)
            ),
            [0x07, 0x00, 0x20, 0xd5, 0x01, 0x05, 0x0e, 0x00]
        );
        assert_eq!(
            vop3_carry(
                Vop3CarryOp::AddCoCiU32,
                0,
                0,
                VSrc::Vgpr(9),
                VSrc::Vgpr(2),
                VSrc::Sgpr(3)
            ),
            [0x00, 0x00, 0x20, 0xd5, 0x09, 0x05, 0x0e, 0x00]
        );
        assert_eq!(
            vop3_carry(
                Vop3CarryOp::AddCoCiU32,
                0,
                0,
                VSrc::Vgpr(1),
                VSrc::Vgpr(2),
                VSrc::Sgpr(VCC_LO)
            ),
            [0x00, 0x00, 0x20, 0xd5, 0x01, 0x05, 0xaa, 0x01]
        );
    }

    // ---- SMEM ----

    #[test]
    fn smem_opcodes_and_fields() {
        // s_load_b32 s0, s[4:5], 0x0    ; encoding: [0x02,0x00,0x00,0xf4,0x00,0x00,0x00,0xf8]
        // s_load_b64 s[0:1], s[4:5], 0x0 ; encoding: [0x02,0x00,0x04,0xf4,0x00,0x00,0x00,0xf8]
        // s_load_b128 s[0:3], s[4:5], 0x0 ; encoding: [0x02,0x00,0x08,0xf4,0x00,0x00,0x00,0xf8]
        // s_load_b256 s[0:7], s[4:5], 0x0 ; encoding: [0x02,0x00,0x0c,0xf4,0x00,0x00,0x00,0xf8]
        // s_load_b512 s[0:15], s[4:5], 0x0 ; encoding: [0x02,0x00,0x10,0xf4,0x00,0x00,0x00,0xf8]
        // (soffset defaults to NULL=124=0x7c when the source omits it, matching llvm-mc; see
        //  the trailing 0xf8 byte in word1 = 0b1111_1000 = SOFFSET(124)<<1 | high offset bit 0)
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 4, 0, None, false, false),
            [0x02, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB64, 0, 4, 0, None, false, false),
            [0x02, 0x00, 0x04, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB128, 0, 4, 0, None, false, false),
            [0x02, 0x00, 0x08, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB256, 0, 4, 0, None, false, false),
            [0x02, 0x00, 0x0c, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB512, 0, 4, 0, None, false, false),
            [0x02, 0x00, 0x10, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
    }

    #[test]
    fn smem_operand_fields() {
        // s_load_b32 s3, s[4:5], 0x0     ; encoding: [0xc2,0x00,0x00,0xf4,0x00,0x00,0x00,0xf8]
        // s_load_b32 s0, s[6:7], 0x0     ; encoding: [0x03,0x00,0x00,0xf4,0x00,0x00,0x00,0xf8]
        // s_load_b32 s0, s[4:5], 0x10    ; encoding: [0x02,0x00,0x00,0xf4,0x10,0x00,0x00,0xf8]
        // s_load_b32 s0, s[4:5], -0x4    ; encoding: [0x02,0x00,0x00,0xf4,0xfc,0xff,0x1f,0xf8]
        // s_load_b32 s0, s[4:5], s3      ; encoding: [0x02,0x00,0x00,0xf4,0x00,0x00,0x00,0x06]
        // s_load_b32 s0, s[4:5], m0      ; encoding: [0x02,0x00,0x00,0xf4,0x00,0x00,0x00,0xfa]
        // s_load_b32 s0, s[4:5], 0x0 glc dlc ; encoding: [0x02,0x60,0x00,0xf4,0x00,0x00,0x00,0xf8]
        assert_eq!(
            smem_load(SmemOp::LoadB32, 3, 4, 0, None, false, false),
            [0xc2, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 6, 0, None, false, false),
            [0x03, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 4, 0x10, None, false, false),
            [0x02, 0x00, 0x00, 0xf4, 0x10, 0x00, 0x00, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 4, -4, None, false, false),
            [0x02, 0x00, 0x00, 0xf4, 0xfc, 0xff, 0x1f, 0xf8]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 4, 0, Some(3), false, false),
            [0x02, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0x06]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 4, 0, Some(M0), false, false),
            [0x02, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0xfa]
        );
        assert_eq!(
            smem_load(SmemOp::LoadB32, 0, 4, 0, None, true, true),
            [0x02, 0x60, 0x00, 0xf4, 0x00, 0x00, 0x00, 0xf8]
        );
    }

    // ---- DS ----

    #[test]
    fn ds_load_opcodes() {
        // ds_load_b32 v0, v1              ; encoding: [0x00,0x00,0xd8,0xd8,0x01,0x00,0x00,0x00]
        // ds_load_b32 v0, v1 offset:4     ; encoding: [0x04,0x00,0xd8,0xd8,0x01,0x00,0x00,0x00]
        // ds_load_b32 v0, v1 offset:255   ; encoding: [0xff,0x00,0xd8,0xd8,0x01,0x00,0x00,0x00]
        // ds_load_b64 v[0:1], v2          ; encoding: [0x00,0x00,0xd8,0xd9,0x02,0x00,0x00,0x00]
        // ds_load_b128 v[0:3], v4         ; encoding: [0x00,0x00,0xfc,0xdb,0x04,0x00,0x00,0x00]
        // ds_load_u8 v0, v1               ; encoding: [0x00,0x00,0xe8,0xd8,0x01,0x00,0x00,0x00]
        // ds_load_i8 v0, v1               ; encoding: [0x00,0x00,0xe4,0xd8,0x01,0x00,0x00,0x00]
        // ds_load_u16 v0, v1              ; encoding: [0x00,0x00,0xf0,0xd8,0x01,0x00,0x00,0x00]
        // ds_load_i16 v0, v1              ; encoding: [0x00,0x00,0xec,0xd8,0x01,0x00,0x00,0x00]
        assert_eq!(
            ds_load(DsLoadOp::B32, 0, 1, 0),
            [0x00, 0x00, 0xd8, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::B32, 0, 1, 4),
            [0x04, 0x00, 0xd8, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::B32, 0, 1, 255),
            [0xff, 0x00, 0xd8, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::B64, 0, 2, 0),
            [0x00, 0x00, 0xd8, 0xd9, 0x02, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::B128, 0, 4, 0),
            [0x00, 0x00, 0xfc, 0xdb, 0x04, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::U8, 0, 1, 0),
            [0x00, 0x00, 0xe8, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::I8, 0, 1, 0),
            [0x00, 0x00, 0xe4, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::U16, 0, 1, 0),
            [0x00, 0x00, 0xf0, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_load(DsLoadOp::I16, 0, 1, 0),
            [0x00, 0x00, 0xec, 0xd8, 0x01, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn ds_store_opcodes() {
        // ds_store_b32 v1, v2            ; encoding: [0x00,0x00,0x34,0xd8,0x01,0x02,0x00,0x00]
        // ds_store_b32 v1, v2 offset:8   ; encoding: [0x08,0x00,0x34,0xd8,0x01,0x02,0x00,0x00]
        // ds_store_b64 v2, v[0:1]        ; encoding: [0x00,0x00,0x34,0xd9,0x02,0x00,0x00,0x00]
        // ds_store_b128 v4, v[0:3]       ; encoding: [0x00,0x00,0x7c,0xdb,0x04,0x00,0x00,0x00]
        // ds_store_b8 v1, v2             ; encoding: [0x00,0x00,0x78,0xd8,0x01,0x02,0x00,0x00]
        // ds_store_b16 v1, v2            ; encoding: [0x00,0x00,0x7c,0xd8,0x01,0x02,0x00,0x00]
        assert_eq!(
            ds_store(DsStoreOp::B32, 1, 2, 0),
            [0x00, 0x00, 0x34, 0xd8, 0x01, 0x02, 0x00, 0x00]
        );
        assert_eq!(
            ds_store(DsStoreOp::B32, 1, 2, 8),
            [0x08, 0x00, 0x34, 0xd8, 0x01, 0x02, 0x00, 0x00]
        );
        assert_eq!(
            ds_store(DsStoreOp::B64, 2, 0, 0),
            [0x00, 0x00, 0x34, 0xd9, 0x02, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_store(DsStoreOp::B128, 4, 0, 0),
            [0x00, 0x00, 0x7c, 0xdb, 0x04, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            ds_store(DsStoreOp::B8, 1, 2, 0),
            [0x00, 0x00, 0x78, 0xd8, 0x01, 0x02, 0x00, 0x00]
        );
        assert_eq!(
            ds_store(DsStoreOp::B16, 1, 2, 0),
            [0x00, 0x00, 0x7c, 0xd8, 0x01, 0x02, 0x00, 0x00]
        );
    }

    // ---- FLAT/GLOBAL ----

    #[test]
    fn global_load_opcodes() {
        // global_load_b32 v0, v[2:3], off       ; encoding: [0x00,0x00,0x52,0xdc,0x02,0x00,0x7c,0x00]
        // global_load_b32 v0, v[2:3], off offset:16 ; encoding: [0x10,0x00,0x52,0xdc,0x02,0x00,0x7c,0x00]
        // global_load_b64 v[0:1], v[2:3], off   ; encoding: [0x00,0x00,0x56,0xdc,0x02,0x00,0x7c,0x00]
        // global_load_b128 v[0:3], v[4:5], off  ; encoding: [0x00,0x00,0x5e,0xdc,0x04,0x00,0x7c,0x00]
        // global_load_u8 v0, v[2:3], off        ; encoding: [0x00,0x00,0x42,0xdc,0x02,0x00,0x7c,0x00]
        // global_load_i8 v0, v[2:3], off        ; encoding: [0x00,0x00,0x46,0xdc,0x02,0x00,0x7c,0x00]
        // global_load_u16 v0, v[2:3], off       ; encoding: [0x00,0x00,0x4a,0xdc,0x02,0x00,0x7c,0x00]
        // global_load_i16 v0, v[2:3], off       ; encoding: [0x00,0x00,0x4e,0xdc,0x02,0x00,0x7c,0x00]
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadB32, 0, 2, None, 0, false),
            [0x00, 0x00, 0x52, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadB32, 0, 2, None, 16, false),
            [0x10, 0x00, 0x52, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadB64, 0, 2, None, 0, false),
            [0x00, 0x00, 0x56, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadB128, 0, 4, None, 0, false),
            [0x00, 0x00, 0x5e, 0xdc, 0x04, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadU8, 0, 2, None, 0, false),
            [0x00, 0x00, 0x42, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadI8, 0, 2, None, 0, false),
            [0x00, 0x00, 0x46, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadU16, 0, 2, None, 0, false),
            [0x00, 0x00, 0x4a, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_load(Seg::Global, FlatOp::LoadI16, 0, 2, None, 0, false),
            [0x00, 0x00, 0x4e, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
    }

    #[test]
    fn global_store_opcodes() {
        // global_store_b32 v[2:3], v0, off      ; encoding: [0x00,0x00,0x6a,0xdc,0x02,0x00,0x7c,0x00]
        // global_store_b64 v[2:3], v[0:1], off  ; encoding: [0x00,0x00,0x6e,0xdc,0x02,0x00,0x7c,0x00]
        // global_store_b128 v[4:5], v[0:3], off ; encoding: [0x00,0x00,0x76,0xdc,0x04,0x00,0x7c,0x00]
        // global_store_b8 v[2:3], v0, off       ; encoding: [0x00,0x00,0x62,0xdc,0x02,0x00,0x7c,0x00]
        // global_store_b16 v[2:3], v0, off      ; encoding: [0x00,0x00,0x66,0xdc,0x02,0x00,0x7c,0x00]
        assert_eq!(
            flat_store(Seg::Global, FlatOp::StoreB32, 2, 0, None, 0, false),
            [0x00, 0x00, 0x6a, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_store(Seg::Global, FlatOp::StoreB64, 2, 0, None, 0, false),
            [0x00, 0x00, 0x6e, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_store(Seg::Global, FlatOp::StoreB128, 4, 0, None, 0, false),
            [0x00, 0x00, 0x76, 0xdc, 0x04, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_store(Seg::Global, FlatOp::StoreB8, 2, 0, None, 0, false),
            [0x00, 0x00, 0x62, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_store(Seg::Global, FlatOp::StoreB16, 2, 0, None, 0, false),
            [0x00, 0x00, 0x66, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
    }

    #[test]
    fn global_atomic_opcodes() {
        // global_atomic_add_u32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xd6,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_add_u32 v0, v[2:3], v1, off glc ; encoding: [0x00,0x40,0xd6,0xdc,0x02,0x01,0x7c,0x00]
        // global_atomic_sub_u32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xda,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_swap_b32 v[2:3], v0, off       ; encoding: [0x00,0x00,0xce,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_cmpswap_b32 v[2:3], v[0:1], off ; encoding: [0x00,0x00,0xd2,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_min_i32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xe2,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_max_i32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xea,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_min_u32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xe6,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_max_u32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xee,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_and_b32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xf2,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_or_b32 v[2:3], v0, off         ; encoding: [0x00,0x00,0xf6,0xdc,0x02,0x00,0x7c,0x00]
        // global_atomic_xor_b32 v[2:3], v0, off        ; encoding: [0x00,0x00,0xfa,0xdc,0x02,0x00,0x7c,0x00]
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicAddU32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xd6, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicAddU32, Some(0), 2, 1, None, 0),
            [0x00, 0x40, 0xd6, 0xdc, 0x02, 0x01, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicSubU32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xda, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicSwapB32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xce, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicCmpswapB32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xd2, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicSminI32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xe2, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicSmaxI32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xea, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicUminU32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xe6, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicUmaxU32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xee, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicAndB32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xf2, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicOrB32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xf6, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        assert_eq!(
            flat_atomic(Seg::Global, FlatOp::AtomicXorB32, None, 2, 0, None, 0),
            [0x00, 0x00, 0xfa, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
    }

    #[test]
    fn flat_scratch_seg_field() {
        // flat_load_b32 v0, v[2:3]      ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x7c,0x00]
        // scratch_load_b32 v0, v2, off  ; encoding: [0x00,0x00,0x51,0xdc,0x02,0x00,0xfc,0x00]
        assert_eq!(
            flat_load(Seg::Flat, FlatOp::LoadB32, 0, 2, None, 0, false),
            [0x00, 0x00, 0x50, 0xdc, 0x02, 0x00, 0x7c, 0x00]
        );
        // `scratch_load_b32 v0, v2, off` addresses via a plain VGPR offset with no SGPR base
        // either (SADDR field also NULL, hence 0xfc = NULL<<1 rather than 0x7c): confirms the
        // SEG field alone (bits 17:16 = 1) distinguishes `scratch` from `flat`/`global` here.
        assert_eq!(
            flat_load(Seg::Scratch, FlatOp::LoadB32, 0, 2, None, 0, false).as_slice()[0..4],
            [0x00, 0x00, 0x51, 0xdc]
        );
    }
}
