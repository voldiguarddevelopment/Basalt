// Hand-rolled RV32IM instruction encoder. Same identity as `basalt-x86/src/enc.rs`: every
// function here builds a 32-bit instruction word directly from the RISC-V ISA manual's own
// bit-field layout (R/I/S/B/U/J formats) — nothing is derived by pattern-matching another
// encoder's output. Every opcode/funct3/funct7 value below was cross-checked byte-for-byte
// against a real assembler (`riscv32-elf-as -march=rv32im -mabi=ilp32`, GNU binutils 2.45)
// during development: a `.s` file exercising one instruction per RV32I/M format was hand-
// assembled and objdumped, and the resulting words are hard-coded into this file's own
// `tests` module below. This is deliberately not a general assembler — only the instruction
// shapes `lower.rs`/`softfloat.rs` actually need are implemented, and no compressed (RVC)
// forms are ever emitted (every instruction here is a fixed 4 bytes).
//
// # Register numbering
//
// `x0`-`x31` exactly as the ISA numbers them; `x0` is hardwired zero. The named constants
// below are the standard ABI mnemonics (`ra`, `sp`, `a0`-`a7`, `t0`-`t6`) purely for
// readability — nothing here enforces the calling-convention *meaning` of a register, that
// is entirely `lower.rs`'s concern.
//
// # Why every branch is a two-instruction "invert-and-jal" sequence
//
// RV32's conditional branches (`beq`/`bne`/`blt`/`bge`/`bltu`/`bgeu`) carry a 13-bit signed
// *byte* offset (B-format), i.e. only ±4KiB reach — nowhere near enough to guarantee reaching
// an arbitrary label in a function with many basic blocks or a large frame. Rather than a
// two-pass branch-relaxation scheme (measuring whether the real target is in range and only
// then choosing short-vs-long form — exactly the kind of "clever" size-based branching this
// project's oracle-style backends deliberately avoid, see `basalt-x86/src/enc.rs`'s own
// disp32-always policy), every conditional branch in this encoder unconditionally expands to
// two instructions: the opposite-condition branch jumping over a single `jal`, followed by an
// unconditional `jal x0, target` (`jal`'s J-format immediate reaches ±1MiB, comfortably enough
// for any kernel this backend targets). One code path, no size-based branching, exactly
// mirroring the x86 encoder's own stated policy.
//
// # Label/fixup mechanism
//
// Identical shape to `basalt-x86/src/enc.rs`'s `Enc`: `label` records the current byte offset
// under a name, every forward/backward reference records a fixup, `finish` resolves every
// fixup once all labels are known. Unlike x86's rel32 patch (a flat 4-byte write), a resolved
// RV32 branch/jump's immediate bits are scattered through the instruction word alongside
// opcode/rd/funct3 fields that are already correct — so a fixup here stores enough
// information (`rd`) to *re-encode the whole word* at resolution time rather than patch a
// sub-field, which is simpler and cannot accidentally corrupt an adjacent field.

/// Zero register — reads as 0, writes are discarded.
pub const ZERO: u8 = 0;
/// Return address — holds the caller's continuation across a `jal`/`jalr` call, and this
/// backend's own incoming return address across any internal call it makes (see `lower.rs`'s
/// `ra_home`).
pub const RA: u8 = 1;
/// Stack pointer.
pub const SP: u8 = 2;
pub const T0: u8 = 5;
pub const T1: u8 = 6;
pub const T2: u8 = 7;
pub const T3: u8 = 28;
pub const T4: u8 = 29;
pub const T5: u8 = 30;
pub const T6: u8 = 31;
/// Integer-class argument/return registers, in passing order (RV32 `ilp32` soft-float ABI:
/// every argument is integer-class, there is no separate float register file).
pub const A0: u8 = 10;
pub const A1: u8 = 11;
pub const A2: u8 = 12;
pub const A3: u8 = 13;
pub const A4: u8 = 14;
pub const A5: u8 = 15;
pub const A6: u8 = 16;
pub const A7: u8 = 17;
/// Argument/return registers, in passing order, for `lower.rs`'s `classify_params`.
pub const ARG_REGS: [u8; 8] = [A0, A1, A2, A3, A4, A5, A6, A7];
/// Callee-saved registers, purely as extra scratch names for `softfloat.rs` (every routine
/// there is a leaf that never actually preserves anything across a call, see that file's own
/// header — these are used here only for their names, not their ABI meaning).
pub const S2: u8 = 18;
pub const S3: u8 = 19;
pub const S4: u8 = 20;
pub const S5: u8 = 21;
pub const S6: u8 = 22;
pub const S7: u8 = 23;
pub const S8: u8 = 24;
pub const S9: u8 = 25;
pub const S10: u8 = 26;
pub const S11: u8 = 27;

const OP: u32 = 0b0110011;
const OP_IMM: u32 = 0b0010011;
const LOAD: u32 = 0b0000011;
const STORE: u32 = 0b0100011;
const BRANCH: u32 = 0b1100011;
const JALR: u32 = 0b1100111;
const JAL: u32 = 0b1101111;
const LUI: u32 = 0b0110111;
const AUIPC: u32 = 0b0010111;

fn r_type(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// `imm` is sign-extended into the 12-bit field verbatim (the caller is responsible for it
/// fitting in `-2048..=2047`; every caller in this crate only ever passes values already
/// known to fit — see `lower.rs`'s `li32`-based addressing policy, which never emits a
/// direct 12-bit-immediate memory access at all).
fn i_type(imm: i32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (((imm as u32) & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn s_type(imm: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    let lo = imm & 0x1F;
    let hi = (imm >> 5) & 0x7F;
    (hi << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | (lo << 7) | opcode
}

/// `imm` must be even (branch targets are always 2-byte aligned) and fit in `-4096..=4094`.
fn b_type(imm: i32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    let bit12 = (imm >> 12) & 1;
    let bit11 = (imm >> 11) & 1;
    let bits10_5 = (imm >> 5) & 0x3F;
    let bits4_1 = (imm >> 1) & 0xF;
    (bit12 << 31)
        | (bits10_5 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (bits4_1 << 8)
        | (bit11 << 7)
        | opcode
}

/// `imm20` is the already-shifted-out upper 20 bits (i.e. the value `lui` loads into bits
/// 31:12 of the destination register), not a full 32-bit immediate.
fn u_type(imm20: u32, rd: u8, opcode: u32) -> u32 {
    ((imm20 & 0xFFFFF) << 12) | ((rd as u32) << 7) | opcode
}

/// `imm` must be even and fit in `-1048576..=1048574` (±1MiB, `jal`'s full reach).
fn j_type(imm: i32, rd: u8, opcode: u32) -> u32 {
    let imm = imm as u32;
    let bit20 = (imm >> 20) & 1;
    let bits10_1 = (imm >> 1) & 0x3FF;
    let bit11 = (imm >> 11) & 1;
    let bits19_12 = (imm >> 12) & 0xFF;
    (bit20 << 31)
        | (bits10_1 << 21)
        | (bit11 << 20)
        | (bits19_12 << 12)
        | ((rd as u32) << 7)
        | opcode
}

#[derive(Clone, Copy)]
pub enum AluOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Sll,
    Srl,
    Sra,
    Slt,
    Sltu,
}

#[derive(Clone, Copy)]
pub enum MulOp {
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
}

/// The six RV32 conditional-branch comparisons.
#[derive(Clone, Copy)]
pub enum BCond {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

impl BCond {
    fn funct3(self) -> u32 {
        match self {
            BCond::Eq => 0b000,
            BCond::Ne => 0b001,
            BCond::Lt => 0b100,
            BCond::Ge => 0b101,
            BCond::Ltu => 0b110,
            BCond::Geu => 0b111,
        }
    }

    /// The negated condition — used by `Enc::branch` to build the "skip the jal unless the
    /// real condition holds" guard (see the module header).
    fn invert(self) -> BCond {
        match self {
            BCond::Eq => BCond::Ne,
            BCond::Ne => BCond::Eq,
            BCond::Lt => BCond::Ge,
            BCond::Ge => BCond::Lt,
            BCond::Ltu => BCond::Geu,
            BCond::Geu => BCond::Ltu,
        }
    }
}

/// A pending `jal rd, <label>` whose target is not yet known. Resolved by re-encoding the
/// whole instruction word once every label has been recorded (see the module header).
struct Fixup {
    pos: usize,
    label: String,
    rd: u8,
}

pub struct Enc {
    code: Vec<u8>,
    labels: std::collections::HashMap<String, usize>,
    fixups: Vec<Fixup>,
}

impl Enc {
    pub fn new() -> Enc {
        Enc {
            code: Vec::new(),
            labels: std::collections::HashMap::new(),
            fixups: Vec::new(),
        }
    }

    pub fn pos(&self) -> usize {
        self.code.len()
    }

    pub fn label(&mut self, name: &str) {
        let prev = self.labels.insert(name.to_string(), self.code.len());
        debug_assert!(prev.is_none(), "label defined twice: {name}");
    }

    fn push(&mut self, word: u32) {
        self.code.extend_from_slice(&word.to_le_bytes());
    }

    pub fn alu_reg(&mut self, op: AluOp, rd: u8, rs1: u8, rs2: u8) {
        let (funct7, funct3) = match op {
            AluOp::Add => (0b0000000, 0b000),
            AluOp::Sub => (0b0100000, 0b000),
            AluOp::Sll => (0b0000000, 0b001),
            AluOp::Slt => (0b0000000, 0b010),
            AluOp::Sltu => (0b0000000, 0b011),
            AluOp::Xor => (0b0000000, 0b100),
            AluOp::Srl => (0b0000000, 0b101),
            AluOp::Sra => (0b0100000, 0b101),
            AluOp::Or => (0b0000000, 0b110),
            AluOp::And => (0b0000000, 0b111),
        };
        self.push(r_type(funct7, rs2, rs1, funct3, rd, OP));
    }

    pub fn mul_reg(&mut self, op: MulOp, rd: u8, rs1: u8, rs2: u8) {
        let funct3 = match op {
            MulOp::Mul => 0b000,
            MulOp::Mulh => 0b001,
            MulOp::Mulhsu => 0b010,
            MulOp::Mulhu => 0b011,
            MulOp::Div => 0b100,
            MulOp::Divu => 0b101,
            MulOp::Rem => 0b110,
            MulOp::Remu => 0b111,
        };
        self.push(r_type(0b0000001, rs2, rs1, funct3, rd, OP));
    }

    /// `imm` must fit `-2048..=2047` — every caller in this crate only ever passes a small
    /// compile-time-known constant (loop increments, the `+8` branch-skip offset, `li32`'s
    /// own low-12-bits split); frame-relative addressing never goes through this directly
    /// (see the module header and `lower.rs`'s `frame_addr`).
    pub fn addi(&mut self, rd: u8, rs1: u8, imm: i32) {
        debug_assert!((-2048..=2047).contains(&imm));
        self.push(i_type(imm, rs1, 0b000, rd, OP_IMM));
    }

    pub fn slti(&mut self, rd: u8, rs1: u8, imm: i32) {
        self.push(i_type(imm, rs1, 0b010, rd, OP_IMM));
    }

    pub fn sltiu(&mut self, rd: u8, rs1: u8, imm: i32) {
        self.push(i_type(imm, rs1, 0b011, rd, OP_IMM));
    }

    pub fn xori(&mut self, rd: u8, rs1: u8, imm: i32) {
        self.push(i_type(imm, rs1, 0b100, rd, OP_IMM));
    }

    pub fn ori(&mut self, rd: u8, rs1: u8, imm: i32) {
        self.push(i_type(imm, rs1, 0b110, rd, OP_IMM));
    }

    pub fn andi(&mut self, rd: u8, rs1: u8, imm: i32) {
        self.push(i_type(imm, rs1, 0b111, rd, OP_IMM));
    }

    pub fn slli(&mut self, rd: u8, rs1: u8, shamt: u32) {
        debug_assert!(shamt < 32);
        self.push(i_type(shamt as i32, rs1, 0b001, rd, OP_IMM));
    }

    pub fn srli(&mut self, rd: u8, rs1: u8, shamt: u32) {
        debug_assert!(shamt < 32);
        self.push(i_type(shamt as i32, rs1, 0b101, rd, OP_IMM));
    }

    pub fn srai(&mut self, rd: u8, rs1: u8, shamt: u32) {
        debug_assert!(shamt < 32);
        self.push(i_type(
            (shamt | (0b0100000 << 5)) as i32,
            rs1,
            0b101,
            rd,
            OP_IMM,
        ));
    }

    /// `mv rd, rs` (`addi rd, rs, 0`).
    pub fn mv(&mut self, rd: u8, rs: u8) {
        self.addi(rd, rs, 0);
    }

    pub fn nop(&mut self) {
        self.addi(ZERO, ZERO, 0);
    }

    pub fn lb(&mut self, rd: u8, offset: i32, base: u8) {
        self.push(i_type(offset, base, 0b000, rd, LOAD));
    }
    pub fn lh(&mut self, rd: u8, offset: i32, base: u8) {
        self.push(i_type(offset, base, 0b001, rd, LOAD));
    }
    pub fn lw(&mut self, rd: u8, offset: i32, base: u8) {
        self.push(i_type(offset, base, 0b010, rd, LOAD));
    }
    pub fn lbu(&mut self, rd: u8, offset: i32, base: u8) {
        self.push(i_type(offset, base, 0b100, rd, LOAD));
    }
    pub fn lhu(&mut self, rd: u8, offset: i32, base: u8) {
        self.push(i_type(offset, base, 0b101, rd, LOAD));
    }

    pub fn sb(&mut self, rs2: u8, offset: i32, base: u8) {
        self.push(s_type(offset, rs2, base, 0b000, STORE));
    }
    pub fn sh(&mut self, rs2: u8, offset: i32, base: u8) {
        self.push(s_type(offset, rs2, base, 0b001, STORE));
    }
    pub fn sw(&mut self, rs2: u8, offset: i32, base: u8) {
        self.push(s_type(offset, rs2, base, 0b010, STORE));
    }

    pub fn lui(&mut self, rd: u8, imm20: u32) {
        self.push(u_type(imm20, rd, LUI));
    }
    pub fn auipc(&mut self, rd: u8, imm20: u32) {
        self.push(u_type(imm20, rd, AUIPC));
    }

    /// `jalr rd, offset(rs1)` — direct form (compile-time-known `offset`), used only for
    /// `ret` (`jalr x0, 0(ra)`) in this crate; every call site uses `jal` (see `call`/`jump`
    /// below), never `jalr`, since every call target is a same-object label.
    pub fn jalr(&mut self, rd: u8, rs1: u8, offset: i32) {
        self.push(i_type(offset, rs1, 0b000, rd, JALR));
    }

    pub fn ret(&mut self) {
        self.jalr(ZERO, RA, 0);
    }

    /// Records a `jal rd, <target_label>` to be resolved once every label is known (see the
    /// module header). A zeroed placeholder word is emitted now.
    fn jal_fixup(&mut self, rd: u8, target_label: &str) {
        let pos = self.code.len();
        self.push(0);
        self.fixups.push(Fixup {
            pos,
            label: target_label.to_string(),
            rd,
        });
    }

    /// `jal ra, <label>` — an internal call to another label in this same object (a
    /// soft-float routine; see `softfloat.rs`). Every routine this crate calls is a leaf (see
    /// `softfloat.rs`'s own header), so nothing here needs to save/restore anything around
    /// the call site itself — only `lower.rs`'s function prologue/epilogue needs to preserve
    /// its *own* incoming `ra` across making any call at all (`Frame::ra_home`).
    pub fn call(&mut self, target_label: &str) {
        self.jal_fixup(RA, target_label);
    }

    /// `jal x0, <label>` — unconditional jump.
    pub fn jump(&mut self, target_label: &str) {
        self.jal_fixup(ZERO, target_label);
    }

    /// `beq`/`bne`/etc to a label: always the two-instruction invert-and-jal expansion (see
    /// the module header) so branch reach is never a size-dependent concern.
    pub fn branch(&mut self, cond: BCond, rs1: u8, rs2: u8, target_label: &str) {
        // Skip over exactly one instruction (the `jal` below, 4 bytes) when the real
        // condition does *not* hold.
        self.push(b_type(8, rs2, rs1, cond.invert().funct3(), BRANCH));
        self.jump(target_label);
    }

    /// Loads an arbitrary 32-bit constant into `rd`: `lui`+`addi` in the general case, or a
    /// single `addi x0, imm` when `imm` already fits a 12-bit signed immediate outright (the
    /// standard RISC-V `li` splitting trick — see the module header's arithmetic-only
    /// self-check in this file's `tests` module for why the split is exact for every `i32`).
    pub fn li32(&mut self, rd: u8, imm: i32) {
        let imm_u = imm as u32;
        let upper20 = (imm_u.wrapping_add(0x800) >> 12) & 0xFFFFF;
        let lower = imm.wrapping_sub((upper20 as i32) << 12);
        if upper20 == 0 {
            self.addi(rd, ZERO, lower);
        } else {
            self.lui(rd, upper20);
            if lower != 0 {
                self.addi(rd, rd, lower);
            }
        }
    }

    /// Resolves every recorded `jal` fixup against the labels known by the time this is
    /// called (an unresolved label is a codegen bug, not a user-facing error — hence the
    /// panic, matching `basalt-x86/src/enc.rs`'s own `finish`) and returns the finished byte
    /// buffer.
    pub fn finish(self) -> Vec<u8> {
        let Enc {
            mut code,
            labels,
            fixups,
        } = self;
        for fx in &fixups {
            let target = *labels
                .get(&fx.label)
                .unwrap_or_else(|| panic!("codegen bug: undefined label `{}`", fx.label));
            let rel = target as i64 - fx.pos as i64;
            let rel = i32::try_from(rel)
                .unwrap_or_else(|_| panic!("codegen bug: jal to `{}` out of range", fx.label));
            debug_assert!(rel % 2 == 0, "jal target must be 2-byte aligned");
            let word = j_type(rel, fx.rd, JAL);
            code[fx.pos..fx.pos + 4].copy_from_slice(&word.to_le_bytes());
        }
        code
    }
}

impl Default for Enc {
    fn default() -> Enc {
        Enc::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected word below was produced by assembling the corresponding line with a
    // real assembler (`riscv32-elf-as -march=rv32im -mabi=ilp32`, GNU binutils 2.45) and
    // reading back the little-endian instruction word with `riscv32-elf-objdump`/`od`. See
    // this crate's own design notes (not checked in verbatim, per project convention of
    // keeping derivations out of comments once verified) for the exact session; the
    // assembly source below reproduces precisely what was assembled.

    #[test]
    fn r_type_base_and_m_extension() {
        // add x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0, 1, OP), 0x003100b3);
        // sub x1, x2, x3
        assert_eq!(r_type(0b0100000, 3, 2, 0, 1, OP), 0x403100b3);
        // sll x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b001, 1, OP), 0x003110b3);
        // slt x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b010, 1, OP), 0x003120b3);
        // sltu x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b011, 1, OP), 0x003130b3);
        // xor x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b100, 1, OP), 0x003140b3);
        // srl x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b101, 1, OP), 0x003150b3);
        // sra x1, x2, x3
        assert_eq!(r_type(0b0100000, 3, 2, 0b101, 1, OP), 0x403150b3);
        // or x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b110, 1, OP), 0x003160b3);
        // and x1, x2, x3
        assert_eq!(r_type(0, 3, 2, 0b111, 1, OP), 0x003170b3);
        // mul x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b000, 1, OP), 0x023100b3);
        // mulh x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b001, 1, OP), 0x023110b3);
        // mulhsu x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b010, 1, OP), 0x023120b3);
        // mulhu x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b011, 1, OP), 0x023130b3);
        // div x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b100, 1, OP), 0x023140b3);
        // divu x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b101, 1, OP), 0x023150b3);
        // rem x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b110, 1, OP), 0x023160b3);
        // remu x1, x2, x3
        assert_eq!(r_type(0b0000001, 3, 2, 0b111, 1, OP), 0x023170b3);
    }

    #[test]
    fn alu_reg_and_mul_reg_match_verified_words() {
        let mut e = Enc::new();
        e.alu_reg(AluOp::Add, 1, 2, 3);
        e.alu_reg(AluOp::Sub, 1, 2, 3);
        e.alu_reg(AluOp::Sll, 1, 2, 3);
        e.alu_reg(AluOp::Slt, 1, 2, 3);
        e.alu_reg(AluOp::Sltu, 1, 2, 3);
        e.alu_reg(AluOp::Xor, 1, 2, 3);
        e.alu_reg(AluOp::Srl, 1, 2, 3);
        e.alu_reg(AluOp::Sra, 1, 2, 3);
        e.alu_reg(AluOp::Or, 1, 2, 3);
        e.alu_reg(AluOp::And, 1, 2, 3);
        e.mul_reg(MulOp::Mul, 1, 2, 3);
        e.mul_reg(MulOp::Mulh, 1, 2, 3);
        e.mul_reg(MulOp::Mulhsu, 1, 2, 3);
        e.mul_reg(MulOp::Mulhu, 1, 2, 3);
        e.mul_reg(MulOp::Div, 1, 2, 3);
        e.mul_reg(MulOp::Divu, 1, 2, 3);
        e.mul_reg(MulOp::Rem, 1, 2, 3);
        e.mul_reg(MulOp::Remu, 1, 2, 3);
        let bytes = e.finish();
        let expect: [u32; 18] = [
            0x003100b3, 0x403100b3, 0x003110b3, 0x003120b3, 0x003130b3, 0x003140b3, 0x003150b3,
            0x403150b3, 0x003160b3, 0x003170b3, 0x023100b3, 0x023110b3, 0x023120b3, 0x023130b3,
            0x023140b3, 0x023150b3, 0x023160b3, 0x023170b3,
        ];
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words, expect);
    }

    #[test]
    fn i_type_alu_and_shifts_match_verified_words() {
        let mut e = Enc::new();
        e.addi(5, 6, 100); // addi x5, x6, 100 -> 06430293
        e.addi(5, 6, -100); // addi x5, x6, -100 -> f9c30293
        e.slti(5, 6, 100); // 06432293
        e.sltiu(5, 6, 100); // 06433293
        e.xori(5, 6, 100); // 06434293
        e.ori(5, 6, 100); // 06436293
        e.andi(5, 6, 100); // 06437293
        e.slli(5, 6, 7); // 00731293
        e.srli(5, 6, 7); // 00735293
        e.srai(5, 6, 7); // 40735293
        let bytes = e.finish();
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            vec![
                0x06430293, 0xf9c30293, 0x06432293, 0x06433293, 0x06434293, 0x06436293, 0x06437293,
                0x00731293, 0x00735293, 0x40735293,
            ]
        );
    }

    #[test]
    fn loads_stores_jalr_match_verified_words() {
        let mut e = Enc::new();
        e.lb(5, 16, 6); // 01030283
        e.lh(5, 16, 6); // 01031283
        e.lw(5, 16, 6); // 01032283
        e.lbu(5, 16, 6); // 01034283
        e.lhu(5, 16, 6); // 01035283
        e.jalr(1, 2, 4); // 004100e7
        e.sb(5, 16, 6); // 00530823
        e.sh(5, 16, 6); // 00531823
        e.sw(5, 16, 6); // 00532823
        let bytes = e.finish();
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            vec![
                0x01030283, 0x01031283, 0x01032283, 0x01034283, 0x01035283, 0x004100e7, 0x00530823,
                0x00531823, 0x00532823,
            ]
        );
    }

    #[test]
    fn branches_lui_auipc_jal_match_verified_words() {
        // From the same verification session as the other tests in this file: six branches
        // at addresses 0x94/0x98/0x9c/0xa0/0xa4/0xa8, all targeting a label at 0xb8, followed
        // by `lui x5,100`, `auipc x5,100`, then `jal x1,<lbl>` at 0xb4 (imm=4).
        assert_eq!(b_type(0xb8 - 0x94, 6, 5, 0b000, BRANCH), 0x02628263); // beq
        assert_eq!(b_type(0xb8 - 0x98, 6, 5, 0b001, BRANCH), 0x02629063); // bne
        assert_eq!(b_type(0xb8 - 0x9c, 6, 5, 0b100, BRANCH), 0x0062ce63); // blt
        assert_eq!(b_type(0xb8 - 0xa0, 6, 5, 0b101, BRANCH), 0x0062dc63); // bge
        assert_eq!(b_type(0xb8 - 0xa4, 6, 5, 0b110, BRANCH), 0x0062ea63); // bltu
        assert_eq!(b_type(0xb8 - 0xa8, 6, 5, 0b111, BRANCH), 0x0062f863); // bgeu
        assert_eq!(u_type(100, 5, LUI), 0x000642b7);
        assert_eq!(u_type(100, 5, AUIPC), 0x00064297);
        assert_eq!(j_type(0xb8 - 0xb4, 1, JAL), 0x004000ef);
    }

    /// Pure arithmetic self-check of `li32`'s upper/lower split (see the module header):
    /// for every probed `i32`, `(upper20 << 12).wrapping_add(sign_extend_12(lower)) == imm`.
    /// This does not execute any RV32 instruction — it only re-derives, in ordinary Rust
    /// 32-bit wrapping arithmetic (which is what `lui`+`addi` are defined to compute), that
    /// the split this crate emits is exact.
    #[test]
    fn li32_split_is_exact() {
        let probes: [i32; 9] = [0, 1, -1, 42, -42, 0x7ff, 0x800, 0x12345678, i32::MIN];
        for &imm in &probes {
            let imm_u = imm as u32;
            let upper20 = (imm_u.wrapping_add(0x800) >> 12) & 0xFFFFF;
            let lower = imm.wrapping_sub((upper20 as i32) << 12);
            assert!(
                (-2048..=2047).contains(&lower),
                "lower out of 12-bit range for {imm}: {lower}"
            );
            let reconstructed = ((upper20 as i32) << 12).wrapping_add(lower);
            assert_eq!(reconstructed, imm, "split did not reconstruct {imm}");
        }
    }

    #[test]
    fn branch_expands_to_invert_and_jal() {
        let mut e = Enc::new();
        e.label("start");
        e.branch(BCond::Eq, 5, 6, "target");
        e.label("target");
        let bytes = e.finish();
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // bne x5, x6, +8 (skip the jal when NOT equal)
        assert_eq!(words[0], b_type(8, 6, 5, BCond::Ne.funct3(), BRANCH));
        // jal x0, +4 (target is the very next instruction here)
        assert_eq!(words[1], j_type(4, ZERO, JAL));
    }

    #[test]
    fn call_and_ret_resolve() {
        let mut e = Enc::new();
        e.call("routine");
        e.label("main_tail");
        e.jump("done");
        e.label("routine");
        e.ret();
        e.label("done");
        let bytes = e.finish();
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words[0], j_type(8, RA, JAL)); // call routine: routine is at pos 8
        assert_eq!(words[1], j_type(8, ZERO, JAL)); // jump done: done is at pos 12, jump at pos 4
        assert_eq!(words[2], i_type(0, RA, 0b000, ZERO, JALR)); // ret
    }
}
