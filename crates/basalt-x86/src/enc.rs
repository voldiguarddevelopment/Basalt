// Hand-rolled x86-64 instruction encoder. No external assembler/encoder crate — this is the
// project's identity (see the workspace-level rationale). Deliberately covers only the
// instruction shapes the oracle backend actually needs; it is not a general-purpose x86
// assembler and never grows opcodes speculatively.
//
// Every encoding helper here builds bytes directly from the Intel manual's own encoding
// tables (REX prefix, ModRM, SIB, displacement, immediate) — nothing is derived by pattern
// matching against another encoder's output.
//
// # Register numbering
//
// GP registers are numbered 0-15 exactly as the ISA does (rax=0 .. r15=15); the low 3 bits go
// in ModRM/opcode-reg fields, bit 3 becomes a REX extension bit. XMM registers are numbered
// 0-7 here since the oracle never touches xmm8-15.
//
// # Addressing modes actually used
//
// Only two, by construction (see `oracle.rs`'s frame design): `[rbp + disp32]` (a stack
// slot; disp32 is always emitted in full, even for small offsets, rather than picking the
// shorter disp8 form when it would fit — one code path, no size-based branching) and
// `[reg]` with zero displacement, dereferencing a real pointer value already sitting in a
// scratch register. The second form's base register is always chosen from a fixed set that
// excludes rsp/rbp/r12/r13, so a SIB byte and the mod=00/rm=101
// RIP-relative special case never come up — this code has no SIB support at all.
//
// # REX policy
//
// A REX prefix is emitted on every GP instruction, even when none of its bits are actually
// needed. This is deliberate, not an oversight: an 8-bit register operand encoded with
// register index 4-7 means AH/CH/DH/BH with no REX prefix present, but SPL/BPL/SIL/DIL with
// any REX prefix present (even `0x40`, all-zero bits). Always emitting REX keeps every
// register (including rsi/rdi, which double as incoming SysV argument registers) usable
// uniformly as an 8-bit operand without a special case for "does this register need REX to
// mean what I want."

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;

/// SysV integer-class argument registers, in passing order.
pub const INT_ARG_REGS: [u8; 6] = [RDI, RSI, RDX, RCX, R8, R9];
/// SSE-class (float/double) argument registers, in passing order.
pub const SSE_ARG_REGS: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// A GP or memory operand shape. Every instruction below reduces to one opcode plus a
/// `(reg-field, Rm)` pair, matching the ModRM byte's own (reg, mod:rm) split.
#[derive(Clone, Copy)]
pub enum Rm {
    /// Register-direct: ModRM mod=11.
    Direct(u8),
    /// `[base]`, no displacement: ModRM mod=00. `base` must not be rsp/rbp/r12/r13.
    IndBase(u8),
    /// `[rbp + disp32]`: ModRM mod=10, rm=101(rbp), disp32 always present.
    RbpDisp(i32),
}

/// Operand width for a GP instruction. Selects the mandatory 0x66 prefix (16-bit) or REX.W
/// (64-bit) and, for mov/alu forms that have a distinct 8-bit opcode, that opcode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum W {
    B1,
    B2,
    B4,
    B8,
}

fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    (mod_bits << 6) | ((reg & 7) << 3) | (rm & 7)
}

fn rm_base_reg(rm: Rm) -> u8 {
    match rm {
        Rm::Direct(r) => r,
        Rm::IndBase(r) => r,
        Rm::RbpDisp(_) => RBP,
    }
}

fn rex_byte(w: bool, reg: u8, rm: Rm) -> u8 {
    let base = rm_base_reg(rm);
    let r_bit = (reg >> 3) & 1;
    let b_bit = (base >> 3) & 1;
    0x40 | ((w as u8) << 3) | (r_bit << 2) | b_bit
}

fn push_modrm(code: &mut Vec<u8>, reg: u8, rm: Rm) {
    match rm {
        Rm::Direct(r) => code.push(modrm(0b11, reg, r)),
        Rm::IndBase(r) => code.push(modrm(0b00, reg, r)),
        Rm::RbpDisp(d) => {
            code.push(modrm(0b10, reg, RBP));
            code.extend_from_slice(&d.to_le_bytes());
        }
    }
}

/// Growable machine-code buffer with a two-pass label/fixup mechanism for local branches:
/// `label` records the current offset under a name; `jump`/`jcc` (via `rel32_fixup`) emit a
/// zeroed placeholder and record where to patch it; `finish` resolves every fixup once all
/// labels are known. This is the only "layout" logic in the whole encoder — no relocations
/// escape into the emitted object, every branch here is function-local.
pub struct Enc {
    code: Vec<u8>,
    labels: std::collections::HashMap<String, usize>,
    fixups: Vec<(usize, String)>,
}

impl Enc {
    pub fn new() -> Enc {
        Enc {
            code: Vec::new(),
            labels: std::collections::HashMap::new(),
            fixups: Vec::new(),
        }
    }

    pub fn label(&mut self, name: &str) {
        let prev = self.labels.insert(name.to_string(), self.code.len());
        debug_assert!(prev.is_none(), "label defined twice: {name}");
    }

    fn insn(&mut self, prefix: Option<u8>, rex_w: bool, opcode: &[u8], reg: u8, rm: Rm) {
        if let Some(p) = prefix {
            self.code.push(p);
        }
        self.code.push(rex_byte(rex_w, reg, rm));
        self.code.extend_from_slice(opcode);
        push_modrm(&mut self.code, reg, rm);
    }

    /// Records a rel32 fixup at the current position (4 zero bytes are emitted now; `finish`
    /// patches them once every label is known).
    fn rel32_fixup(&mut self, target_label: &str) {
        let pos = self.code.len();
        self.code.extend_from_slice(&[0, 0, 0, 0]);
        self.fixups.push((pos, target_label.to_string()));
    }

    pub fn jmp(&mut self, target_label: &str) {
        self.code.push(0xE9);
        self.rel32_fixup(target_label);
    }

    /// `cc` is the condition-code nibble as used by both Jcc (`0F 80+cc`) and SETcc
    /// (`0F 90+cc`), e.g. 4=E/Z, 5=NE/NZ, 0xC=L, 0xD=GE, 0xE=LE, 0xF=G, 2=B/C, 3=AE/NC,
    /// 6=BE, 7=A, 0xA=P(PF=1), 0xB=NP(PF=0).
    pub fn jcc(&mut self, cc: u8, target_label: &str) {
        self.code.push(0x0F);
        self.code.push(0x80 | cc);
        self.rel32_fixup(target_label);
    }

    pub fn setcc(&mut self, cc: u8, dst: u8) {
        // SETcc r/m8: 0F 90+cc /0. Always REX'd (see module header) so any of the 16 GP
        // registers' low byte is addressable uniformly.
        self.code.push(rex_byte(false, 0, Rm::Direct(dst)));
        self.code.push(0x0F);
        self.code.push(0x90 | cc);
        self.code.push(modrm(0b11, 0, dst));
    }

    pub fn ret(&mut self) {
        self.code.push(0xC3);
    }

    pub fn push_reg(&mut self, r: u8) {
        self.code.push(rex_byte(false, 0, Rm::Direct(r)));
        self.code.push(0x50 | (r & 7));
    }

    pub fn pop_reg(&mut self, r: u8) {
        self.code.push(rex_byte(false, 0, Rm::Direct(r)));
        self.code.push(0x58 | (r & 7));
    }

    /// `mov rbp, rsp`.
    pub fn mov_rbp_rsp(&mut self) {
        self.mov_reg_reg(W::B8, RBP, RSP);
    }

    /// `mov rsp, rbp`.
    pub fn mov_rsp_rbp(&mut self) {
        self.mov_reg_reg(W::B8, RSP, RBP);
    }

    /// `sub rsp, imm32`.
    pub fn sub_rsp_imm(&mut self, imm: i32) {
        self.code.push(rex_byte(true, 0, Rm::Direct(RSP)));
        self.code.push(0x81);
        self.code.push(modrm(0b11, 5, RSP));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    fn width_prefix(w: W) -> Option<u8> {
        if w == W::B2 {
            Some(0x66)
        } else {
            None
        }
    }

    /// `mov dst, [rbp+disp]` at width `w`, zero/sign-extension left entirely to the caller
    /// (see `oracle.rs`'s width-exactness design): only the low `w.bytes()` bytes of `dst`
    /// are meaningfully written here.
    pub fn mov_reg_rbp(&mut self, w: W, dst: u8, disp: i32) {
        let opcode: &[u8] = if w == W::B1 { &[0x8A] } else { &[0x8B] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            dst,
            Rm::RbpDisp(disp),
        );
    }

    pub fn mov_rbp_reg(&mut self, w: W, disp: i32, src: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0x88] } else { &[0x89] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            src,
            Rm::RbpDisp(disp),
        );
    }

    pub fn mov_reg_ind(&mut self, w: W, dst: u8, base: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0x8A] } else { &[0x8B] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            dst,
            Rm::IndBase(base),
        );
    }

    pub fn mov_ind_reg(&mut self, w: W, base: u8, src: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0x88] } else { &[0x89] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            src,
            Rm::IndBase(base),
        );
    }

    pub fn mov_reg_reg(&mut self, w: W, dst: u8, src: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0x88] } else { &[0x89] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            src,
            Rm::Direct(dst),
        );
    }

    /// `movabs dst, imm64` (opcode+reg form, always full 64-bit — see `oracle.rs` for why a
    /// uniform 8-byte immediate load is always correct regardless of the value's logical
    /// width).
    pub fn movabs(&mut self, dst: u8, imm: i64) {
        self.code.push(rex_byte(true, 0, Rm::Direct(dst)));
        self.code.push(0xB8 | (dst & 7));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    pub fn lea_rbp(&mut self, dst: u8, disp: i32) {
        self.insn(None, true, &[0x8D], dst, Rm::RbpDisp(disp));
    }

    /// `mov dst, imm32` sign-extended into a 64-bit register (used only for small constants
    /// where a full movabs would be needlessly verbose, e.g. the thread-loop bound `0`/`1`).
    pub fn mov_reg_imm32(&mut self, w: W, dst: u8, imm: i32) {
        match w {
            W::B8 => {
                // REX.W + C7 /0 id: sign-extends the 32-bit immediate to 64 bits.
                self.code.push(rex_byte(true, 0, Rm::Direct(dst)));
                self.code.push(0xC7);
                self.code.push(modrm(0b11, 0, dst));
                self.code.extend_from_slice(&imm.to_le_bytes());
            }
            W::B4 => {
                self.code.push(rex_byte(false, 0, Rm::Direct(dst)));
                self.code.push(0xC7);
                self.code.push(modrm(0b11, 0, dst));
                self.code.extend_from_slice(&imm.to_le_bytes());
            }
            W::B2 => {
                self.code.push(0x66);
                self.code.push(rex_byte(false, 0, Rm::Direct(dst)));
                self.code.push(0xC7);
                self.code.push(modrm(0b11, 0, dst));
                self.code.extend_from_slice(&(imm as i16).to_le_bytes());
            }
            W::B1 => {
                self.code.push(rex_byte(false, 0, Rm::Direct(dst)));
                self.code.push(0xC6);
                self.code.push(modrm(0b11, 0, dst));
                self.code.push(imm as u8);
            }
        }
    }

    /// One of the six "op r/m, r" ALU forms (ADD/OR/AND/SUB/XOR/CMP), operating in place on
    /// `dst` (`dst := dst OP src`, or just flags for CMP).
    pub fn alu_reg_reg(&mut self, op: AluOp, w: W, dst: u8, src: u8) {
        let opcode = [op.opcode(w)];
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            &opcode,
            src,
            Rm::Direct(dst),
        );
    }

    /// Two-operand signed multiply, `dst := dst * src` (16/32/64-bit native form only — the
    /// oracle promotes 8-bit multiplies to 32-bit before calling this, since IMUL has no
    /// two-operand 8-bit encoding).
    pub fn imul_reg_reg(&mut self, w: W, dst: u8, src: u8) {
        debug_assert!(w != W::B1);
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            &[0x0F, 0xAF],
            dst,
            Rm::Direct(src),
        );
    }

    /// `cdq`/`cqo`/`cwd`: sign-extends the accumulator (eax/rax/ax) into edx:eax / rdx:rax /
    /// dx:ax, as required before `idiv`.
    pub fn cdq(&mut self, w: W) {
        match w {
            W::B8 => {
                self.code.push(rex_byte(true, 0, Rm::Direct(0)));
                self.code.push(0x99);
            }
            W::B2 => {
                self.code.push(0x66);
                self.code.push(0x99);
            }
            _ => self.code.push(0x99),
        }
    }

    /// `idiv r/m` (signed divide edx:eax / rdx:rax / dx:ax by `divisor`; quotient in
    /// eax/rax/ax, remainder in edx/rdx/dx). No 8-bit form is ever requested (see
    /// `oracle.rs`).
    pub fn idiv_reg(&mut self, w: W, divisor: u8) {
        debug_assert!(w != W::B1);
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            &[0xF7],
            7,
            Rm::Direct(divisor),
        );
    }

    /// `shl`/`shr`/`sar dst, cl`.
    pub fn shift_cl(&mut self, kind: ShiftKind, w: W, dst: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0xD2] } else { &[0xD3] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            kind.ext(),
            Rm::Direct(dst),
        );
    }

    /// `shl`/`shr`/`sar dst, 1` (the dedicated shift-by-one opcode, avoiding a CL load for
    /// the fixed-shift-by-one case the `uitofp` software path needs).
    pub fn shift1(&mut self, kind: ShiftKind, w: W, dst: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0xD0] } else { &[0xD1] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            kind.ext(),
            Rm::Direct(dst),
        );
    }

    /// `test reg, reg` (sets ZF iff `reg == 0`, SF from `reg`'s own sign bit) — the uniform
    /// way this backend checks "is this value zero" (`select`'s condition, a loop's exit
    /// test) without needing a separate compare-against-immediate encoding.
    pub fn test_reg_reg(&mut self, w: W, reg: u8) {
        let opcode: &[u8] = if w == W::B1 { &[0x84] } else { &[0x85] };
        self.insn(
            Self::width_prefix(w),
            w == W::B8,
            opcode,
            reg,
            Rm::Direct(reg),
        );
    }

    /// One of the six Group-1 "op r/m, imm" forms, `dst := dst OP imm` in place. `imm` is
    /// truncated to `w`'s width (sign bits beyond that width are never emitted).
    pub fn alu_reg_imm32(&mut self, op: AluOp, w: W, dst: u8, imm: i32) {
        let opcode: &[u8] = if w == W::B1 { &[0x80] } else { &[0x81] };
        if let Some(p) = Self::width_prefix(w) {
            self.code.push(p);
        }
        self.code.push(rex_byte(w == W::B8, 0, Rm::Direct(dst)));
        self.code.extend_from_slice(opcode);
        self.code.push(modrm(0b11, op.group1_ext(), dst));
        match w {
            W::B1 => self.code.push(imm as u8),
            W::B2 => self.code.extend_from_slice(&(imm as i16).to_le_bytes()),
            W::B4 | W::B8 => self.code.extend_from_slice(&imm.to_le_bytes()),
        }
    }

    /// `movzx`/`movsx dst(32/64), src(8/16)`, or `movsxd dst(64), src(32)` for the one 32->64
    /// sign-extension case (there is no `movzxd`; a plain 32-bit write already zero-extends).
    pub fn movzx(&mut self, dst_w: W, src_w: W, dst: u8, rm: Rm) {
        debug_assert!(src_w == W::B1 || src_w == W::B2);
        let opcode: &[u8] = if src_w == W::B1 {
            &[0x0F, 0xB6]
        } else {
            &[0x0F, 0xB7]
        };
        self.insn(None, dst_w == W::B8, opcode, dst, rm);
    }

    pub fn movsx(&mut self, dst_w: W, src_w: W, dst: u8, rm: Rm) {
        match src_w {
            W::B1 => self.insn(None, dst_w == W::B8, &[0x0F, 0xBE], dst, rm),
            W::B2 => self.insn(None, dst_w == W::B8, &[0x0F, 0xBF], dst, rm),
            W::B4 => self.insn(None, true, &[0x63], dst, rm),
            W::B8 => unreachable!("sign-extending from 8 bytes"),
        }
    }

    // ---- SSE scalar float ops --------------------------------------------------------

    pub fn movss_load(&mut self, dst_xmm: u8, rm: Rm) {
        self.insn(Some(0xF3), false, &[0x0F, 0x10], dst_xmm, rm);
    }
    pub fn movss_store(&mut self, rm: Rm, src_xmm: u8) {
        self.insn(Some(0xF3), false, &[0x0F, 0x11], src_xmm, rm);
    }
    pub fn movsd_load(&mut self, dst_xmm: u8, rm: Rm) {
        self.insn(Some(0xF2), false, &[0x0F, 0x10], dst_xmm, rm);
    }
    pub fn movsd_store(&mut self, rm: Rm, src_xmm: u8) {
        self.insn(Some(0xF2), false, &[0x0F, 0x11], src_xmm, rm);
    }

    /// `movss`/`movsd dst_xmm, src_xmm` (register-to-register form) — the ordinary way to
    /// copy one scalar float register to another, reusing the same load opcode with a
    /// register-direct `Rm`.
    pub fn sse_move(&mut self, dst_xmm: u8, src_xmm: u8, is_f64: bool) {
        if is_f64 {
            self.movsd_load(dst_xmm, Rm::Direct(src_xmm));
        } else {
            self.movss_load(dst_xmm, Rm::Direct(src_xmm));
        }
    }

    /// `op dst_xmm, src` for the four scalar-float arithmetic ops, f32 (`movss`-family
    /// mandatory prefix `F3`) or f64 (`F2`).
    pub fn sse_arith(&mut self, op: SseArith, is_f64: bool, dst_xmm: u8, rm: Rm) {
        let prefix = if is_f64 { 0xF2 } else { 0xF3 };
        self.insn(Some(prefix), false, &[0x0F, op.opcode()], dst_xmm, rm);
    }

    pub fn ucomiss(&mut self, a_xmm: u8, rm: Rm) {
        self.insn(None, false, &[0x0F, 0x2E], a_xmm, rm);
    }
    pub fn ucomisd(&mut self, a_xmm: u8, rm: Rm) {
        self.insn(Some(0x66), false, &[0x0F, 0x2E], a_xmm, rm);
    }

    pub fn cvtss2sd(&mut self, dst_xmm: u8, rm: Rm) {
        self.insn(Some(0xF3), false, &[0x0F, 0x5A], dst_xmm, rm);
    }
    pub fn cvtsd2ss(&mut self, dst_xmm: u8, rm: Rm) {
        self.insn(Some(0xF2), false, &[0x0F, 0x5A], dst_xmm, rm);
    }

    /// `cvttss2si`/`cvttsd2si dst_gpr, src_xmm/m`: truncating float-to-signed-int, `w`
    /// selects the 32- or 64-bit destination GPR form.
    pub fn cvtt_to_si(&mut self, is_f64: bool, w: W, dst_gpr: u8, rm: Rm) {
        let prefix = if is_f64 { 0xF2 } else { 0xF3 };
        self.insn(Some(prefix), w == W::B8, &[0x0F, 0x2C], dst_gpr, rm);
    }

    /// `cvtsi2ss`/`cvtsi2sd dst_xmm, src_gpr/m`: `w` selects the 32- or 64-bit source GPR
    /// form.
    pub fn cvt_si_to(&mut self, is_f64: bool, w: W, dst_xmm: u8, rm: Rm) {
        let prefix = if is_f64 { 0xF2 } else { 0xF3 };
        self.insn(Some(prefix), w == W::B8, &[0x0F, 0x2A], dst_xmm, rm);
    }

    /// `movd`/`movq xmm, r/m` (int -> xmm bit copy).
    pub fn movd_to_xmm(&mut self, w: W, dst_xmm: u8, rm: Rm) {
        self.insn(Some(0x66), w == W::B8, &[0x0F, 0x6E], dst_xmm, rm);
    }
    /// `movd`/`movq r/m, xmm` (xmm -> int bit copy).
    pub fn movd_from_xmm(&mut self, w: W, rm: Rm, src_xmm: u8) {
        self.insn(Some(0x66), w == W::B8, &[0x0F, 0x7E], src_xmm, rm);
    }

    pub fn nop(&mut self) {
        self.code.push(0x90);
    }

    /// Resolves every recorded branch fixup against the labels defined by the time this is
    /// called (every label must be defined — an unresolved label is a codegen bug, not a
    /// user-facing error, hence the panic) and returns the finished byte buffer.
    pub fn finish(self) -> Vec<u8> {
        let Enc {
            mut code,
            labels,
            fixups,
        } = self;
        for (pos, name) in &fixups {
            let target = *labels
                .get(name)
                .unwrap_or_else(|| panic!("codegen bug: undefined label `{name}`"));
            let rel = target as i64 - (*pos as i64 + 4);
            let rel = i32::try_from(rel)
                .unwrap_or_else(|_| panic!("codegen bug: branch to `{name}` out of rel32 range"));
            code[*pos..*pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
        code
    }
}

impl Default for Enc {
    fn default() -> Enc {
        Enc::new()
    }
}

#[derive(Clone, Copy)]
pub enum AluOp {
    Add,
    Or,
    And,
    Sub,
    Xor,
    Cmp,
}

impl AluOp {
    fn opcode(self, w: W) -> u8 {
        let base = match self {
            AluOp::Add => 0x00,
            AluOp::Or => 0x08,
            AluOp::And => 0x20,
            AluOp::Sub => 0x28,
            AluOp::Xor => 0x30,
            AluOp::Cmp => 0x38,
        };
        // "op r/m8, r8" is the base opcode; "op r/m16/32/64, r16/32/64" is base+1.
        if w == W::B1 {
            base
        } else {
            base + 1
        }
    }

    /// The Group-1 opcode-extension digit (ModRM.reg) selecting this op for the `80`/`81`
    /// "op r/m, imm" forms.
    fn group1_ext(self) -> u8 {
        match self {
            AluOp::Add => 0,
            AluOp::Or => 1,
            AluOp::And => 4,
            AluOp::Sub => 5,
            AluOp::Xor => 6,
            AluOp::Cmp => 7,
        }
    }
}

#[derive(Clone, Copy)]
pub enum ShiftKind {
    Shl,
    Shr,
    Sar,
}

impl ShiftKind {
    fn ext(self) -> u8 {
        match self {
            ShiftKind::Shl => 4,
            ShiftKind::Shr => 5,
            ShiftKind::Sar => 7,
        }
    }
}

#[derive(Clone, Copy)]
pub enum SseArith {
    Add,
    Mul,
    Sub,
    Div,
}

impl SseArith {
    fn opcode(self) -> u8 {
        match self {
            SseArith::Add => 0x58,
            SseArith::Mul => 0x59,
            SseArith::Sub => 0x5C,
            SseArith::Div => 0x5E,
        }
    }
}

/// Condition-code nibbles shared by `Jcc`/`SETcc`.
pub mod cc {
    pub const B: u8 = 0x2;
    pub const AE: u8 = 0x3;
    pub const E: u8 = 0x4;
    pub const NE: u8 = 0x5;
    pub const BE: u8 = 0x6;
    pub const A: u8 = 0x7;
    pub const S: u8 = 0x8;
    pub const P: u8 = 0xA;
    pub const NP: u8 = 0xB;
    pub const L: u8 = 0xC;
    pub const GE: u8 = 0xD;
    pub const LE: u8 = 0xE;
    pub const G: u8 = 0xF;
}
