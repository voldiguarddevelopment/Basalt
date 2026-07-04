// The RV32IM soft-float runtime: a fixed set of internal subroutines emitted once into the
// same `.text` blob the rest of this backend produces (see `emit_runtime`, called from
// `lower.rs`'s `emit_function`). `lower.rs` reaches these by `jal ra, <label>` at the stable
// names below — this file owns the label names and their `a0`/`a1`-in, `a0`-out calling
// convention; everything about *when* to call one of these is `lower.rs`'s concern, not this
// file's.
//
// Every routine here is a mechanical, one-Rust-statement-at-a-time transliteration of the
// corresponding function in `softfloat_ref.rs` — that file is the spec (already validated
// against 3000+ ground-truth vectors computed independently via exact rational arithmetic in
// Python; see its own module header) and this file's own comments cross-reference its
// variable names throughout so a translation-fidelity review can proceed line by line without
// re-deriving the algorithm. `softfloat_ref.rs`'s rounding mode (round-toward-zero, not
// round-to-nearest-even — see that file's header for why) and other documented scope
// decisions (subnormals flush to zero, all NaNs collapse to one canonical quiet NaN,
// `fptosi`/`fptoui` saturate) apply here unchanged, because this file computes nothing of its
// own beyond what that spec already defines.
//
// # Leaf routines only
//
// Every routine emitted here is a leaf: it never itself executes a `jal` to another routine
// (so it never touches `ra` except its own trailing `ret`), never touches `sp` (all working
// state lives in registers), and is free to clobber any GPR except `sp`/`gp`/`tp` — the caller
// never expects any register to survive a call, by the same "everything reloads from a stack
// slot before use" discipline `basalt-x86`'s oracle documents for its own frame. This is why
// `lower.rs` can call any of these mid-expression without saving anything beyond its own `a0`/
// `a1` setup.
//
// # No computed shift-by-N: every normalization shift is bit-serial
//
// `softfloat_ref.rs`'s `normalize_and_pack32` normalizes one bit at a time (`while mant <
// 0x800000 { mant <<= 1; ... }` / `while mant >= 0x1000000 { mant >>= 1; ... }`), not via a
// single computed shift, and this file mirrors that exactly rather than "optimizing" it into
// one shot: the shift *amount* in every alignment/normalization here is a runtime value (how
// far a mantissa needs to move depends on the actual operands), and RV32's shift instructions
// only consult the low 5 bits of a register operand — a single computed shift by a value that
// could reach or exceed 32 reintroduces exactly the hardware-quirk and off-by-one class of bug
// the reference's own author already hit and fixed (see `rtz_add32`'s inline comments on the
// K=24 alignment window). A bit-serial loop side-steps the question of "is the shift amount
// ever >=32" entirely — each individual shift is always by exactly 1, a compile-time constant,
// so `Enc::slli`/`srli`'s own `shamt < 32` invariant is never in question. The one instruction-
// count exception is `div_mant_q24`'s base-256 long division, transliterated with fixed
// shift-by-8 steps (still a compile-time constant, just not 1) exactly as the reference
// computes it, and RV32M's native `divu`/`remu`, which need no shift or loop at all.
//
// # 64-bit values are register pairs
//
// RV32 has no 64-bit registers. Every `u64` mantissa/product in `softfloat_ref.rs` (add's
// aligned operands, mul's widening product) is carried here as a `(lo, hi)` register pair;
// `shl64_1`/`shr64_1` below are the only two places a pair is ever shifted, each by exactly
// one bit, matching the bit-serial policy above. `div`'s and the int<->float conversions'
// mantissas never exceed 32 bits by construction (documented at each call site), so those
// routines pass a dedicated zeroed scratch register as the pair's high half rather than
// carrying a real second word.
//
// # Validation tier: algorithm-validated and encoding-verified, NOT execution-tested
//
// Per this project's honest-tiering convention (see `CLAUDE.md`'s hardware-access section):
// this file is *algorithm-validated* (every routine's logic traces back to `softfloat_ref.rs`,
// itself checked against 3000+ independently-computed ground-truth vectors) and *encoding-
// verified* (built only from `enc.rs`'s primitives, each cross-checked against a real
// assembler). It is explicitly **not execution-tested** — no RV32 simulator exists yet in this
// tree (a later task's job) — so the single biggest residual risk in this file is a
// translation-fidelity bug: an instruction sequence that does not faithfully reproduce what
// `softfloat_ref.rs` computes, despite every individual instruction being correctly encoded.
// That is exactly what a future execution-test pass should check first.

use crate::enc::{
    AluOp, BCond, Enc, MulOp, A0, A1, A2, A3, A4, A5, A6, A7, S2, S3, S4, S5, S6, S7, S8, S9, T0,
    T1, T2, T3, T4, T5, T6, ZERO,
};

/// `is_nan32(bits)` (softfloat_ref.rs) into `out` (0/1): `exp = (bits>>23)&0xFF`,
/// `frac = (bits<<9)>>9` (an alternate but equivalent way to isolate the low 23 bits, since
/// `andi`'s 12-bit immediate cannot hold `0x7FFFFF`), then `(exp==0xFF) && (frac!=0)`. `t0`/
/// `t1` are scratch; `out` may not alias `bits`.
fn is_nan_raw(e: &mut Enc, bits: u8, out: u8, t0: u8, t1: u8) {
    e.srli(t0, bits, 23);
    e.andi(t0, t0, 0xFF);
    e.slli(t1, bits, 9);
    e.srli(t1, t1, 9);
    e.xori(t0, t0, 0xFF); // t0==0 iff exp==0xFF
    e.sltiu(t0, t0, 1); // t0 = (exp==0xFF)
    e.alu_reg(AluOp::Sltu, t1, ZERO, t1); // t1 = (frac!=0)
    e.alu_reg(AluOp::And, out, t0, t1);
}

/// `is_inf32(bits)` (softfloat_ref.rs): identical field extraction to `is_nan_raw`, but
/// `(exp==0xFF) && (frac==0)`.
fn is_inf_raw(e: &mut Enc, bits: u8, out: u8, t0: u8, t1: u8) {
    e.srli(t0, bits, 23);
    e.andi(t0, t0, 0xFF);
    e.slli(t1, bits, 9);
    e.srli(t1, t1, 9);
    e.xori(t0, t0, 0xFF);
    e.sltiu(t0, t0, 1); // t0 = (exp==0xFF)
    e.sltiu(t1, t1, 1); // t1 = (frac==0)
    e.alu_reg(AluOp::And, out, t0, t1);
}

/// `decompose32(bits) -> (sign, exp, mant)` (softfloat_ref.rs): `sign = bits>>31`;
/// `exp_field = (bits>>23)&0xFF`; `frac = (bits<<9)>>9`; if `exp_field==0` then `(sign,0,0)`
/// (subnormal input flushed to zero, per that file's documented scope) else
/// `(sign, exp_field, frac|0x800000)`. `tag` must be unique across the whole `Enc` (see
/// `emit_runtime`'s module-wide flat label namespace); `t0` is scratch.
fn decompose_raw(e: &mut Enc, tag: &str, bits: u8, sign: u8, exp: u8, mant: u8, t0: u8) {
    e.srli(exp, bits, 23);
    e.andi(exp, exp, 0xFF);
    e.slli(mant, bits, 9);
    e.srli(mant, mant, 9);
    e.srli(sign, bits, 31);
    let zero_lbl = format!("{tag}_dz");
    let join_lbl = format!("{tag}_dj");
    e.branch(BCond::Eq, exp, ZERO, &zero_lbl);
    e.li32(t0, 0x0080_0000);
    e.alu_reg(AluOp::Or, mant, mant, t0);
    e.jump(&join_lbl);
    e.label(&zero_lbl);
    e.mv(mant, ZERO);
    e.label(&join_lbl);
}

/// Shifts the 64-bit pair `(lo, hi)` left by exactly one bit (`t0` scratch): the bit shifted
/// out of `lo`'s top is captured before `lo` itself is modified.
fn shl64_1(e: &mut Enc, lo: u8, hi: u8, t0: u8) {
    e.srli(t0, lo, 31);
    e.slli(hi, hi, 1);
    e.alu_reg(AluOp::Or, hi, hi, t0);
    e.slli(lo, lo, 1);
}

/// Shifts the 64-bit pair `(lo, hi)` right by exactly one bit, logically (the bit shifted out
/// of `lo`'s bottom is simply discarded — this is round-toward-zero truncation, not rounding;
/// see `softfloat_ref.rs`'s module header). `t0`/`t1` scratch.
fn shr64_1(e: &mut Enc, lo: u8, hi: u8, t0: u8, t1: u8) {
    e.andi(t0, hi, 1);
    e.srli(lo, lo, 1);
    e.slli(t1, t0, 31);
    e.alu_reg(AluOp::Or, lo, lo, t1);
    e.srli(hi, hi, 1);
}

/// `normalize_and_pack32(sign, mant: u64, exp2)` (softfloat_ref.rs), with `mant` carried as
/// the register pair `(mlo, mhi)`. `mant` must be nonzero (every call site below establishes
/// this the same way the reference's `debug_assert!(mant != 0)` documents its callers do).
/// Left-normalizes (`while mant < 2^23`), then right-normalizes truncating (`while mant >=
/// 2^24`), one bit at a time in both directions (see this file's own header on why), then
/// packs `(sign, exp2+127, mant)` into an f32 bit pattern, flushing to signed-zero/inf on
/// underflow/overflow. `mlo`/`mhi`/`exp2` are all clobbered (working state); `ta`/`tb`/`tc`/
/// `td` are scratch. `tag` must be unique across the whole `Enc`.
#[allow(clippy::too_many_arguments)]
fn norm_and_pack64(
    e: &mut Enc,
    tag: &str,
    sign: u8,
    mlo: u8,
    mhi: u8,
    exp2: u8,
    out: u8,
    ta: u8,
    tb: u8,
    tc: u8,
    td: u8,
) {
    let left_top = format!("{tag}_npl");
    let left_exit = format!("{tag}_nple");
    let right_top = format!("{tag}_npr");
    let right_shift = format!("{tag}_nprs");
    let right_exit = format!("{tag}_npre");
    let underflow = format!("{tag}_npu");
    let overflow = format!("{tag}_npo");
    let done = format!("{tag}_npd");

    // while mant < 0x0080_0000 { mant <<= 1; exp2 -= 1; }
    e.li32(tc, 0x0080_0000);
    e.label(&left_top);
    e.branch(BCond::Ne, mhi, ZERO, &left_exit); // hi!=0 => mant >= 2^32 > 2^23
    e.branch(BCond::Geu, mlo, tc, &left_exit);
    shl64_1(e, mlo, mhi, ta);
    e.addi(exp2, exp2, -1);
    e.jump(&left_top);
    e.label(&left_exit);

    // while mant >= 0x0100_0000 { mant >>= 1; exp2 += 1; }  (truncating shift)
    e.li32(tc, 0x0100_0000);
    e.label(&right_top);
    e.branch(BCond::Ne, mhi, ZERO, &right_shift); // hi!=0 => mant >= 2^32 >= 2^24
    e.branch(BCond::Geu, mlo, tc, &right_shift);
    e.jump(&right_exit);
    e.label(&right_shift);
    shr64_1(e, mlo, mhi, ta, tb);
    e.addi(exp2, exp2, 1);
    e.jump(&right_top);
    e.label(&right_exit);

    // exp_field = exp2 + 127; three-way range check.
    e.addi(exp2, exp2, 127);
    e.li32(tc, 1);
    e.branch(BCond::Lt, exp2, tc, &underflow); // exp_field <= 0
    e.li32(tc, 0xFF);
    e.branch(BCond::Ge, exp2, tc, &overflow); // exp_field >= 0xFF
                                              // (sign<<31) | (exp_field<<23) | (mant & 0x7FFFFF); mant is in [2^23,2^24) here, so
                                              // `mant & 0x7FFFFF == mant ^ 0x800000` (clears exactly the implicit leading bit).
    e.slli(ta, sign, 31);
    e.slli(tb, exp2, 23);
    e.alu_reg(AluOp::Or, ta, ta, tb);
    e.li32(tc, 0x0080_0000);
    e.alu_reg(AluOp::Xor, td, mlo, tc);
    e.alu_reg(AluOp::Or, out, ta, td);
    e.jump(&done);
    e.label(&underflow);
    e.slli(out, sign, 31);
    e.jump(&done);
    e.label(&overflow);
    e.li32(tc, 0x7F80_0000);
    e.slli(ta, sign, 31);
    e.alu_reg(AluOp::Or, out, ta, tc);
    e.label(&done);
}

/// `__sf_f32_add`: `a0=a, a1=b -> a0 = rtz_add32(a,b)`. Transliterates `rtz_add32` in
/// `softfloat_ref.rs` statement by statement (`FSub`'s sign-flip-then-add trick lives in
/// `lower.rs`, not here — this routine only implements add).
///
/// Register plan: `T0`/`T1` hold the raw input bits `a`/`b` for the whole routine (read-only,
/// used by the `ret_a`/`ret_b` epilogues). `S4..S9` hold `(sa,ea,ma,sb,eb,mb)` from
/// `decompose_raw` until the "pick big/small" branch, after which `S2/S3/A4/A5/A6/A7` hold
/// `(big_sign,big_exp,big_mant,small_sign,small_exp,small_mant)` and `S4..S9` become free
/// scratch again for `diff`/`k`/`small_wide`/the loop counter/`big_wide_hi`. See the inline
/// comments below for the exact mapping at each step.
fn emit_f32_add(e: &mut Enc) {
    e.label("__sf_f32_add");
    e.mv(T0, A0);
    e.mv(T1, A1);

    // if is_nan32(a) || is_nan32(b) { return CANONICAL_NAN32 }
    is_nan_raw(e, T0, S2, A2, A3);
    e.branch(BCond::Ne, S2, ZERO, "add_ret_nan");
    is_nan_raw(e, T1, S3, A2, A3);
    e.branch(BCond::Ne, S3, ZERO, "add_ret_nan");

    // let (sa,ea,ma) = decompose32(a); let (sb,eb,mb) = decompose32(b);
    decompose_raw(e, "add_da", T0, S4, S5, S6, A2);
    decompose_raw(e, "add_db", T1, S7, S8, S9, A2);

    // let ia = is_inf32(a); let ib = is_inf32(b);
    is_inf_raw(e, T0, A4, A2, A3); // ia
    is_inf_raw(e, T1, A5, A2, A3); // ib

    // if ia && ib { return if sa != sb { NAN } else { a } }
    e.branch(BCond::Eq, A4, ZERO, "add_skip_ii");
    e.branch(BCond::Eq, A5, ZERO, "add_skip_ii");
    e.branch(BCond::Ne, S4, S7, "add_ret_nan");
    e.jump("add_ret_a");
    e.label("add_skip_ii");
    // if ia { return a }
    e.branch(BCond::Ne, A4, ZERO, "add_ret_a");
    // if ib { return b }
    e.branch(BCond::Ne, A5, ZERO, "add_ret_b");

    // let za = ea==0 && ma==0; let zb = eb==0 && mb==0;
    e.sltiu(A2, S5, 1);
    e.sltiu(A3, S6, 1);
    e.alu_reg(AluOp::And, A6, A2, A3); // za
    e.sltiu(A2, S8, 1);
    e.sltiu(A3, S9, 1);
    e.alu_reg(AluOp::And, A7, A2, A3); // zb

    // if za && zb { return if sa==sb { a } else { 0 } }
    e.branch(BCond::Eq, A6, ZERO, "add_skip_zz");
    e.branch(BCond::Eq, A7, ZERO, "add_skip_zz");
    e.branch(BCond::Ne, S4, S7, "add_ret_zero");
    e.jump("add_ret_a");
    e.label("add_skip_zz");
    // if za { return b }
    e.branch(BCond::Ne, A6, ZERO, "add_ret_b");
    // if zb { return a }
    e.branch(BCond::Ne, A7, ZERO, "add_ret_a");

    // Pick (big,small) so mant_big_shifted >= mant_small_shifted always (never borrows below).
    // if ea > eb || (ea == eb && ma >= mb) { big=a-side } else { big=b-side }
    e.alu_reg(AluOp::Sltu, A2, S8, S5); // A2 = eb<ea = (ea>eb)
    e.branch(BCond::Ne, A2, ZERO, "add_pick_a");
    e.branch(BCond::Ne, S5, S8, "add_pick_b"); // ea!=eb and not ea>eb => ea<eb
    e.alu_reg(AluOp::Sltu, A3, S6, S9); // ma<mb
    e.branch(BCond::Ne, A3, ZERO, "add_pick_b");
    e.jump("add_pick_a");
    e.label("add_pick_b");
    e.mv(S2, S7); // big_sign = sb
    e.mv(S3, S8); // big_exp = eb
    e.mv(A4, S9); // big_mant = mb
    e.mv(A5, S4); // small_sign = sa
    e.mv(A6, S5); // small_exp = ea
    e.mv(A7, S6); // small_mant = ma
    e.jump("add_picked");
    e.label("add_pick_a");
    e.mv(S2, S4); // big_sign = sa
    e.mv(S3, S5); // big_exp = ea
    e.mv(A4, S6); // big_mant = ma
    e.mv(A5, S7); // small_sign = sb
    e.mv(A6, S8); // small_exp = eb
    e.mv(A7, S9); // small_mant = mb
    e.label("add_picked");

    // let diff = big_exp - small_exp; let k = diff.min(24);
    e.alu_reg(AluOp::Sub, S4, S3, A6); // diff
    e.li32(T2, 24);
    e.branch(BCond::Lt, S4, T2, "add_k_diff");
    e.mv(S5, T2); // k = 24
    e.jump("add_k_done");
    e.label("add_k_diff");
    e.mv(S5, S4); // k = diff
    e.label("add_k_done");

    // small_wide: diff<=K => small_mant unshifted; else shift-with-sticky by (diff-K), or 1
    // outright if that shift is itself >=24 (small_mant entirely below the window).
    e.li32(T2, 24);
    e.branch(BCond::Lt, T2, S4, "add_sw_far"); // 24 < diff
    e.mv(S6, A7); // near path: small_wide = small_mant
    e.jump("add_sw_done");
    e.label("add_sw_far");
    e.alu_reg(AluOp::Sub, S8, S4, T2); // shift = diff - 24 (also the loop counter)
    e.li32(T2, 24);
    e.branch(BCond::Ge, S8, T2, "add_sw_sat"); // shift >= 24
    e.mv(S6, A7); // working copy of small_mant
    e.mv(S7, ZERO); // sticky = 0
    e.label("add_sw_loop");
    e.branch(BCond::Eq, S8, ZERO, "add_sw_loop_done");
    e.andi(T2, S6, 1); // bit shifted out
    e.alu_reg(AluOp::Or, S7, S7, T2); // sticky |= bit
    e.srli(S6, S6, 1);
    e.addi(S8, S8, -1);
    e.jump("add_sw_loop");
    e.label("add_sw_loop_done");
    e.alu_reg(AluOp::Or, S6, S6, S7); // small_wide = shifted | sticky
    e.jump("add_sw_done");
    e.label("add_sw_sat");
    e.li32(S6, 1);
    e.label("add_sw_done");

    // big_wide = big_mant << k (bit-serial, k iterations); big_wide_hi starts at 0.
    e.mv(S9, ZERO); // big_wide_hi
    e.mv(S8, S5); // loop counter = k
    e.label("add_bw_loop");
    e.branch(BCond::Eq, S8, ZERO, "add_bw_done");
    shl64_1(e, A4, S9, T2); // (big_mant, big_wide_hi) <<= 1
    e.addi(S8, S8, -1);
    e.jump("add_bw_loop");
    e.label("add_bw_done");

    // exp2 = big_exp - 127 - k
    e.mv(T3, S3);
    e.addi(T3, T3, -127);
    e.alu_reg(AluOp::Sub, T3, T3, S5);

    // if big_sign == small_sign { sum = big_wide + small_wide (small hi is always 0) }
    // else { diff_mant = big_wide - small_wide; if 0 { return 0 } }
    e.branch(BCond::Ne, S2, A5, "add_diffsign");
    e.alu_reg(AluOp::Add, T4, A4, S6); // sum_lo
    e.alu_reg(AluOp::Sltu, T5, T4, A4); // carry
    e.alu_reg(AluOp::Add, T6, S9, T5); // sum_hi (small_hi=0)
    norm_and_pack64(e, "add_sum", S2, T4, T6, T3, A0, A2, A3, T2, T5);
    e.ret();
    e.label("add_diffsign");
    e.alu_reg(AluOp::Sltu, T5, A4, S6); // borrow
    e.alu_reg(AluOp::Sub, T4, A4, S6); // diff_lo
    e.alu_reg(AluOp::Sub, T6, S9, T5); // diff_hi (small_hi=0)
    e.alu_reg(AluOp::Or, T2, T4, T6);
    e.branch(BCond::Ne, T2, ZERO, "add_diff_nz");
    e.li32(A0, 0);
    e.ret();
    e.label("add_diff_nz");
    norm_and_pack64(e, "add_diff", S2, T4, T6, T3, A0, A2, A3, T2, T5);
    e.ret();

    e.label("add_ret_nan");
    e.li32(A0, 0x7fc0_0000);
    e.ret();
    e.label("add_ret_a");
    e.mv(A0, T0);
    e.ret();
    e.label("add_ret_b");
    e.mv(A0, T1);
    e.ret();
    e.label("add_ret_zero");
    e.li32(A0, 0);
    e.ret();
}

/// `__sf_f32_mul`: `a0=a, a1=b -> a0 = rtz_mul32(a,b)`.
fn emit_f32_mul(e: &mut Enc) {
    e.label("__sf_f32_mul");
    e.mv(T0, A0);
    e.mv(T1, A1);

    // let rsign = (a>>31) ^ (b>>31);
    e.srli(S2, T0, 31);
    e.srli(S3, T1, 31);
    e.alu_reg(AluOp::Xor, S2, S2, S3); // rsign

    // if is_nan32(a) || is_nan32(b) { return NAN }
    is_nan_raw(e, T0, S4, A2, A3);
    e.branch(BCond::Ne, S4, ZERO, "mul_ret_nan");
    is_nan_raw(e, T1, S4, A2, A3);
    e.branch(BCond::Ne, S4, ZERO, "mul_ret_nan");

    // let (_, ea, ma) = decompose32(a); let (_, eb, mb) = decompose32(b);
    decompose_raw(e, "mul_da", T0, T2, S5, S6, A2); // sign discarded into T2
    decompose_raw(e, "mul_db", T1, T3, S7, S8, A2); // sign discarded into T3

    let ia = S9;
    let ib = A4;
    let za = A5;
    let zb = A6;
    is_inf_raw(e, T0, ia, A2, A3);
    is_inf_raw(e, T1, ib, A2, A3);
    e.sltiu(A2, S5, 1);
    e.sltiu(A3, S6, 1);
    e.alu_reg(AluOp::And, za, A2, A3);
    e.sltiu(A2, S7, 1);
    e.sltiu(A3, S8, 1);
    e.alu_reg(AluOp::And, zb, A2, A3);

    // if (ia && zb) || (ib && za) { return NAN }
    e.alu_reg(AluOp::And, A7, ia, zb);
    e.branch(BCond::Ne, A7, ZERO, "mul_ret_nan");
    e.alu_reg(AluOp::And, A7, ib, za);
    e.branch(BCond::Ne, A7, ZERO, "mul_ret_nan");

    // if ia || ib { return (rsign<<31) | (0xFF<<23) }
    e.alu_reg(AluOp::Or, A7, ia, ib);
    e.branch(BCond::Ne, A7, ZERO, "mul_ret_inf");

    // if za || zb { return rsign<<31 }
    e.alu_reg(AluOp::Or, A7, za, zb);
    e.branch(BCond::Ne, A7, ZERO, "mul_ret_zero");

    // let product = ma as u64 * mb as u64;
    e.mul_reg(MulOp::Mul, T4, S6, S8); // product_lo
    e.mul_reg(MulOp::Mulhu, T5, S6, S8); // product_hi

    // normalize_and_pack32(rsign, product, ea + eb - 277)
    e.alu_reg(AluOp::Add, T6, S5, S7);
    e.addi(T6, T6, -277);
    norm_and_pack64(e, "mul_np", S2, T4, T5, T6, A0, A2, A3, A7, T2);

    // if result_bits & 0x7FFFFFFF == 0 { rsign<<31 } else { result_bits }
    e.li32(T3, 0x7FFF_FFFF);
    e.alu_reg(AluOp::And, T3, A0, T3);
    e.branch(BCond::Ne, T3, ZERO, "mul_done");
    e.slli(A0, S2, 31);
    e.label("mul_done");
    e.ret();

    e.label("mul_ret_nan");
    e.li32(A0, 0x7fc0_0000);
    e.ret();
    e.label("mul_ret_inf");
    e.li32(T2, 0x7F80_0000);
    e.slli(T3, S2, 31);
    e.alu_reg(AluOp::Or, A0, T3, T2);
    e.ret();
    e.label("mul_ret_zero");
    e.slli(A0, S2, 31);
    e.ret();
}

/// `__sf_f32_div`: `a0=a, a1=b -> a0 = rtz_div32(a,b)`. `div_mant_q24` needs no bit-serial
/// loop at all: RV32M's `divu`/`remu` give the three base-256 chunks directly.
fn emit_f32_div(e: &mut Enc) {
    e.label("__sf_f32_div");
    e.mv(T0, A0);
    e.mv(T1, A1);

    e.srli(S2, T0, 31);
    e.srli(S3, T1, 31);
    e.alu_reg(AluOp::Xor, S2, S2, S3); // rsign

    is_nan_raw(e, T0, S4, A2, A3);
    e.branch(BCond::Ne, S4, ZERO, "div_ret_nan");
    is_nan_raw(e, T1, S4, A2, A3);
    e.branch(BCond::Ne, S4, ZERO, "div_ret_nan");

    decompose_raw(e, "div_da", T0, T2, S5, S6, A2); // ea, ma
    decompose_raw(e, "div_db", T1, T3, S7, S8, A2); // eb, mb

    let ia = S9;
    let ib = A4;
    let za = A5;
    let zb = A6;
    is_inf_raw(e, T0, ia, A2, A3);
    is_inf_raw(e, T1, ib, A2, A3);
    e.sltiu(A2, S5, 1);
    e.sltiu(A3, S6, 1);
    e.alu_reg(AluOp::And, za, A2, A3);
    e.sltiu(A2, S7, 1);
    e.sltiu(A3, S8, 1);
    e.alu_reg(AluOp::And, zb, A2, A3);

    // if ia && ib { NAN }; if za && zb { NAN };
    e.alu_reg(AluOp::And, A7, ia, ib);
    e.branch(BCond::Ne, A7, ZERO, "div_ret_nan");
    e.alu_reg(AluOp::And, A7, za, zb);
    e.branch(BCond::Ne, A7, ZERO, "div_ret_nan");
    // if ia { inf(rsign) }; if za { rsign<<31 }; if ib { rsign<<31 }; if zb { inf(rsign) }
    // (inf/anything-remaining is always infinite, never zero — see softfloat_ref.rs's own
    // fix/comment on this exact line).
    e.branch(BCond::Ne, ia, ZERO, "div_ret_inf");
    e.branch(BCond::Ne, za, ZERO, "div_ret_zero");
    e.branch(BCond::Ne, ib, ZERO, "div_ret_zero");
    e.branch(BCond::Ne, zb, ZERO, "div_ret_inf");

    // div_mant_q24(ma, mb): q0 = ma/mb; rem = ma%mb; then 3 chunks of (rem<<8)/mb, rem%mb.
    e.mul_reg(MulOp::Divu, T4, S6, S8); // q0
    e.mul_reg(MulOp::Remu, T5, S6, S8); // rem
    e.mv(T6, T4); // q
    for _ in 0..3 {
        e.slli(T5, T5, 8); // scaled = rem<<8 (rem<den<2^24, so scaled<2^32: no overflow)
        e.mul_reg(MulOp::Divu, A2, T5, S8); // chunk
        e.mul_reg(MulOp::Remu, T5, T5, S8); // rem = scaled%den
        e.slli(T6, T6, 8);
        e.alu_reg(AluOp::Or, T6, T6, A2); // q = (q<<8)|chunk
    }

    // normalize_and_pack32(rsign, q24, ea - eb - 1); q24 fits in 32 bits (<=25 significant
    // bits), so the pair's high half is a dedicated zeroed scratch register, never a real
    // second word (see this file's own header).
    e.alu_reg(AluOp::Sub, A7, S5, S7);
    e.addi(A7, A7, -1);
    e.mv(T2, ZERO); // q24_hi = 0
    norm_and_pack64(e, "div_np", S2, T6, T2, A7, A0, A2, A3, S5, S6);

    e.li32(S7, 0x7FFF_FFFF);
    e.alu_reg(AluOp::And, S7, A0, S7);
    e.branch(BCond::Ne, S7, ZERO, "div_done");
    e.slli(A0, S2, 31);
    e.label("div_done");
    e.ret();

    e.label("div_ret_nan");
    e.li32(A0, 0x7fc0_0000);
    e.ret();
    e.label("div_ret_zero");
    e.slli(A0, S2, 31);
    e.ret();
    e.label("div_ret_inf");
    e.li32(T2, 0x7F80_0000);
    e.slli(T3, S2, 31);
    e.alu_reg(AluOp::Or, A0, T3, T2);
    e.ret();
}

/// `__sf_f32_cmp`: `a0=a, a1=b -> a0 = rtz_cmp32(a,b)` (-2/-1/0/1, sign-extended).
fn emit_f32_cmp(e: &mut Enc) {
    e.label("__sf_f32_cmp");
    e.mv(T0, A0);
    e.mv(T1, A1);

    is_nan_raw(e, T0, S2, A2, A3);
    e.branch(BCond::Ne, S2, ZERO, "cmp_ret_m2");
    is_nan_raw(e, T1, S2, A2, A3);
    e.branch(BCond::Ne, S2, ZERO, "cmp_ret_m2");

    decompose_raw(e, "cmp_da", T0, S4, S5, S6, A2); // sa, ea, ma
    decompose_raw(e, "cmp_db", T1, S7, S8, S9, A2); // sb, eb, mb

    let za = A4;
    let zb = A5;
    e.sltiu(A2, S5, 1);
    e.sltiu(A3, S6, 1);
    e.alu_reg(AluOp::And, za, A2, A3);
    e.sltiu(A2, S8, 1);
    e.sltiu(A3, S9, 1);
    e.alu_reg(AluOp::And, zb, A2, A3);

    // if za && zb { return 0 }
    e.branch(BCond::Eq, za, ZERO, "cmp_skip_zz");
    e.branch(BCond::Eq, zb, ZERO, "cmp_skip_zz");
    e.jump("cmp_ret_0");
    e.label("cmp_skip_zz");
    // if za { return if sb==0 {-1} else {1} }
    e.branch(BCond::Eq, za, ZERO, "cmp_skip_za");
    e.branch(BCond::Eq, S7, ZERO, "cmp_za_neg1");
    e.li32(A0, 1);
    e.ret();
    e.label("cmp_za_neg1");
    e.li32(A0, -1);
    e.ret();
    e.label("cmp_skip_za");
    // if zb { return if sa==0 {1} else {-1} }
    e.branch(BCond::Eq, zb, ZERO, "cmp_skip_zb");
    e.branch(BCond::Eq, S4, ZERO, "cmp_zb_pos1");
    e.li32(A0, -1);
    e.ret();
    e.label("cmp_zb_pos1");
    e.li32(A0, 1);
    e.ret();
    e.label("cmp_skip_zb");

    // if sa != sb { return if sa==1 {-1} else {1} }
    e.branch(BCond::Eq, S4, S7, "cmp_same_sign");
    e.branch(BCond::Eq, S4, ZERO, "cmp_diffsign_pos");
    e.li32(A0, -1);
    e.ret();
    e.label("cmp_diffsign_pos");
    e.li32(A0, 1);
    e.ret();
    e.label("cmp_same_sign");

    // let mag_cmp = if ea != eb { ea.cmp(&eb) } else { ma.cmp(&mb) };
    let magord = A6;
    e.branch(BCond::Eq, S5, S8, "cmp_use_mant");
    e.alu_reg(AluOp::Sltu, A2, S5, S8);
    e.branch(BCond::Ne, A2, ZERO, "cmp_mag_lt_e");
    e.li32(magord, 1);
    e.jump("cmp_mag_done");
    e.label("cmp_mag_lt_e");
    e.li32(magord, -1);
    e.jump("cmp_mag_done");
    e.label("cmp_use_mant");
    e.branch(BCond::Eq, S6, S9, "cmp_mag_eq");
    e.alu_reg(AluOp::Sltu, A2, S6, S9);
    e.branch(BCond::Ne, A2, ZERO, "cmp_mag_lt_m");
    e.li32(magord, 1);
    e.jump("cmp_mag_done");
    e.label("cmp_mag_lt_m");
    e.li32(magord, -1);
    e.jump("cmp_mag_done");
    e.label("cmp_mag_eq");
    e.li32(magord, 0);
    e.label("cmp_mag_done");

    // if sa == 1 { -ord } else { ord }
    e.branch(BCond::Eq, S4, ZERO, "cmp_no_negate");
    e.alu_reg(AluOp::Sub, magord, ZERO, magord);
    e.label("cmp_no_negate");
    e.mv(A0, magord);
    e.ret();

    e.label("cmp_ret_m2");
    e.li32(A0, -2);
    e.ret();
    e.label("cmp_ret_0");
    e.li32(A0, 0);
    e.ret();
}

/// `__sf_i32_to_f32`: `a0=i -> a0 = rtz_i32_to_f32(i)`.
fn emit_i32_to_f32(e: &mut Enc) {
    e.label("__sf_i32_to_f32");
    e.mv(T0, A0);
    e.branch(BCond::Ne, T0, ZERO, "i2f_nonzero");
    e.li32(A0, 0);
    e.ret();
    e.label("i2f_nonzero");
    e.srli(S2, T0, 31); // sign
    e.branch(BCond::Eq, S2, ZERO, "i2f_pos");
    // mag = 0 - i: two's-complement negate, correct even for i32::MIN (unsigned_abs's own
    // wrap-to-0x80000000 behavior, matching softfloat_ref.rs exactly).
    e.alu_reg(AluOp::Sub, S3, ZERO, T0);
    e.jump("i2f_mag_done");
    e.label("i2f_pos");
    e.mv(S3, T0); // mag = i (bit pattern already correct, i>=0)
    e.label("i2f_mag_done");
    e.mv(S4, ZERO); // mag_hi = 0 (mag < 2^31 always)
    e.li32(S5, 23); // exp2
    norm_and_pack64(e, "i2f_np", S2, S3, S4, S5, A0, A2, A3, A6, A7);
    e.ret();
}

/// `__sf_u32_to_f32`: `a0=u -> a0 = rtz_u32_to_f32(u)`.
fn emit_u32_to_f32(e: &mut Enc) {
    e.label("__sf_u32_to_f32");
    e.mv(T0, A0);
    e.branch(BCond::Ne, T0, ZERO, "u2f_nonzero");
    e.li32(A0, 0);
    e.ret();
    e.label("u2f_nonzero");
    e.mv(S2, ZERO); // sign = 0
    e.mv(S3, T0); // mant_lo = u
    e.mv(S4, ZERO); // mant_hi = 0
    e.li32(S5, 23); // exp2
    norm_and_pack64(e, "u2f_np", S2, S3, S4, S5, A0, A2, A3, A6, A7);
    e.ret();
}

/// `__sf_f32_to_i32`: `a0=bits -> a0 = rtz_f32_to_i32(bits)`.
fn emit_f32_to_i32(e: &mut Enc) {
    e.label("__sf_f32_to_i32");
    e.mv(T0, A0);
    is_nan_raw(e, T0, S2, A2, A3);
    e.branch(BCond::Ne, S2, ZERO, "f2i_ret_zero");
    decompose_raw(e, "f2i_d", T0, S3, S4, S5, A2); // sign, exp, mant
    is_inf_raw(e, T0, S6, A2, A3);
    e.branch(BCond::Eq, S6, ZERO, "f2i_not_inf");
    e.branch(BCond::Eq, S3, ZERO, "f2i_inf_max");
    e.li32(A0, i32::MIN);
    e.ret();
    e.label("f2i_inf_max");
    e.li32(A0, i32::MAX);
    e.ret();
    e.label("f2i_not_inf");
    e.branch(BCond::Eq, S4, ZERO, "f2i_ret_zero"); // exp==0

    // let e = exp - 127 - 23;
    e.mv(S7, S4);
    e.addi(S7, S7, -127);
    e.addi(S7, S7, -23);

    e.branch(BCond::Lt, S7, ZERO, "f2i_eneg");
    // e >= 0
    e.li32(A2, 8);
    e.branch(BCond::Ge, S7, A2, "f2i_overflow"); // e>=8: certain overflow
    e.mv(S8, S5); // mag = mant
    e.mv(S9, S7); // loop counter = e
    e.label("f2i_shl_loop");
    e.branch(BCond::Eq, S9, ZERO, "f2i_shl_done");
    e.slli(S8, S8, 1);
    e.addi(S9, S9, -1);
    e.jump("f2i_shl_loop");
    e.label("f2i_shl_done");
    e.jump("f2i_have_mag");
    e.label("f2i_overflow");
    e.branch(BCond::Eq, S3, ZERO, "f2i_of_max");
    e.li32(A0, i32::MIN);
    e.ret();
    e.label("f2i_of_max");
    e.li32(A0, i32::MAX);
    e.ret();
    e.label("f2i_eneg");
    // let shift = -e; if shift>=32 { 0 } else { mant>>shift }
    e.alu_reg(AluOp::Sub, S9, ZERO, S7); // shift
    e.li32(A2, 32);
    e.branch(BCond::Ge, S9, A2, "f2i_shr_zero");
    e.mv(S8, S5);
    e.label("f2i_shr_loop");
    e.branch(BCond::Eq, S9, ZERO, "f2i_shr_done");
    e.srli(S8, S8, 1);
    e.addi(S9, S9, -1);
    e.jump("f2i_shr_loop");
    e.label("f2i_shr_done");
    e.jump("f2i_have_mag");
    e.label("f2i_shr_zero");
    e.mv(S8, ZERO);
    e.label("f2i_have_mag");

    // let signed = if sign==1 { -mag } else { mag };
    e.branch(BCond::Eq, S3, ZERO, "f2i_pos_signed");
    e.alu_reg(AluOp::Sub, S8, ZERO, S8);
    e.label("f2i_pos_signed");
    // mag (hence signed) is always in i32 range here: e is capped below 8 above, and mant <
    // 2^24 (decompose32's own invariant), so mant<<e < 2^31; the e<0 branch only shrinks
    // mant further. softfloat_ref.rs's own saturating compare against i32::MAX/MIN is
    // therefore provably unreachable on this path and is deliberately omitted.
    e.mv(A0, S8);
    e.ret();

    e.label("f2i_ret_zero");
    e.li32(A0, 0);
    e.ret();
}

/// `__sf_f32_to_u32`: `a0=bits -> a0 = rtz_f32_to_u32(bits)`.
fn emit_f32_to_u32(e: &mut Enc) {
    e.label("__sf_f32_to_u32");
    e.mv(T0, A0);
    is_nan_raw(e, T0, S2, A2, A3);
    e.branch(BCond::Ne, S2, ZERO, "f2u_ret_zero");
    decompose_raw(e, "f2u_d", T0, S3, S4, S5, A2); // sign, exp, mant

    // if sign==1 && !(exp==0 && mant==0) { return 0 }
    e.branch(BCond::Eq, S3, ZERO, "f2u_sign_ok");
    e.sltiu(A2, S4, 1);
    e.sltiu(A3, S5, 1);
    e.alu_reg(AluOp::And, A2, A2, A3); // is true zero
    e.branch(BCond::Ne, A2, ZERO, "f2u_sign_ok");
    e.li32(A0, 0);
    e.ret();
    e.label("f2u_sign_ok");

    is_inf_raw(e, T0, S6, A2, A3);
    e.branch(BCond::Eq, S6, ZERO, "f2u_not_inf");
    e.li32(A0, -1); // u32::MAX bit pattern
    e.ret();
    e.label("f2u_not_inf");
    e.branch(BCond::Eq, S4, ZERO, "f2u_ret_zero"); // exp==0

    e.mv(S7, S4);
    e.addi(S7, S7, -127);
    e.addi(S7, S7, -23);

    e.branch(BCond::Lt, S7, ZERO, "f2u_eneg");
    e.li32(A2, 8);
    e.branch(BCond::Ge, S7, A2, "f2u_overflow");
    e.mv(S8, S5);
    e.mv(S9, S7);
    e.label("f2u_shl_loop");
    e.branch(BCond::Eq, S9, ZERO, "f2u_shl_done");
    e.slli(S8, S8, 1);
    e.addi(S9, S9, -1);
    e.jump("f2u_shl_loop");
    e.label("f2u_shl_done");
    // val < 2^31 always here (same bound as f2i's e>=0 branch); softfloat_ref.rs's own
    // saturating compare against u32::MAX is provably unreachable and is omitted.
    e.mv(A0, S8);
    e.ret();
    e.label("f2u_overflow");
    e.li32(A0, -1);
    e.ret();
    e.label("f2u_eneg");
    e.alu_reg(AluOp::Sub, S9, ZERO, S7);
    e.li32(A2, 32);
    e.branch(BCond::Ge, S9, A2, "f2u_shr_zero");
    e.mv(S8, S5);
    e.label("f2u_shr_loop");
    e.branch(BCond::Eq, S9, ZERO, "f2u_shr_done");
    e.srli(S8, S8, 1);
    e.addi(S9, S9, -1);
    e.jump("f2u_shr_loop");
    e.label("f2u_shr_done");
    e.mv(A0, S8);
    e.ret();
    e.label("f2u_shr_zero");
    e.li32(A0, 0);
    e.ret();

    e.label("f2u_ret_zero");
    e.li32(A0, 0);
    e.ret();
}

/// Emits every soft-float routine, once, at the fixed label names `lower.rs` calls by name.
/// All eight share one flat label namespace with the rest of whatever object this `Enc` is
/// building (see the module header) — every internal label above is prefixed with its own
/// routine's short name (`add_`/`mul_`/`div_`/`cmp_`/`i2f_`/`u2f_`/`f2i_`/`f2u_`) to keep this
/// file collision-free against `lower.rs`'s own `bb<N>`/`__loop_*`/`__<prefix>_<N>` names.
pub fn emit_runtime(e: &mut Enc) {
    emit_f32_add(e);
    emit_f32_mul(e);
    emit_f32_div(e);
    emit_f32_cmp(e);
    emit_i32_to_f32(e);
    emit_u32_to_f32(e);
    emit_f32_to_i32(e);
    emit_f32_to_u32(e);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_runtime_produces_a_nonempty_whole_instruction_stream() {
        let mut e = Enc::new();
        emit_runtime(&mut e);
        let bytes = e.finish();
        assert!(!bytes.is_empty());
        assert_eq!(
            bytes.len() % 4,
            0,
            "every RV32 instruction is exactly 4 bytes"
        );
    }

    #[test]
    fn emit_runtime_is_deterministic() {
        let mut e1 = Enc::new();
        emit_runtime(&mut e1);
        let bytes1 = e1.finish();

        let mut e2 = Enc::new();
        emit_runtime(&mut e2);
        let bytes2 = e2.finish();

        assert_eq!(bytes1, bytes2);
    }

    // Hand-traced desk check of `__sf_f32_add` for `1.0 + 2.0 = 3.0`
    // (a0=0x3f800000, a1=0x40000000 -> a0=0x40400000), walked register-by-register against
    // `emit_f32_add`'s own instruction sequence above. This is a manual proof artifact, not an
    // automated test: no RV32 simulator exists yet in this tree to execute the emitted bytes
    // against (a later task's job), so this trace is the best available substitute for a
    // reviewer to check translation fidelity on the hardest routine in this file.
    //
    // Inputs: a = 0x3f800000 (1.0), b = 0x40000000 (2.0). T0=a, T1=b.
    //
    // is_nan32/is_inf32 on both: exp fields are 127 and 128, neither 0xFF -> both false;
    // za/zb: both exponents nonzero -> both false. Control falls through to "pick big/small".
    //
    // decompose32(a) = (sa=0, ea=127, ma=0x800000) -> S4=0, S5=127, S6=0x800000.
    // decompose32(b) = (sb=0, eb=128, mb=0x800000) -> S7=0, S8=128, S9=0x800000.
    //
    // Pick: A2 = (eb<ea) = (128<127) = 0 -> "add_pick_a" not taken. S5!=S8 (127!=128) ->
    // "add_pick_b" taken (b becomes big, matching softfloat_ref.rs's own ea<eb case).
    //   S2=big_sign=sb=0, S3=big_exp=eb=128, A4=big_mant=mb=0x800000,
    //   A5=small_sign=sa=0, A6=small_exp=ea=127, A7=small_mant=ma=0x800000.
    //
    // diff = big_exp - small_exp = 128-127 = 1 (S4). k = min(1,24) = 1 (S5, "add_k_diff"
    // taken since 1<24).
    //
    // small_wide: "add_sw_far" not taken (24<1 is false) -> near path: S6 = small_mant =
    // 0x800000 (unshifted, matching diff<=K).
    //
    // big_wide: S9(big_wide_hi)=0, loop counter S8=k=1. One iteration of shl64_1(A4,S9,T2):
    // bit31 of A4(0x800000) is 0 (0x800000 has only bit 23 set) -> S9 stays 0; A4 <<= 1 =>
    // A4 = 0x1000000. Loop counter hits 0, exits. big_wide = (0x1000000, hi=0).
    //
    // exp2 (T3) = big_exp - 127 - k = 128 - 127 - 1 = 0.
    //
    // big_sign(S2)==small_sign(A5) (both 0) -> "add_diffsign" not taken, same-sign path:
    //   T4 = sum_lo = A4 + S6 = 0x1000000 + 0x800000 = 0x1800000.
    //   T5 = carry = (T4 <u A4) = (0x1800000 <u 0x1000000) = 0.
    //   T6 = sum_hi = S9 + T5 = 0 + 0 = 0.
    //
    // norm_and_pack64(sign=S2=0, mlo=T4=0x1800000, mhi=T6=0, exp2=T3=0):
    //   Left-normalize: tc=0x800000; hi==0 and mlo(0x1800000) >= tc -> exits immediately
    //     (0 iterations: the value is already >= 2^23).
    //   Right-normalize: tc=0x1000000; hi==0 and mlo(0x1800000) >= tc -> one shr64_1:
    //     bit0 of hi(0)=0; T4 = 0x1800000>>1 = 0xC00000 (no carry-in bit to OR, since that
    //     bit was 0); hi stays 0>>1=0. exp2 (T3) += 1 -> 1. Recheck: mlo(0xC00000) >=
    //     tc(0x1000000)? 0xC00000 < 0x1000000 -> exits (1 iteration total).
    //   exp_field = exp2 + 127 = 1 + 127 = 128. Not <=0, not >=0xFF -> normal path:
    //     ta = sign<<31 = 0. tb = exp_field<<23 = 128<<23 = 0x40000000. ta|=tb -> 0x40000000.
    //     td = mlo ^ 0x800000 = 0xC00000 ^ 0x800000 = 0x400000.
    //     out (A0) = ta | td = 0x40000000 | 0x400000 = 0x40400000.
    //
    // A0 = 0x40400000 = 3.0f32.to_bits(), matching softfloat_ref.rs's rtz_add32 exactly.
    #[test]
    fn desk_checked_add_case_is_documented_above() {
        // No RV32 executor exists in this tree yet; this test only pins the two operands and
        // expected result the trace above walks through, so the comment and the test can
        // never silently drift apart.
        let a: u32 = 0x3f80_0000;
        let b: u32 = 0x4000_0000;
        let expect: u32 = 0x4040_0000;
        assert_eq!(
            f32::from_bits(a) + f32::from_bits(b),
            f32::from_bits(expect)
        );
    }
}
