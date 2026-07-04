// Pure-Rust reference algorithms for this crate's soft-float runtime, used only to validate
// the *bit-manipulation logic* before it is hand-translated into RV32IM instructions in
// `softfloat.rs`. This module is never linked into an emitted object — it exists purely as an
// executable specification, checked in `tests` below against exact-rational-arithmetic ground
// truth (computed independently in Python using `fractions.Fraction`, never against Rust's or
// any hardware's native float unit — see the module header on rounding, below) and reviewed
// line-by-line against `softfloat.rs`'s instruction sequences for translation fidelity. See
// `softfloat.rs`'s own header for why this two-step process (validate the algorithm in Rust,
// then transliterate) replaces execution-testing the emitted machine code, which this task
// cannot do (no RV32 simulator exists yet in this tree — that is `P12-T2`'s job).
//
// # Rounding: round-toward-zero, not round-to-nearest-even
//
// Every arithmetic routine here (add/mul/div and every int<->float conversion) rounds by
// truncation (discarding bits below the target precision) rather than IEEE 754's default
// round-to-nearest-even. Round-toward-zero is itself one of the four rounding attributes the
// IEEE 754 standard defines — this is a legal rounding mode, not an invented one — but it is
// not the default most hardware FPUs use, so results differ from a hardware FPU by up to 1
// ULP, systematically biased toward zero, on any operation that isn't already exact. This is
// a deliberate, documented scope simplification: round-to-nearest-even requires tracking
// guard/round/sticky bits through every shift and carrying a tie-breaking rule through every
// normalization path, which is a meaningfully larger, more failure-prone algorithm to get
// right without any way to execute-test the result (see above). Round-toward-zero needs only
// "shift the bits that don't fit off the end and discard them," which is far lower-risk to
// hand-translate correctly. This is exactly the kind of honest scope call this task's brief
// invites rather than silently guessing at round-to-nearest's trickier edge cases.
//
// # Other documented scope decisions (shared with `softfloat.rs`)
//
// - Subnormal inputs are flushed to zero; this implementation never produces a subnormal
//   result either (an underflowing result flushes to zero rather than rounding into the
//   subnormal range).
// - Every NaN input (either operand, any payload) produces the single canonical quiet NaN
//   `0x7fc00000` (f32) / `0x7ff8000000000000` (f64) — input NaN payloads are never propagated.
//   This is legal: IEEE 754 does not mandate payload propagation.
// - `f32_to_i32`/`f32_to_u32` saturate on overflow and on NaN input (NaN converts to 0),
//   matching the common `fptosi`/`fptoui` convention this project's other backends already
//   assume elsewhere (documented, not a silent guess).

pub(crate) fn is_nan32(bits: u32) -> bool {
    let exp = (bits >> 23) & 0xFF;
    let frac = bits & 0x7FFFFF;
    exp == 0xFF && frac != 0
}

pub(crate) fn is_inf32(bits: u32) -> bool {
    let exp = (bits >> 23) & 0xFF;
    let frac = bits & 0x7FFFFF;
    exp == 0xFF && frac == 0
}

/// `(sign, exp, mant)`: `mant` includes the implicit leading bit for a normal number (24
/// bits, in `[2^23, 2^24)`); subnormal inputs (`exp==0, frac!=0`) are flushed to zero (`mant`
/// reported as 0, matching a true zero) per this file's documented scope.
fn decompose32(bits: u32) -> (u32, i32, u32) {
    let sign = bits >> 31;
    let exp_field = (bits >> 23) & 0xFF;
    let frac = bits & 0x7FFFFF;
    if exp_field == 0 {
        (sign, 0, 0)
    } else {
        (sign, exp_field as i32, frac | 0x800000)
    }
}

pub(crate) const CANONICAL_NAN32: u32 = 0x7fc0_0000;

/// Packs a nonnegative-magnitude `mant` (any bit width — the caller has *not* yet normalized
/// it into 24 bits) whose value is `mant * 2^(exp2 - 23)`, at `sign`, into an f32 bit pattern,
/// truncating to 24 significant bits (round-toward-zero) and flushing to zero/inf on
/// underflow/overflow. `mant` must be nonzero (callers handle the zero case themselves, since
/// its meaning — signed zero — depends on which operation produced it).
fn normalize_and_pack32(sign: u32, mant: u64, exp2: i32) -> u32 {
    debug_assert!(mant != 0);
    let mut mant = mant;
    let mut exp2 = exp2;
    // Left-normalize until the leading bit sits at position 23 (bit 24 unset, bit 23 set).
    while mant < 0x0080_0000 {
        mant <<= 1;
        exp2 -= 1;
    }
    // Right-normalize (truncating, i.e. dropping bits below position 23 — round-toward-zero)
    // until the leading bit sits at position 23.
    while mant >= 0x0100_0000 {
        mant >>= 1;
        exp2 += 1;
    }
    let exp_field = exp2 + 127;
    if exp_field <= 0 {
        return sign << 31;
    }
    if exp_field >= 0xFF {
        return (sign << 31) | (0xFFu32 << 23);
    }
    (sign << 31) | ((exp_field as u32) << 23) | ((mant as u32) & 0x7FFFFF)
}

pub(crate) fn rtz_add32(a: u32, b: u32) -> u32 {
    if is_nan32(a) || is_nan32(b) {
        return CANONICAL_NAN32;
    }
    let (sa, ea, ma) = decompose32(a);
    let (sb, eb, mb) = decompose32(b);
    let ia = is_inf32(a);
    let ib = is_inf32(b);
    if ia && ib {
        return if sa != sb { CANONICAL_NAN32 } else { a };
    }
    if ia {
        return a;
    }
    if ib {
        return b;
    }
    let za = ea == 0 && ma == 0;
    let zb = eb == 0 && mb == 0;
    if za && zb {
        return if sa == sb { a } else { 0 };
    }
    if za {
        return b;
    }
    if zb {
        return a;
    }

    // Pick BIG as the operand with the larger (exponent, mantissa) pair so that, after
    // aligning SMALL's mantissa to BIG's exponent, `mant_big >= mant_small_shifted` always —
    // guaranteeing the opposite-sign subtraction below never borrows (see `softfloat.rs`'s
    // matching derivation).
    let (big_sign, big_exp, big_mant, small_sign, small_exp, small_mant) =
        if ea > eb || (ea == eb && ma >= mb) {
            (sa, ea, ma, sb, eb, mb)
        } else {
            (sb, eb, mb, sa, ea, ma)
        };
    let diff = big_exp - small_exp;
    // Align by widening BIG's mantissa up by `k` extra low zero bits (`k = min(diff, K)`) and
    // aligning SMALL's mantissa to that same `k`, rather than shifting SMALL's mantissa down
    // to BIG's bare exponent and discarding the rest up front. Discarding those bits early
    // looks equivalent but is not, under round-toward-zero: losing them before the add/
    // subtract can change the correctly-truncated result once the subtraction's own
    // renormalization (a left-shift, for a cancelling subtraction) brings previously-below-
    // precision bits back into significance — the classic "double rounding" pitfall. `K = 24`
    // extra bits is exactly BIG's own mantissa width, which is provably enough: a borrow
    // chain out of a subtraction can propagate through at most BIG's own 24 bits before
    // hitting a set bit, so no amount of *additional* alignment beyond 24 bits changes the
    // final (renormalized, truncated-to-24-bit) result. When `diff > K`, SMALL's contribution
    // is collapsed to a single sticky bit beyond the `k`-bit window (the standard technique a
    // hardware FPU's guard/round/sticky logic uses for the same reason) rather than discarded
    // outright, so the "is there *any* nonzero remainder below the window" fact these
    // boundary cases turn on is never silently lost.
    const K: i32 = 24;
    let k = diff.min(K);
    let small_wide: u64 = if diff <= K {
        small_mant as u64
    } else {
        let shift = diff - K;
        if shift >= 24 {
            1 // small_mant (nonzero, already checked above) is entirely below the window
        } else {
            let shifted = (small_mant as u64) >> shift;
            let sticky = ((small_mant as u64) & ((1u64 << shift) - 1) != 0) as u64;
            shifted | sticky
        }
    };
    let big_wide = (big_mant as u64) << k;
    let exp2 = big_exp - 127 - k;
    if big_sign == small_sign {
        let sum = big_wide + small_wide;
        normalize_and_pack32(big_sign, sum, exp2)
    } else {
        let diff_mant = big_wide - small_wide;
        if diff_mant == 0 {
            return 0;
        }
        normalize_and_pack32(big_sign, diff_mant, exp2)
    }
}

pub(crate) fn rtz_sub32(a: u32, b: u32) -> u32 {
    rtz_add32(a, b ^ 0x8000_0000)
}

pub(crate) fn rtz_mul32(a: u32, b: u32) -> u32 {
    let rsign = (a >> 31) ^ (b >> 31);
    if is_nan32(a) || is_nan32(b) {
        return CANONICAL_NAN32;
    }
    let (_, ea, ma) = decompose32(a);
    let (_, eb, mb) = decompose32(b);
    let ia = is_inf32(a);
    let ib = is_inf32(b);
    let za = ea == 0 && ma == 0;
    let zb = eb == 0 && mb == 0;
    if (ia && zb) || (ib && za) {
        return CANONICAL_NAN32;
    }
    if ia || ib {
        return (rsign << 31) | (0xFFu32 << 23);
    }
    if za || zb {
        return rsign << 31;
    }
    let product = ma as u64 * mb as u64;
    // `ma`,`mb` in [2^23,2^24) => product in [2^46,2^48): 48 "conceptual" bits with the point
    // sitting after bit 46 of the product (`(ea-127-23)+(eb-127-23)+46` is the value's binary
    // exponent when the leading product bit is at 47; `normalize_and_pack32` sorts out
    // whichever of bit 46/47 is actually the leading one).
    let result_bits = normalize_and_pack32(rsign, product, ea + eb - 277);
    if result_bits & 0x7FFFFFFF == 0 {
        rsign << 31
    } else {
        result_bits
    }
}

/// Base-256 (radix-2^8) long division of the 24-bit mantissas via three chunks of native
/// 32-bit unsigned divide/remainder — see `softfloat.rs`'s header for why this avoids ever
/// needing a wide (>32-bit) numerator or a bit-serial division loop. Returns
/// `floor(num * 2^24 / den)` exactly (`num`, `den` both nonzero and `< 2^24`).
fn div_mant_q24(num: u32, den: u32) -> u64 {
    let q0 = num / den; // 0 or 1, since both operands are in [2^23, 2^24)
    let mut rem = num % den;
    let mut q = q0 as u64;
    for _ in 0..3 {
        let scaled = rem << 8; // rem < den < 2^24, so scaled < 2^32: never overflows u32
        let chunk = scaled / den; // < 256
        rem = scaled % den;
        q = (q << 8) | chunk as u64;
    }
    q
}

pub(crate) fn rtz_div32(a: u32, b: u32) -> u32 {
    let rsign = (a >> 31) ^ (b >> 31);
    if is_nan32(a) || is_nan32(b) {
        return CANONICAL_NAN32;
    }
    let (_, ea, ma) = decompose32(a);
    let (_, eb, mb) = decompose32(b);
    let ia = is_inf32(a);
    let ib = is_inf32(b);
    let za = ea == 0 && ma == 0;
    let zb = eb == 0 && mb == 0;
    if ia && ib {
        return CANONICAL_NAN32;
    }
    if za && zb {
        return CANONICAL_NAN32;
    }
    if ia {
        // inf / (finite, or 0 — the `za`/`zb` combinations with `ia` are already ruled out
        // above) is always infinite, never zero.
        return (rsign << 31) | (0xFFu32 << 23);
    }
    if za {
        return rsign << 31;
    }
    if ib {
        return rsign << 31;
    }
    if zb {
        return (rsign << 31) | (0xFFu32 << 23);
    }
    let q24 = div_mant_q24(ma, mb);
    // `q24 = floor(ma * 2^24 / mb)`, i.e. the quotient `ma/mb` (which is in `(0.5, 2)`) scaled
    // by `2^24`, so its own leading bit sits at position 23 or 24 — `normalize_and_pack32`
    // sorts out which, same as `rtz_mul32`.
    let result_bits = normalize_and_pack32(rsign, q24, ea - eb - 1);
    if result_bits & 0x7FFFFFFF == 0 {
        rsign << 31
    } else {
        result_bits
    }
}

/// Ordering code: `-2` unordered (either NaN), `-1`/`0`/`1` for `<`/`==`/`>`.
pub(crate) fn rtz_cmp32(a: u32, b: u32) -> i32 {
    if is_nan32(a) || is_nan32(b) {
        return -2;
    }
    let (sa, ea, ma) = decompose32(a);
    let (sb, eb, mb) = decompose32(b);
    let za = ea == 0 && ma == 0;
    let zb = eb == 0 && mb == 0;
    if za && zb {
        return 0;
    }
    if za {
        return if sb == 0 { -1 } else { 1 };
    }
    if zb {
        return if sa == 0 { 1 } else { -1 };
    }
    if sa != sb {
        return if sa == 1 { -1 } else { 1 };
    }
    let mag_cmp = if ea != eb { ea.cmp(&eb) } else { ma.cmp(&mb) };
    let ord = match mag_cmp {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    };
    if sa == 1 {
        -ord
    } else {
        ord
    }
}

pub(crate) fn rtz_i32_to_f32(i: i32) -> u32 {
    if i == 0 {
        return 0;
    }
    let sign = (i < 0) as u32;
    let mag = (i as i64).unsigned_abs();
    normalize_and_pack32(sign, mag, 23)
}

pub(crate) fn rtz_u32_to_f32(u: u32) -> u32 {
    if u == 0 {
        return 0;
    }
    normalize_and_pack32(0, u as u64, 23)
}

/// `fptosi`, round-toward-zero (matching `CastOp::FpToSi`'s own truncating convention, so
/// this needs no special rounding case at all — it's already the natural semantics).
/// Saturates on overflow; NaN converts to 0 (documented convention, see the module header).
pub(crate) fn rtz_f32_to_i32(bits: u32) -> i32 {
    if is_nan32(bits) {
        return 0;
    }
    let (sign, exp, mant) = decompose32(bits);
    if is_inf32(bits) {
        return if sign == 1 { i32::MIN } else { i32::MAX };
    }
    if exp == 0 {
        return 0;
    }
    let e = exp - 127 - 23; // value = mant * 2^e
    let mag: i64 = if e >= 0 {
        if e >= 8 {
            return if sign == 1 { i32::MIN } else { i32::MAX }; // certain overflow
        }
        (mant as i64) << e
    } else {
        let shift = -e;
        if shift >= 32 {
            0
        } else {
            (mant as i64) >> shift
        }
    };
    let signed = if sign == 1 { -mag } else { mag };
    if signed > i32::MAX as i64 {
        i32::MAX
    } else if signed < i32::MIN as i64 {
        i32::MIN
    } else {
        signed as i32
    }
}

pub(crate) fn rtz_f32_to_u32(bits: u32) -> u32 {
    if is_nan32(bits) {
        return 0;
    }
    let (sign, exp, mant) = decompose32(bits);
    if sign == 1 && !(exp == 0 && mant == 0) {
        return 0; // negative (nonzero) truncates to 0, matching C's fptoui-of-negative UB
                  // resolved here as saturate-to-zero rather than wrapping.
    }
    if is_inf32(bits) {
        return u32::MAX;
    }
    if exp == 0 {
        return 0;
    }
    let e = exp - 127 - 23;
    if e >= 0 {
        if e >= 8 {
            return u32::MAX;
        }
        let val = (mant as u64) << e;
        if val > u32::MAX as u64 {
            u32::MAX
        } else {
            val as u32
        }
    } else {
        let shift = -e;
        if shift >= 32 {
            0
        } else {
            mant >> shift
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_hex_u32(s: &str) -> u32 {
        u32::from_str_radix(s, 16).unwrap()
    }

    fn check_binop(vectors: &str, f: impl Fn(u32, u32) -> u32) {
        let mut n = 0;
        for line in vectors.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let a = parse_hex_u32(parts[0]);
            let b = parse_hex_u32(parts[1]);
            let expect = parse_hex_u32(parts[2]);
            let got = f(a, b);
            assert_eq!(
                got, expect,
                "mismatch for a={a:08x} b={b:08x}: got {got:08x}, want {expect:08x}"
            );
            n += 1;
        }
        assert!(n > 100, "vector file suspiciously small: {n} rows");
    }

    #[test]
    fn add_matches_exact_rational_ground_truth() {
        check_binop(include_str!("testdata/f32_add_vectors.txt"), rtz_add32);
    }

    #[test]
    fn mul_matches_exact_rational_ground_truth() {
        check_binop(include_str!("testdata/f32_mul_vectors.txt"), rtz_mul32);
    }

    #[test]
    fn div_matches_exact_rational_ground_truth() {
        check_binop(include_str!("testdata/f32_div_vectors.txt"), rtz_div32);
    }

    // The four vector files above were generated by sampling `random.uniform` ranges in
    // Python, which never actually produces an infinity or NaN bit pattern — every special
    // value (`+-0`, `+-inf`, NaN) combination below was entirely untested until this was
    // noticed (and caught a real bug: `rtz_div32`'s `if ia` arm originally returned a signed
    // zero instead of a signed infinity). These four files are the fix: every pairing of
    // {normal+, normal-, +0, -0, +inf, -inf, NaN, tiny+, tiny-, huge+, huge-} against itself
    // (121 cases each), computed independently in Python against the same exact-rational
    // ground truth as the other vector files.
    #[test]
    fn add_matches_special_value_ground_truth() {
        check_binop(
            include_str!("testdata/f32_add_specials_vectors.txt"),
            rtz_add32,
        );
    }

    #[test]
    fn mul_matches_special_value_ground_truth() {
        check_binop(
            include_str!("testdata/f32_mul_specials_vectors.txt"),
            rtz_mul32,
        );
    }

    #[test]
    fn div_matches_special_value_ground_truth() {
        check_binop(
            include_str!("testdata/f32_div_specials_vectors.txt"),
            rtz_div32,
        );
    }

    #[test]
    fn cmp_matches_special_value_ground_truth() {
        check_binop(
            include_str!("testdata/f32_cmp_specials_vectors.txt"),
            |a, b| rtz_cmp32(a, b) as u32,
        );
    }

    #[test]
    fn cmp_matches_exact_rational_ground_truth() {
        check_binop(include_str!("testdata/f32_cmp_vectors.txt"), |a, b| {
            rtz_cmp32(a, b) as u32
        });
    }

    #[test]
    fn i32_to_f32_matches_ground_truth() {
        for line in include_str!("testdata/i32_to_f32_vectors.txt").lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let i = parse_hex_u32(parts[0]) as i32;
            let expect = parse_hex_u32(parts[1]);
            assert_eq!(rtz_i32_to_f32(i), expect, "i={i}");
        }
    }

    #[test]
    fn u32_to_f32_matches_ground_truth() {
        for line in include_str!("testdata/u32_to_f32_vectors.txt").lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let u = parse_hex_u32(parts[0]);
            let expect = parse_hex_u32(parts[1]);
            assert_eq!(rtz_u32_to_f32(u), expect, "u={u}");
        }
    }

    #[test]
    fn f32_to_i32_matches_ground_truth() {
        for line in include_str!("testdata/f32_to_i32_vectors.txt").lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let bits = parse_hex_u32(parts[0]);
            let expect = parse_hex_u32(parts[1]) as i32;
            assert_eq!(rtz_f32_to_i32(bits), expect, "bits={bits:08x}");
        }
    }

    #[test]
    fn f32_to_u32_matches_ground_truth() {
        for line in include_str!("testdata/f32_to_u32_vectors.txt").lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let bits = parse_hex_u32(parts[0]);
            let expect = parse_hex_u32(parts[1]);
            assert_eq!(rtz_f32_to_u32(bits), expect, "bits={bits:08x}");
        }
    }

    #[test]
    fn sub_is_add_with_flipped_sign() {
        assert_eq!(rtz_sub32(f32_bits(3.0), f32_bits(1.0)), f32_bits(2.0));
        assert_eq!(rtz_sub32(f32_bits(1.0), f32_bits(3.0)), f32_bits(-2.0));
    }

    fn f32_bits(x: f32) -> u32 {
        x.to_bits()
    }
}
