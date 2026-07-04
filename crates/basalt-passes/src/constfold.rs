// Constant folding: replaces any `Bin`/`ICmp`/`FCmp`/`Cast` instruction whose operands are
// all compile-time constants with a plain `ConstInt`/`ConstFloat`, in place.
//
// Out of scope, deliberately: `Select`, `Load`, `Store`, `Phi`, GPU intrinsics, atomics, and
// anything else not in the list above. Folding a constant-condition `select` or propagating a
// folded value through a copy is a simplification pass's job, not this one's.
//
// # Why one forward pass is enough
//
// A function's instruction arena is populated in program order and every instruction only
// ever references earlier instructions (see `ir.rs`'s header) — there are no forward
// references. Walking `insts` once, front to back, and mutating each instruction's `Op` in
// place the moment it turns out to be foldable therefore already sees every operand in its
// final, most-folded form: by the time instruction `k` is examined, any earlier instruction
// `j < k` that could fold already has. No fixed-point iteration is needed. A
// `BTreeMap<InstId, ConstVal>` is threaded through the walk recording, for every instruction
// visited so far, the constant it evaluates to (whether it was a literal in the input or
// folded just now); resolving an operand is exactly one map lookup.
//
// Each instruction keeps its `InstId` and its declared `Ty` — only `Op` changes — so nothing
// downstream ever needs renumbering.
//
// # Integer width and the div/rem-by-zero rule
//
// An integer constant is stored as an `i64` canonicalized the same way the rest of the tree
// already relies on (see e.g. `basalt-sema`'s lowering pass using `ConstInt(-1)` as an
// all-ones pattern regardless of the value's declared width): the low N bits (N = the
// scalar's bit width) hold the value, sign-extended up through the rest of the `i64`. Folding
// re-derives this canonical form after every operation by masking to N bits and
// sign-extending back.
//
// `div`/`rem` are folded as **signed** operations. BIR's `Bin` carries no signed/unsigned
// distinction for these two ops (`ir.rs`'s `BinOp` has one `Div` and one `Rem`, not a
// signed/unsigned pair), and the x86 oracle backend's own header comment documents picking
// signed (`idiv`) as the one interpretation for exactly this reason — this pass matches that
// documented convention rather than inventing a different one.
//
// A divisor that folds to zero is never folded through: real division by zero is a hardware
// fault (`idiv` raises `#DE`), not a value, so a compile-time fold must leave the instruction
// exactly as it found it rather than inventing a placeholder or panicking. The one other
// native-divide trap case, `MIN / -1` at a type's own full width, is not given the same
// treatment — it folds to the two's-complement wraparound result (computed via a widening
// `i128` intermediate, so folding itself never panics), matching ordinary wrapping-arithmetic
// behavior rather than being singled out as unsafe to fold; only a zero divisor is special.
//
// Shift amounts are interpreted against the operand's own declared bit width: an amount at or
// beyond that width saturates (every bit shifted out) rather than wrapping around modulo some
// narrower hardware count register. BIR has no documented cross-backend convention for an
// out-of-range shift (unlike `div`/`rem`'s signedness, this is not called out anywhere as a
// deliberate backend choice to match), so this pass uses the portable, spec-clean reading
// rather than guessing at any one backend's native shift instruction's own masking quirk.
//
// # Casts
//
// `trunc`/`zext`/`sext` reinterpret an integer's bit pattern at a new width (dropping bits,
// zero-filling, or sign-filling respectively). `fpext` never changes the numeric value —
// widening a float's precision is exact by definition, so it degenerates to a copy. `fptrunc`
// rounds to the destination precision (round-to-nearest, ties-to-even), including the `f16`
// case, which this crate hand-rolls (bit-level IEEE 754 binary16 <-> binary64 conversion,
// converting directly rather than by double-rounding through `f32`) since nothing in the
// dependency graph already provides it. `fptosi`/`fptoui` truncate toward zero and saturate at
// the destination width's representable range (NaN folds to zero) — the same total,
// deterministic behavior Rust's own `as` float-to-int casts define, just generalized to an
// arbitrary bit width. `sitofp`/`uitofp` convert then round to the destination precision the
// same way `fptrunc` does. `bitcast` reinterprets the raw bits of a same-width source as the
// destination type; a width mismatch (never valid BIR) is left unfolded rather than guessed at.

use std::collections::BTreeMap;

use basalt_bir::{BinOp, CastOp, FCmpPred, ICmpPred, InstId, Module, Op, Scalar, Ty, ValRef};

/// A resolved compile-time value. Kept separate from `Op::ConstInt`/`Op::ConstFloat` only
/// because it is convenient to carry around before deciding which `Op` variant to write back.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConstVal {
    Int(i64),
    Float(f64),
}

/// Folds every foldable `Bin`/`ICmp`/`FCmp`/`Cast` instruction in `module` to a plain constant,
/// in place per function. See the module header for exactly what folds and what does not.
pub fn constant_fold(module: &Module) -> Module {
    let mut out = module.clone();
    for f in &mut out.funcs {
        fold_function(f);
    }
    out
}

fn fold_function(f: &mut basalt_bir::Function) {
    let mut consts: BTreeMap<InstId, ConstVal> = BTreeMap::new();

    for idx in 0..f.insts.len() {
        let id = InstId(idx as u32);
        let ty = f.insts[idx].ty;

        match f.insts[idx].op {
            Op::ConstInt(v) => {
                consts.insert(id, ConstVal::Int(v));
                continue;
            }
            Op::ConstFloat(v) => {
                consts.insert(id, ConstVal::Float(v));
                continue;
            }
            _ => {}
        }

        // Cheap to clone: `Op` is small, and every variant this pass touches is a handful of
        // scalars/`ValRef`s. Cloning sidesteps holding a borrow of `f.insts[idx]` across the
        // lookups into `f.insts`/`consts` that resolving its operands needs.
        let folded = match f.insts[idx].op.clone() {
            Op::Bin(op, a, b) => fold_bin(op, ty, resolve(a, &consts), resolve(b, &consts)),
            Op::ICmp(pred, oty, a, b) => {
                fold_icmp(pred, oty, resolve(a, &consts), resolve(b, &consts))
            }
            Op::FCmp(pred, oty, a, b) => {
                fold_fcmp(pred, oty, resolve(a, &consts), resolve(b, &consts))
            }
            Op::Cast(cop, src_ty, v) => fold_cast(cop, src_ty, ty, resolve(v, &consts)),
            _ => None,
        };

        if let Some(c) = folded {
            f.insts[idx].op = match c {
                ConstVal::Int(v) => Op::ConstInt(v),
                ConstVal::Float(v) => Op::ConstFloat(v),
            };
            consts.insert(id, c);
        }
    }
}

fn resolve(v: ValRef, consts: &BTreeMap<InstId, ConstVal>) -> Option<ConstVal> {
    match v {
        ValRef::Param(_) => None,
        ValRef::Val(id) => consts.get(&id).copied(),
    }
}

// --- width helpers -----------------------------------------------------------------------

fn int_width(s: Scalar) -> Option<u32> {
    match s {
        Scalar::I1 => Some(1),
        Scalar::I8 => Some(8),
        Scalar::I16 => Some(16),
        Scalar::I32 => Some(32),
        Scalar::I64 => Some(64),
        Scalar::F16 | Scalar::F32 | Scalar::F64 => None,
    }
}

fn is_float_scalar(s: Scalar) -> bool {
    matches!(s, Scalar::F16 | Scalar::F32 | Scalar::F64)
}

fn scalar_of(ty: Ty) -> Option<Scalar> {
    match ty {
        Ty::Scalar(s) => Some(s),
        _ => None,
    }
}

fn scalar_int_width(ty: Ty) -> Option<u32> {
    scalar_of(ty).and_then(int_width)
}

/// The width `icmp` interprets its operands at: a scalar integer's own width, or 64 for a
/// pointer (BIR's synthesized pointer constants are plain 64-bit integers under an opaque
/// `Ty::Ptr` tag — see `ssa.rs`'s header and `basalt-sema`'s lowering pass).
fn icmp_width(ty: Ty) -> Option<u32> {
    match ty {
        Ty::Scalar(s) => int_width(s),
        Ty::Ptr(_) => Some(64),
        _ => None,
    }
}

/// Every bit width this pass ever needs to reinterpret raw bits at, integer or float.
fn bit_width_any(s: Scalar) -> u32 {
    match s {
        Scalar::I1 => 1,
        Scalar::I8 => 8,
        Scalar::I16 | Scalar::F16 => 16,
        Scalar::I32 | Scalar::F32 => 32,
        Scalar::I64 | Scalar::F64 => 64,
    }
}

fn mask_for(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Sign-extends the low `width` bits of `v` up through the rest of the `i64` — this pass's
/// canonical storage form for an integer constant of any narrower declared width.
fn sign_extend(v: u64, width: u32) -> i64 {
    if width >= 64 {
        return v as i64;
    }
    let mask = mask_for(width);
    let x = v & mask;
    let sign = 1u64 << (width - 1);
    if x & sign != 0 {
        (x | !mask) as i64
    } else {
        x as i64
    }
}

// --- Bin -----------------------------------------------------------------------------------

fn fold_bin(op: BinOp, ty: Ty, a: Option<ConstVal>, b: Option<ConstVal>) -> Option<ConstVal> {
    let (a, b) = (a?, b?);
    match op {
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Rem
        | BinOp::And
        | BinOp::Or
        | BinOp::Xor
        | BinOp::Shl
        | BinOp::Lshr
        | BinOp::Ashr => {
            let width = scalar_int_width(ty)?;
            match (a, b) {
                (ConstVal::Int(araw), ConstVal::Int(braw)) => fold_int_bin(op, width, araw, braw),
                _ => None,
            }
        }
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => match (a, b) {
            (ConstVal::Float(fa), ConstVal::Float(fb)) => {
                Some(ConstVal::Float(fold_float_bin(op, fa, fb)))
            }
            _ => None,
        },
    }
}

fn fold_int_bin(op: BinOp, width: u32, araw: i64, braw: i64) -> Option<ConstVal> {
    let mask = mask_for(width);
    let ua = (araw as u64) & mask;
    let ub = (braw as u64) & mask;

    let result_u = match op {
        BinOp::Add => ua.wrapping_add(ub) & mask,
        BinOp::Sub => ua.wrapping_sub(ub) & mask,
        BinOp::Mul => ua.wrapping_mul(ub) & mask,
        BinOp::And => ua & ub,
        BinOp::Or => ua | ub,
        BinOp::Xor => ua ^ ub,
        BinOp::Shl => {
            if ub >= width as u64 {
                0
            } else {
                ua.wrapping_shl(ub as u32) & mask
            }
        }
        BinOp::Lshr => {
            if ub >= width as u64 {
                0
            } else {
                ua >> (ub as u32)
            }
        }
        BinOp::Ashr => {
            let sa = sign_extend(ua, width);
            let shifted = if ub >= width as u64 {
                if sa < 0 {
                    -1i64
                } else {
                    0i64
                }
            } else {
                sa >> (ub as u32)
            };
            (shifted as u64) & mask
        }
        BinOp::Div | BinOp::Rem => {
            if ub == 0 {
                // Never fold a division/remainder by a constant zero: on real hardware
                // this is a fault, not a value, and folding must not paper over that.
                return None;
            }
            let sa = sign_extend(ua, width) as i128;
            let sb = sign_extend(ub, width) as i128;
            let result = if op == BinOp::Div { sa / sb } else { sa % sb };
            (result as i64 as u64) & mask
        }
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
            unreachable!("float BinOp variants never reach fold_int_bin")
        }
    };

    Some(ConstVal::Int(sign_extend(result_u, width)))
}

fn fold_float_bin(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::FAdd => a + b,
        BinOp::FSub => a - b,
        BinOp::FMul => a * b,
        BinOp::FDiv => a / b,
        // Rust's float `%` is truncated-division remainder (same sign as the dividend),
        // exactly the software `frem` emulation the x86 oracle backend documents using.
        BinOp::FRem => a % b,
        _ => unreachable!("integer BinOp variants never reach fold_float_bin"),
    }
}

// --- ICmp / FCmp -----------------------------------------------------------------------------

fn fold_icmp(pred: ICmpPred, ty: Ty, a: Option<ConstVal>, b: Option<ConstVal>) -> Option<ConstVal> {
    let (a, b) = (a?, b?);
    let width = icmp_width(ty)?;
    let (ConstVal::Int(araw), ConstVal::Int(braw)) = (a, b) else {
        return None;
    };
    let mask = mask_for(width);
    let ua = (araw as u64) & mask;
    let ub = (braw as u64) & mask;
    let sa = sign_extend(ua, width);
    let sb = sign_extend(ub, width);

    let result = match pred {
        ICmpPred::Eq => ua == ub,
        ICmpPred::Ne => ua != ub,
        ICmpPred::Slt => sa < sb,
        ICmpPred::Sle => sa <= sb,
        ICmpPred::Sgt => sa > sb,
        ICmpPred::Sge => sa >= sb,
        ICmpPred::Ult => ua < ub,
        ICmpPred::Ule => ua <= ub,
        ICmpPred::Ugt => ua > ub,
        ICmpPred::Uge => ua >= ub,
    };
    Some(ConstVal::Int(result as i64))
}

fn fold_fcmp(pred: FCmpPred, ty: Ty, a: Option<ConstVal>, b: Option<ConstVal>) -> Option<ConstVal> {
    let (a, b) = (a?, b?);
    scalar_of(ty).filter(|s| is_float_scalar(*s))?;
    let (ConstVal::Float(fa), ConstVal::Float(fb)) = (a, b) else {
        return None;
    };

    let result = match pred {
        FCmpPred::Oeq => fa == fb,
        FCmpPred::One => !fa.is_nan() && !fb.is_nan() && fa != fb,
        FCmpPred::Olt => fa < fb,
        FCmpPred::Ole => fa <= fb,
        FCmpPred::Ogt => fa > fb,
        FCmpPred::Oge => fa >= fb,
        FCmpPred::Ord => !fa.is_nan() && !fb.is_nan(),
        FCmpPred::Uno => fa.is_nan() || fb.is_nan(),
    };
    Some(ConstVal::Int(result as i64))
}

// --- Cast ------------------------------------------------------------------------------------

fn fold_cast(op: CastOp, src_ty: Ty, dest_ty: Ty, v: Option<ConstVal>) -> Option<ConstVal> {
    let v = v?;
    match op {
        CastOp::Trunc => {
            let dest_w = scalar_int_width(dest_ty)?;
            let ConstVal::Int(raw) = v else { return None };
            Some(ConstVal::Int(sign_extend(
                (raw as u64) & mask_for(dest_w),
                dest_w,
            )))
        }
        CastOp::Zext => {
            let src_w = scalar_int_width(src_ty)?;
            let ConstVal::Int(raw) = v else { return None };
            Some(ConstVal::Int(((raw as u64) & mask_for(src_w)) as i64))
        }
        CastOp::Sext => {
            let src_w = scalar_int_width(src_ty)?;
            let ConstVal::Int(raw) = v else { return None };
            Some(ConstVal::Int(sign_extend(
                (raw as u64) & mask_for(src_w),
                src_w,
            )))
        }
        CastOp::FpTrunc => {
            let dest_s = scalar_of(dest_ty).filter(|s| is_float_scalar(*s))?;
            let ConstVal::Float(f) = v else { return None };
            Some(ConstVal::Float(round_to_precision(f, dest_s)))
        }
        CastOp::FpExt => {
            // Widening a float's precision never changes its value.
            let ConstVal::Float(_) = v else { return None };
            Some(v)
        }
        CastOp::FpToSi => {
            let dest_w = scalar_int_width(dest_ty)?;
            let ConstVal::Float(f) = v else { return None };
            Some(ConstVal::Int(fp_to_int(f, dest_w, false)))
        }
        CastOp::FpToUi => {
            let dest_w = scalar_int_width(dest_ty)?;
            let ConstVal::Float(f) = v else { return None };
            Some(ConstVal::Int(fp_to_int(f, dest_w, true)))
        }
        CastOp::SiToFp => {
            let src_w = scalar_int_width(src_ty)?;
            let dest_s = scalar_of(dest_ty).filter(|s| is_float_scalar(*s))?;
            let ConstVal::Int(raw) = v else { return None };
            let sa = sign_extend((raw as u64) & mask_for(src_w), src_w);
            Some(ConstVal::Float(round_to_precision(sa as f64, dest_s)))
        }
        CastOp::UiToFp => {
            let src_w = scalar_int_width(src_ty)?;
            let dest_s = scalar_of(dest_ty).filter(|s| is_float_scalar(*s))?;
            let ConstVal::Int(raw) = v else { return None };
            let ua = (raw as u64) & mask_for(src_w);
            Some(ConstVal::Float(round_to_precision(ua as f64, dest_s)))
        }
        CastOp::Bitcast => fold_bitcast(src_ty, dest_ty, v),
    }
}

/// Truncating float -> integer conversion, saturating at the destination width's representable
/// range (NaN folds to zero) — the same total behavior Rust's own `as` float-to-int casts
/// define for `i8`/`i16`/`i32`/`i64`, generalized to an arbitrary bit width.
fn fp_to_int(f: f64, width: u32, unsigned: bool) -> i64 {
    if f.is_nan() {
        return 0;
    }
    let t = f.trunc();
    if unsigned {
        let max = mask_for(width);
        let clamped: u64 = if t <= 0.0 {
            0
        } else if t >= max as f64 {
            max
        } else {
            t as u64
        };
        sign_extend(clamped, width)
    } else {
        let min: i64 = if width >= 64 {
            i64::MIN
        } else {
            -(1i64 << (width - 1))
        };
        let max: i64 = if width >= 64 {
            i64::MAX
        } else {
            (1i64 << (width - 1)) - 1
        };
        let clamped: i64 = if t <= min as f64 {
            min
        } else if t >= max as f64 {
            max
        } else {
            t as i64
        };
        sign_extend(clamped as u64, width)
    }
}

fn round_to_precision(f: f64, dest: Scalar) -> f64 {
    match dest {
        Scalar::F64 => f,
        Scalar::F32 => (f as f32) as f64,
        Scalar::F16 => f16_to_f64(f64_to_f16(f)),
        Scalar::I1 | Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64 => {
            unreachable!("round_to_precision is only ever called with a float destination")
        }
    }
}

fn fold_bitcast(src_ty: Ty, dest_ty: Ty, v: ConstVal) -> Option<ConstVal> {
    let src_s = scalar_of(src_ty)?;
    let dest_s = scalar_of(dest_ty)?;
    let src_bits = bit_width_any(src_s);
    let dest_bits = bit_width_any(dest_s);
    if src_bits != dest_bits {
        // Never valid BIR (a real bitcast never changes bit width) — leave it unfolded.
        return None;
    }

    let raw: u64 = match (src_s, v) {
        (s, ConstVal::Int(raw)) if !is_float_scalar(s) => (raw as u64) & mask_for(src_bits),
        (Scalar::F16, ConstVal::Float(f)) => f64_to_f16(f) as u64,
        (Scalar::F32, ConstVal::Float(f)) => (f as f32).to_bits() as u64,
        (Scalar::F64, ConstVal::Float(f)) => f.to_bits(),
        _ => return None,
    };

    if is_float_scalar(dest_s) {
        let out = match dest_s {
            Scalar::F16 => f16_to_f64(raw as u16),
            Scalar::F32 => f32::from_bits(raw as u32) as f64,
            Scalar::F64 => f64::from_bits(raw),
            _ => return None,
        };
        Some(ConstVal::Float(out))
    } else {
        Some(ConstVal::Int(sign_extend(raw, dest_bits)))
    }
}

// --- hand-rolled IEEE 754 binary16 <-> binary64 conversion ---------------------------------
//
// Nothing else in the dependency graph provides `f16`; `basalt-passes` depends on nothing but
// `basalt-bir`, so this is a direct bit-level port of the standard round-to-nearest-even
// binary64 <-> binary16 conversion, operating on the binary64 bits directly rather than by
// double-rounding through an `f32` intermediate.

fn f64_to_f16(f: f64) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;

    if exp == 0x7ff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let m16 = (mantissa >> 42) as u16;
        return sign | 0x7c00 | if m16 == 0 { 0x0200 } else { m16 };
    }

    let half_exp = exp - 1023 + 15;

    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }

    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let m = mantissa | 0x0010_0000_0000_0000u64;
        let shift = (43 - half_exp) as u32;
        let mut half_mantissa = (m >> shift) as u32;
        let round_bit = 1u64 << (shift - 1);
        let sticky = m & (round_bit - 1);
        if (m & round_bit) != 0 && (sticky != 0 || (half_mantissa & 1) != 0) {
            half_mantissa += 1;
        }
        return sign | (half_mantissa as u16);
    }

    let mut half_mantissa = (mantissa >> 42) as u32;
    let round_bit = 1u64 << 41;
    let sticky = mantissa & (round_bit - 1);
    if (mantissa & round_bit) != 0 && (sticky != 0 || (half_mantissa & 1) != 0) {
        half_mantissa += 1;
    }
    let mut half_exp = half_exp as u32;
    if half_mantissa == 0x400 {
        half_mantissa = 0;
        half_exp += 1;
        if half_exp >= 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | ((half_exp as u16) << 10) | (half_mantissa as u16)
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = ((h & 0x8000) as u64) << 48;
    let exp = ((h >> 10) & 0x1f) as u64;
    let mantissa = (h & 0x3ff) as u64;

    let bits = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut e: i64 = -1;
            let mut m = mantissa;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let real_exp = (1023 - 15 + 1 + e) as u64;
            sign | (real_exp << 52) | (m << 42)
        }
    } else if exp == 0x1f {
        if mantissa == 0 {
            sign | 0x7ff0_0000_0000_0000
        } else {
            sign | 0x7ff0_0000_0000_0000 | (mantissa << 42)
        }
    } else {
        let real_exp = exp + (1023 - 15);
        sign | (real_exp << 52) | (mantissa << 42)
    };
    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{AddrSpace, Block, Function, ICmpPred as Pred, Inst, Term};

    fn scalar_fn(name: &str, params: Vec<Ty>, ret: Ty, insts: Vec<Inst>, term: Term) -> Function {
        Function {
            is_kernel: true,
            name: name.to_string(),
            params,
            ret,
            blocks: vec![Block {
                insts: (0..insts.len() as u32).map(InstId).collect(),
                term,
            }],
            insts,
        }
    }

    fn module_of(f: Function) -> Module {
        Module {
            funcs: vec![f],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    fn roundtrips(m: &Module) {
        let text = basalt_bir::print(m);
        let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
        assert_eq!(&reparsed, m, "parse(print(m)) != m\n{text}");
    }

    fn i(ty: Ty, op: Op) -> Inst {
        Inst { ty, op }
    }

    const I8: Ty = Ty::Scalar(Scalar::I8);
    const I32: Ty = Ty::Scalar(Scalar::I32);
    const I1: Ty = Ty::Scalar(Scalar::I1);
    const F32: Ty = Ty::Scalar(Scalar::F32);
    const F64: Ty = Ty::Scalar(Scalar::F64);

    fn val(idx: u32) -> ValRef {
        ValRef::Val(InstId(idx))
    }

    #[test]
    fn int_add_sub_mul_fold() {
        let f = scalar_fn(
            "k",
            vec![],
            I32,
            vec![
                i(I32, Op::ConstInt(7)),
                i(I32, Op::ConstInt(3)),
                i(I32, Op::Bin(BinOp::Add, val(0), val(1))),
                i(I32, Op::Bin(BinOp::Sub, val(0), val(1))),
                i(I32, Op::Bin(BinOp::Mul, val(0), val(1))),
            ],
            Term::Ret(Some(val(4))),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[2].op, Op::ConstInt(10));
        assert_eq!(insts[3].op, Op::ConstInt(4));
        assert_eq!(insts[4].op, Op::ConstInt(21));
        roundtrips(&out);
    }

    #[test]
    fn i8_add_wraps_at_eight_bits() {
        let f = scalar_fn(
            "k",
            vec![],
            I8,
            vec![
                i(I8, Op::ConstInt(120)),
                i(I8, Op::ConstInt(100)),
                i(I8, Op::Bin(BinOp::Add, val(0), val(1))),
            ],
            Term::Ret(Some(val(2))),
        );
        let out = constant_fold(&module_of(f));
        // 120 + 100 = 220, which as a signed 8-bit value wraps to 220 - 256 = -36.
        assert_eq!(out.funcs[0].insts[2].op, Op::ConstInt(-36));
        roundtrips(&out);
    }

    #[test]
    fn shift_and_bitwise_fold() {
        let f = scalar_fn(
            "k",
            vec![],
            I32,
            vec![
                i(I32, Op::ConstInt(0b1010)),
                i(I32, Op::ConstInt(2)),
                i(I32, Op::Bin(BinOp::Shl, val(0), val(1))),
                i(I32, Op::Bin(BinOp::And, val(0), val(1))),
                i(I32, Op::Bin(BinOp::Xor, val(0), val(1))),
            ],
            Term::Ret(Some(val(4))),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[2].op, Op::ConstInt(0b101000));
        assert_eq!(insts[3].op, Op::ConstInt(0b1010 & 2));
        assert_eq!(insts[4].op, Op::ConstInt(0b1010 ^ 2));
        roundtrips(&out);
    }

    #[test]
    fn float_arith_folds_ieee_correct() {
        let f = scalar_fn(
            "k",
            vec![],
            F64,
            vec![
                i(F64, Op::ConstFloat(1.5)),
                i(F64, Op::ConstFloat(0.25)),
                i(F64, Op::Bin(BinOp::FAdd, val(0), val(1))),
            ],
            Term::Ret(Some(val(2))),
        );
        let out = constant_fold(&module_of(f));
        assert_eq!(out.funcs[0].insts[2].op, Op::ConstFloat(1.75));
        roundtrips(&out);
    }

    #[test]
    fn icmp_and_fcmp_fold_to_zero_or_one() {
        let f = scalar_fn(
            "k",
            vec![],
            I1,
            vec![
                i(I32, Op::ConstInt(3)),
                i(I32, Op::ConstInt(5)),
                i(I1, Op::ICmp(Pred::Slt, I32, val(0), val(1))),
                i(F64, Op::ConstFloat(3.0)),
                i(F64, Op::ConstFloat(5.0)),
                i(
                    I1,
                    Op::FCmp(
                        FCmpPred::Ogt,
                        F64,
                        ValRef::Val(InstId(3)),
                        ValRef::Val(InstId(4)),
                    ),
                ),
            ],
            Term::Ret(Some(val(5))),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[2].op, Op::ConstInt(1));
        assert_eq!(insts[5].op, Op::ConstInt(0));
        roundtrips(&out);
    }

    #[test]
    fn every_cast_op_folds_one_representative_case() {
        let f = scalar_fn(
            "k",
            vec![],
            Ty::Void,
            vec![
                // 0: trunc i32 -1 (0xffffffff) -> i8 : low byte 0xff -> -1
                i(I32, Op::ConstInt(-1)),
                i(I8, Op::Cast(CastOp::Trunc, I32, val(0))),
                // 2: zext i8 -1 (0xff) -> i32 : 255
                i(I8, Op::ConstInt(-1)),
                i(I32, Op::Cast(CastOp::Zext, I8, val(2))),
                // 4: sext i8 -1 -> i32 : -1
                i(I8, Op::ConstInt(-1)),
                i(I32, Op::Cast(CastOp::Sext, I8, val(4))),
                // 6: fpext f32 1.5 -> f64 : 1.5
                i(F32, Op::ConstFloat(1.5)),
                i(F64, Op::Cast(CastOp::FpExt, F32, val(6))),
                // 8: fptrunc f64 1.5 -> f32 : 1.5 (exact)
                i(F64, Op::ConstFloat(1.5)),
                i(F32, Op::Cast(CastOp::FpTrunc, F64, val(8))),
                // 10: fptosi f64 3.75 -> i32 : 3
                i(F64, Op::ConstFloat(3.75)),
                i(I32, Op::Cast(CastOp::FpToSi, F64, val(10))),
                // 12: fptoui f64 3.75 -> i32 : 3
                i(F64, Op::ConstFloat(3.75)),
                i(I32, Op::Cast(CastOp::FpToUi, F64, val(12))),
                // 14: sitofp i32 -7 -> f64 : -7.0
                i(I32, Op::ConstInt(-7)),
                i(F64, Op::Cast(CastOp::SiToFp, I32, val(14))),
                // 16: uitofp i8 -1 (255 unsigned) -> f64 : 255.0
                i(I8, Op::ConstInt(-1)),
                i(F64, Op::Cast(CastOp::UiToFp, I8, val(16))),
                // 18: bitcast i32 bit pattern of 1.0f32 -> f32 : 1.0
                i(I32, Op::ConstInt(1.0f32.to_bits() as i32 as i64)),
                i(F32, Op::Cast(CastOp::Bitcast, I32, val(18))),
            ],
            Term::Ret(None),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[1].op, Op::ConstInt(-1), "trunc");
        assert_eq!(insts[3].op, Op::ConstInt(255), "zext");
        assert_eq!(insts[5].op, Op::ConstInt(-1), "sext");
        assert_eq!(insts[7].op, Op::ConstFloat(1.5), "fpext");
        assert_eq!(insts[9].op, Op::ConstFloat(1.5), "fptrunc");
        assert_eq!(insts[11].op, Op::ConstInt(3), "fptosi");
        assert_eq!(insts[13].op, Op::ConstInt(3), "fptoui");
        assert_eq!(insts[15].op, Op::ConstFloat(-7.0), "sitofp");
        assert_eq!(insts[17].op, Op::ConstFloat(255.0), "uitofp");
        assert_eq!(insts[19].op, Op::ConstFloat(1.0), "bitcast");
        roundtrips(&out);
    }

    #[test]
    fn fptrunc_to_f16_rounds_correctly() {
        let f = scalar_fn(
            "k",
            vec![],
            Ty::Scalar(Scalar::F16),
            vec![
                i(F64, Op::ConstFloat(100.1)),
                i(
                    Ty::Scalar(Scalar::F16),
                    Op::Cast(CastOp::FpTrunc, F64, val(0)),
                ),
            ],
            Term::Ret(Some(val(1))),
        );
        let out = constant_fold(&module_of(f));
        // f16 has 10 explicit mantissa bits; 100.1 rounds to 100.125 at that precision.
        assert_eq!(out.funcs[0].insts[1].op, Op::ConstFloat(100.125));
    }

    #[test]
    fn sign_extend_distinguishes_from_zero_extend() {
        let f = scalar_fn(
            "k",
            vec![],
            Ty::Void,
            vec![
                i(I8, Op::ConstInt(-2)), // 0xfe
                i(I32, Op::Cast(CastOp::Sext, I8, val(0))),
                i(I32, Op::Cast(CastOp::Zext, I8, val(0))),
            ],
            Term::Ret(None),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[1].op, Op::ConstInt(-2));
        assert_eq!(insts[2].op, Op::ConstInt(254));
    }

    #[test]
    fn chained_fold_within_single_pass() {
        // %2 = add(%0, %1); %3 = mul(%2, %2) — folding %2 must be visible when %3 is folded,
        // all within the one forward walk (no fixed-point loop).
        let f = scalar_fn(
            "k",
            vec![],
            I32,
            vec![
                i(I32, Op::ConstInt(2)),
                i(I32, Op::ConstInt(3)),
                i(I32, Op::Bin(BinOp::Add, val(0), val(1))),
                i(I32, Op::Bin(BinOp::Mul, val(2), val(2))),
            ],
            Term::Ret(Some(val(3))),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[2].op, Op::ConstInt(5));
        assert_eq!(insts[3].op, Op::ConstInt(25));
        roundtrips(&out);
    }

    #[test]
    fn div_and_rem_by_nonzero_constant_fold() {
        let f = scalar_fn(
            "k",
            vec![],
            I32,
            vec![
                i(I32, Op::ConstInt(17)),
                i(I32, Op::ConstInt(5)),
                i(I32, Op::Bin(BinOp::Div, val(0), val(1))),
                i(I32, Op::Bin(BinOp::Rem, val(0), val(1))),
            ],
            Term::Ret(Some(val(2))),
        );
        let out = constant_fold(&module_of(f));
        let insts = &out.funcs[0].insts;
        assert_eq!(insts[2].op, Op::ConstInt(3));
        assert_eq!(insts[3].op, Op::ConstInt(2));
        roundtrips(&out);
    }

    #[test]
    fn div_and_rem_by_constant_zero_left_unfolded() {
        let f = scalar_fn(
            "k",
            vec![],
            I32,
            vec![
                i(I32, Op::ConstInt(17)),
                i(I32, Op::ConstInt(0)),
                i(I32, Op::Bin(BinOp::Div, val(0), val(1))),
                i(I32, Op::Bin(BinOp::Rem, val(0), val(1))),
            ],
            Term::Ret(Some(val(2))),
        );
        let before = module_of(f);
        let out = constant_fold(&before);
        assert_eq!(out.funcs[0].insts[2].op, before.funcs[0].insts[2].op);
        assert_eq!(out.funcs[0].insts[3].op, before.funcs[0].insts[3].op);
        assert!(matches!(
            out.funcs[0].insts[2].op,
            Op::Bin(BinOp::Div, _, _)
        ));
        assert!(matches!(
            out.funcs[0].insts[3].op,
            Op::Bin(BinOp::Rem, _, _)
        ));
    }

    #[test]
    fn non_constant_operand_left_untouched() {
        // A single i32 parameter, added to a constant: the result must not fold.
        let f = scalar_fn(
            "k",
            vec![I32],
            I32,
            vec![
                i(I32, Op::ConstInt(1)),
                i(I32, Op::Bin(BinOp::Add, ValRef::Param(0), val(0))),
            ],
            Term::Ret(Some(val(1))),
        );
        let before = module_of(f);
        let out = constant_fold(&before);
        assert_eq!(out, before);
        roundtrips(&out);
    }

    #[test]
    fn ptr_const_ignored_by_load_not_folded() {
        // Sanity: a `load` is never touched by this pass even when its `ptr` operand is a
        // constant address (out of this pass's declared scope).
        let f = scalar_fn(
            "k",
            vec![],
            I32,
            vec![
                i(Ty::Ptr(AddrSpace::Global), Op::ConstInt(0)),
                i(
                    I32,
                    Op::Load {
                        ptr: val(0),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                ),
            ],
            Term::Ret(Some(val(1))),
        );
        let before = module_of(f);
        let out = constant_fold(&before);
        assert_eq!(out, before);
    }
}
