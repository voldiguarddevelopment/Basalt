// The checker's own type representation, distinct from `ast::Type` (the syntax as written).
// Two spellings of the same type must compare equal here: a typedef and the type it aliases,
// or `struct Foo` written with and without the tag keyword, both resolve to the same `Ty`.
//
// `Ty::Unknown` stands in for anything this pass cannot pin down without real template
// instantiation, plus recovery after an earlier error. It is compatible with everything, so
// it never itself produces a diagnostic and never suppresses one already reported elsewhere.

use basalt_frontend_c::ast::ScalarKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Ty {
    Scalar(ScalarKind),
    Pointer(Box<Ty>),
    Array(Box<Ty>),
    Struct(String),
    Union(String),
    Enum(String),
    Function {
        ret: Box<Ty>,
        params: Vec<Ty>,
        variadic: bool,
    },
    /// Template-instantiated types and error-recovery placeholders. Always compatible.
    Unknown,
}

fn is_integer_kind(k: ScalarKind) -> bool {
    !matches!(
        k,
        ScalarKind::Void | ScalarKind::Float | ScalarKind::Double | ScalarKind::LongDouble
    )
}

fn is_float_kind(k: ScalarKind) -> bool {
    matches!(
        k,
        ScalarKind::Float | ScalarKind::Double | ScalarKind::LongDouble
    )
}

/// Whether an integer `ScalarKind` is signed. Used by BIR lowering (`lower.rs`) to pick
/// `icmp`'s signed vs. unsigned predicates and `sext` vs. `zext` for a widening integer cast;
/// meaningless for a float kind (never called with one).
pub(crate) fn is_signed_kind(k: ScalarKind) -> bool {
    !matches!(
        k,
        ScalarKind::Bool
            | ScalarKind::UChar
            | ScalarKind::UShort
            | ScalarKind::UInt
            | ScalarKind::ULong
            | ScalarKind::ULongLong
            | ScalarKind::WcharT
    )
}

impl Ty {
    pub(crate) fn is_unknown(&self) -> bool {
        matches!(self, Ty::Unknown)
    }

    pub(crate) fn is_pointer(&self) -> bool {
        matches!(self, Ty::Pointer(_))
    }

    pub(crate) fn is_pointer_like(&self) -> bool {
        matches!(self, Ty::Pointer(_) | Ty::Array(_))
    }

    pub(crate) fn is_integer(&self) -> bool {
        match self {
            Ty::Scalar(k) => is_integer_kind(*k),
            Ty::Enum(_) => true,
            _ => false,
        }
    }

    pub(crate) fn is_float(&self) -> bool {
        matches!(self, Ty::Scalar(k) if is_float_kind(*k))
    }

    pub(crate) fn is_arithmetic(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// A condition/operand position that C accepts as "truthy": arithmetic or pointer.
    pub(crate) fn is_scalar_condition(&self) -> bool {
        self.is_arithmetic() || self.is_pointer_like()
    }

    /// The pointee (pointer) or element (array) type, for `*`/`[]`/`->`.
    pub(crate) fn deref_target(&self) -> Option<Ty> {
        match self {
            Ty::Pointer(p) => Some((**p).clone()),
            Ty::Array(e) => Some((**e).clone()),
            _ => None,
        }
    }

    fn is_void_pointee(&self) -> bool {
        matches!(self, Ty::Scalar(ScalarKind::Void))
    }
}

/// Is `value` assignable to a variable/parameter/return slot of type `target`? Deliberately
/// permissive relative to strict C (no distinct integer-conversion-rank/promotion ladder, and
/// pointer-from-integer is allowed outright rather than only for a null-constant `0`) — the
/// point of this pass is to catch outright category errors (assigning a struct to an int, a
/// pointer to an incompatible pointer, ...), not to fully police implicit-conversion warnings.
pub(crate) fn assignable(target: &Ty, value: &Ty) -> bool {
    if target.is_unknown() || value.is_unknown() {
        return true;
    }
    if target == value {
        return true;
    }
    match (target, value) {
        (Ty::Scalar(a), Ty::Scalar(b)) => *a != ScalarKind::Void && *b != ScalarKind::Void,
        (Ty::Scalar(a), Ty::Enum(_)) => is_integer_kind(*a),
        (Ty::Enum(_), Ty::Scalar(b)) => is_integer_kind(*b),
        (Ty::Enum(_), Ty::Enum(_)) => true,
        (Ty::Pointer(pa), Ty::Pointer(pb)) => {
            pa.is_void_pointee() || pb.is_void_pointee() || pa == pb
        }
        (Ty::Pointer(pa), Ty::Array(eb)) => pa.is_void_pointee() || **pa == **eb,
        (Ty::Pointer(_), Ty::Scalar(b)) => is_integer_kind(*b),
        (Ty::Struct(a), Ty::Struct(b)) => a == b,
        (Ty::Union(a), Ty::Union(b)) => a == b,
        _ => false,
    }
}

/// Rough arithmetic promotion for a binary operator's result: not the real C ladder (no
/// signedness-driven "usual arithmetic conversions"), just enough to give a sane result type
/// when both operands already passed the arithmetic check.
pub(crate) fn promote(a: &Ty, b: &Ty) -> Ty {
    use ScalarKind::*;
    let rank = |k: ScalarKind| -> i32 {
        match k {
            Void => -1,
            Bool => 0,
            Char | SChar | UChar => 1,
            Short | UShort => 2,
            Int | UInt | WcharT => 3,
            Long | ULong => 4,
            LongLong | ULongLong => 5,
            Float => 6,
            Double => 7,
            LongDouble => 8,
        }
    };
    let scalar_of = |t: &Ty| -> ScalarKind {
        match t {
            Ty::Scalar(k) => *k,
            Ty::Enum(_) => Int,
            _ => Int,
        }
    };
    let (ka, kb) = (scalar_of(a), scalar_of(b));
    if rank(ka) >= rank(kb) {
        Ty::Scalar(if rank(ka) < rank(Int) { Int } else { ka })
    } else {
        Ty::Scalar(if rank(kb) < rank(Int) { Int } else { kb })
    }
}
