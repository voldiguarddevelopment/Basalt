// Scalar, pointer, and vector types for BIR values (ARCHITECTURE §3: "vector types
// (float2..4, int2..4) as first-class, not lowered early"). Pointers are opaque — they
// carry only an address space, never a pointee type — so a `load`/`store`'s own type
// field is the single source of truth for what is being read or written.

use std::fmt;

/// A scalar element type: a fixed-width integer or IEEE float.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

impl Scalar {
    /// Every scalar type, for tests and `--ir` dtype-set validation.
    pub const ALL: &'static [Scalar] = &[
        Scalar::I1,
        Scalar::I8,
        Scalar::I16,
        Scalar::I32,
        Scalar::I64,
        Scalar::F16,
        Scalar::F32,
        Scalar::F64,
    ];

    pub fn text(self) -> &'static str {
        match self {
            Scalar::I1 => "i1",
            Scalar::I8 => "i8",
            Scalar::I16 => "i16",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::F16 => "f16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
        }
    }

    pub fn parse(s: &str) -> Option<Scalar> {
        Scalar::ALL.iter().copied().find(|sc| sc.text() == s)
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}

/// The address space a pointer value refers into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrSpace {
    Global,
    Shared,
    Constant,
    Local,
    Param,
}

impl AddrSpace {
    pub const ALL: &'static [AddrSpace] = &[
        AddrSpace::Global,
        AddrSpace::Shared,
        AddrSpace::Constant,
        AddrSpace::Local,
        AddrSpace::Param,
    ];

    pub fn text(self) -> &'static str {
        match self {
            AddrSpace::Global => "global",
            AddrSpace::Shared => "shared",
            AddrSpace::Constant => "constant",
            AddrSpace::Local => "local",
            AddrSpace::Param => "param",
        }
    }

    pub fn parse(s: &str) -> Option<AddrSpace> {
        AddrSpace::ALL.iter().copied().find(|a| a.text() == s)
    }
}

impl fmt::Display for AddrSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}

/// A BIR value type. `Void` is only ever used as an instruction's result type, marking it
/// as producing no SSA value (e.g. `store`, `barrier`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Scalar(Scalar),
    Ptr(AddrSpace),
    Vec(Scalar, u8),
    Void,
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Ty::Scalar(s) => write!(f, "{s}"),
            Ty::Ptr(a) => write!(f, "ptr.{a}"),
            Ty::Vec(s, n) => write!(f, "v{n}{s}"),
            Ty::Void => write!(f, "void"),
        }
    }
}

impl Ty {
    /// Parses a type from its printed word form (`i32`, `ptr.global`, `v4f32`, `void`).
    /// Lives here (rather than only in the parser) since it is pure text <-> value
    /// conversion with no lexer/cursor state involved.
    pub fn parse(w: &str) -> Option<Ty> {
        if w == "void" {
            return Some(Ty::Void);
        }
        if let Some(rest) = w.strip_prefix("ptr.") {
            return AddrSpace::parse(rest).map(Ty::Ptr);
        }
        if let Some(s) = Scalar::parse(w) {
            return Some(Ty::Scalar(s));
        }
        if let Some(rest) = w.strip_prefix('v') {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                let lanes: u8 = digits.parse().ok()?;
                let scalar = Scalar::parse(&rest[digits.len()..])?;
                return Some(Ty::Vec(scalar, lanes));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_text_round_trips() {
        for &s in Scalar::ALL {
            assert_eq!(Scalar::parse(s.text()), Some(s));
        }
    }

    #[test]
    fn addr_space_text_round_trips() {
        for &a in AddrSpace::ALL {
            assert_eq!(AddrSpace::parse(a.text()), Some(a));
        }
    }

    #[test]
    fn ty_text_round_trips() {
        let tys = [
            Ty::Void,
            Ty::Scalar(Scalar::I32),
            Ty::Ptr(AddrSpace::Shared),
            Ty::Vec(Scalar::F32, 4),
            Ty::Vec(Scalar::I32, 2),
        ];
        for ty in tys {
            assert_eq!(Ty::parse(&ty.to_string()), Some(ty));
        }
    }

    #[test]
    fn unknown_type_word_rejected() {
        assert_eq!(Ty::parse("not_a_type"), None);
    }
}
