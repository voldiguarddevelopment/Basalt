// Shared SysV-ELF object writer. Every hand-rolled backend that emits a
// relocatable object — `basalt-x86` (oracle + regalloc), `basalt-rv`, `basalt-amdgpu`
// (HSACO is itself an ELF container) — builds its machine code and constant data, then
// hands them to `write_elf_object` instead of poking at ELF headers itself. This keeps the
// only ELF-layout knowledge in one place (backend isolation — a
// target crate owns its encoder, not the container format).
//
// Built on the `object` crate's write side, which already omits anything host- or
// time-dependent from the ELF output (unlike its COFF/PE writer, which stamps a build
// time). Nothing in `ElfObjectSpec` may vary run-to-run for a fixed input, so the result
// is deterministic: same spec in, byte-identical object out.

use object::write::{
    Object, StandardSection, Symbol, SymbolFlags, SymbolKind, SymbolScope, SymbolSection,
};
use object::BinaryFormat;
pub use object::{Architecture, Endianness};

use basalt_diag::{Diag, ECode};

/// A single global symbol exported at the start of `.text`, sized to cover the whole
/// section. Sufficient for one kernel/function per object, which is all the hand-rolled
/// backends need for now; multi-symbol objects can grow this later without
/// changing the shape of this module's contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfObjectSpec {
    /// ISA the object targets. Determines how the ELF header's `e_machine` field reads.
    pub architecture: Architecture,
    /// Byte order of the target ISA.
    pub endian: Endianness,
    /// Name bound to offset 0 of `.text`, `size = text.len()`.
    pub symbol: String,
    /// Machine code for `.text`. May be empty (an object with only data is legal ELF).
    pub text: Vec<u8>,
    /// Required alignment of `.text` within the object, in bytes. Must be a power of two.
    pub text_align: u64,
    /// Initialized read-only data for `.rodata`, if the kernel has any constants.
    pub rodata: Option<Vec<u8>>,
    /// Initialized read-write data for `.data`, if the kernel has any.
    pub data: Option<Vec<u8>>,
}

impl ElfObjectSpec {
    /// A spec with just a symbol and its `.text` bytes, 16-byte aligned (the common case:
    /// no globals, no read-only constants).
    pub fn new(
        architecture: Architecture,
        endian: Endianness,
        symbol: impl Into<String>,
        text: Vec<u8>,
    ) -> ElfObjectSpec {
        ElfObjectSpec {
            architecture,
            endian,
            symbol: symbol.into(),
            text,
            text_align: 16,
            rodata: None,
            data: None,
        }
    }

    #[must_use]
    pub fn with_rodata(mut self, rodata: Vec<u8>) -> ElfObjectSpec {
        self.rodata = Some(rodata);
        self
    }

    #[must_use]
    pub fn with_data(mut self, data: Vec<u8>) -> ElfObjectSpec {
        self.data = Some(data);
        self
    }
}

/// Builds a relocatable SysV ELF object from `spec`: a `.text` section holding
/// `spec.text`, one global function symbol naming its start, and optional `.rodata`/
/// `.data` sections. Returns the object's bytes, ready to write straight to a `.o` file.
///
/// Deterministic: the same `ElfObjectSpec` always serializes to the same bytes, on any
/// host, any number of times.
pub fn write_elf_object(spec: &ElfObjectSpec) -> Result<Vec<u8>, Diag> {
    let mut obj = Object::new(BinaryFormat::Elf, spec.architecture, spec.endian);

    let text_id = obj.section_id(StandardSection::Text);
    let symbol_id = obj.add_symbol(Symbol {
        name: spec.symbol.clone().into_bytes(),
        value: 0,
        size: spec.text.len() as u64,
        kind: SymbolKind::Text,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    obj.add_symbol_data(symbol_id, text_id, &spec.text, spec.text_align);

    if let Some(rodata) = &spec.rodata {
        let id = obj.section_id(StandardSection::ReadOnlyData);
        obj.append_section_data(id, rodata, 1);
    }
    if let Some(data) = &spec.data {
        let id = obj.section_id(StandardSection::Data);
        obj.append_section_data(id, data, 1);
    }

    obj.write()
        .map_err(|e| Diag::new(ECode::IoError).with_arg(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use object::read::{Object as ReadObject, ObjectSection, ObjectSymbol};

    fn sample_spec() -> ElfObjectSpec {
        ElfObjectSpec::new(
            Architecture::X86_64,
            Endianness::Little,
            "kernel_entry",
            vec![0x90, 0x90, 0xc3], // nop; nop; ret
        )
    }

    #[test]
    fn builds_a_valid_elf_object() {
        let bytes = write_elf_object(&sample_spec()).expect("write succeeds");

        let file = object::read::File::parse(&*bytes).expect("parses as an object file");
        assert_eq!(file.format(), object::BinaryFormat::Elf);
        assert_eq!(file.architecture(), object::Architecture::X86_64);

        let text = file
            .section_by_name(".text")
            .expect(".text section present");
        assert_eq!(text.data().unwrap(), &[0x90, 0x90, 0xc3]);

        let sym = file
            .symbols()
            .find(|s| s.name() == Ok("kernel_entry"))
            .expect("symbol present");
        assert_eq!(sym.size(), 3);
    }

    #[test]
    fn carries_optional_rodata_and_data_sections() {
        let spec = sample_spec()
            .with_rodata(vec![0x01, 0x02, 0x03, 0x04])
            .with_data(vec![0xaa, 0xbb]);
        let bytes = write_elf_object(&spec).expect("write succeeds");

        let file = object::read::File::parse(&*bytes).expect("parses as an object file");
        let rodata = file.section_by_name(".rodata").expect(".rodata present");
        assert_eq!(rodata.data().unwrap(), &[0x01, 0x02, 0x03, 0x04]);
        let data = file.section_by_name(".data").expect(".data present");
        assert_eq!(data.data().unwrap(), &[0xaa, 0xbb]);
    }

    #[test]
    fn same_spec_produces_byte_identical_output() {
        let spec = sample_spec();
        let a = write_elf_object(&spec).unwrap();
        let b = write_elf_object(&spec).unwrap();
        assert_eq!(
            a, b,
            "determinism: same spec must yield byte-identical output"
        );
    }
}
