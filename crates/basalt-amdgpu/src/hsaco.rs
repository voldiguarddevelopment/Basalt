// HSACO (HSA Code Object) container writer: the ELF wrapper a real AMDGPU loader expects
// around a kernel's machine code, wave-registers/-segment accounting, and metadata. This is
// the AMDGPU equivalent of `basalt-backend/src/elf.rs`'s `write_elf_object`, but hand-rolled
// separately rather than routed through it (backend isolation — `object`'s write side has no
// `Architecture::AmdGpu`, EM_AMDGPU=224/ELFOSABI_AMDGPU_HSA=64/AMDGPU `e_flags` are entirely
// this target's own concern, and the container has structure no other backend needs: a kernel
// descriptor, a self-relocating entry-point field, NT_AMDGPU_METADATA).
//
// Every field/offset/byte pattern below was pinned down empirically against a real HSACO
// produced by LLVM's own AMDGPU `TargetMachine` object emission (`llvm-mc`'s sibling, the
// direct-to-object path) for a real kernel, cross-checked three ways: `readelf -h/-S/-x`,
// `llvm-objdump -d/-s`, and tinygrad's own `elf_loader`/`AMDProgram` (a real, compiler-agnostic
// ET_REL AMDGPU object loader — it re-derives kernel dispatch parameters straight from the
// kernel descriptor bytes and applies `R_AMDGPU_REL64` relocations itself, so it is exactly as
// strict a reader as anything this project could hand-write). The reference object is `REL`
// (relocatable), not `DYN`: tinygrad's mock RDNA3 backend loads that raw, unlinked object
// directly, and so do we — there is no `ld.lld` step anywhere in this pipeline.
//
// # Kernel descriptor (`amd_kernel_code_t`, code object v5 form)
//
// A fixed 64-byte struct, placed at `.rodata` offset 0 by this writer (one kernel per
// object, same simplification `ElfObjectSpec` makes for `.text`). Field layout (offset,
// size): `group_segment_fixed_size` (0, 4), `private_segment_fixed_size` (4, 4),
// `kernarg_size` (8, 4), reserved (12, 4), `kernel_code_entry_byte_offset` (16, 8), reserved
// (24, 20), `compute_pgm_rsrc3` (44, 4), `compute_pgm_rsrc1` (48, 4), `compute_pgm_rsrc2` (52,
// 4), `kernel_code_properties` (56, 2), `kernarg_preload` (58, 2), reserved (60, 4).
//
// `kernel_code_entry_byte_offset` is never filled in directly: the real object leaves it zero
// and instead carries one `R_AMDGPU_REL64` relocation (type 5) against the kernel's own `.text`
// symbol, `r_offset` = 0x10 (the field's own offset within `.rodata`), addend = 0x10. Working
// through the relocation arithmetic a real loader applies (`value = S + A - P`, then the
// consumer computes `entry = rodata_addr + value`) shows the addend exactly cancels the field's
// own offset, so `entry` resolves to the kernel symbol's address with no further adjustment —
// this only works out because the field sits at `.rodata + 0x10` and the kernel body starts at
// the referenced symbol's value with no header bytes in front of it, both true here by
// construction. This is why the relocation is emitted rather than a precomputed constant: a
// real linker relaying this object elsewhere would place `.text`/`.rodata` at addresses this
// writer cannot know in advance, and the relocation is what stays correct regardless.
//
// `compute_pgm_rsrc1`'s VGPR-count field is granulated in units of 8 registers on RDNA
// (`gfx1100`, wave32) — confirmed against the reference (`vgpr_count = 32` in its metadata
// produced a granulated field of exactly 3, i.e. `32/8 - 1`) — while its SGPR-count field reads
// zero in the same reference regardless of the metadata's own (non-zero) SGPR count, matching
// every other GFX10+ note that SGPR allocation is no longer tracked per-wave in this field.
// Every other `rsrc1` mode bit (float round/denorm mode, IEEE mode, DX10 clamp, WGP mode, ...)
// defaults to 0 here (hardware's own default, "nothing special requested") since nothing this
// writer emits yet depends on them; a real lowering pass gets to set them once it exists.
//
// # `.note` metadata
//
// One `NT_AMDGPU_METADATA` (type 32) note, name `"AMDGPU\0"` (namesz 7, padded to 8), whose
// descriptor is a MessagePack-encoded map decoded by hand from the reference (`readelf -x
// .note`, then walked byte-by-byte against MessagePack's format spec — fixmap/fixarray/fixstr
// and the `0xcc`/`0xcd` uint prefixes are all this schema needs): a top-level 3-entry map
// (`amdhsa.kernels`, `amdhsa.target`, `amdhsa.version`, in that — alphabetical — key order,
// matching the reference's own), one 15-key map per kernel inside `amdhsa.kernels` (also
// alphabetical), and `amdhsa.version = [1, 2]` (code object v5's own version pair; the v5
// marker is actually `e_ident[EI_ABIVERSION] == 3`, not the `.note` version array, which is
// why the ELF header comment above calls out `ELFABIVERSION_AMDGPU_HSA_V5` specifically). This
// writer's own kernels always report zero arguments (`.args = []`): populating real per-
// argument entries (`value_kind`/`offset`/`size`, including the `hidden_*` implicit ones a
// real kernarg segment needs) is a lowering-pass concern once one exists to describe.

use basalt_diag::Diag;

const ELFOSABI_AMDGPU_HSA: u8 = 64;
const ELFABIVERSION_AMDGPU_HSA_V5: u8 = 3;
const EM_AMDGPU: u16 = 224;
const ET_REL: u16 = 1;

const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOTE: u32 = 7;

const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;

const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_OBJECT: u8 = 1;
const STV_DEFAULT: u8 = 0;
const STV_PROTECTED: u8 = 3;

const R_AMDGPU_REL64: u32 = 5;
const NT_AMDGPU_METADATA: u32 = 32;

/// The GFX architecture a HSACO targets: the six RDNA3/GFX11 variants confirmed, empirically,
/// to assemble this backend's own instruction encoder (`enc.rs`) identically to `gfx1100` (see
/// `GfxArch::parse`'s own doc comment for the verification method). GFX10, GFX12, and CDNA are
/// deliberately not variants here — each is a real, confirmed ISA-level break with this
/// encoder (different memory-op mnemonics on GFX10, a different atomic/SMEM cache-policy-bit
/// layout on GFX12, near-total incompatibility on CDNA; see `lower.rs`'s own module header) —
/// so a caller asking for one of those gets a clean refusal (`GfxArch::parse` returns `None`)
/// rather than a `GfxArch` that would misrepresent it as "just another GFX11 flavor."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxArch {
    Gfx1100,
    Gfx1101,
    Gfx1102,
    Gfx1103,
    Gfx1150,
    Gfx1151,
}

impl GfxArch {
    /// Parses a `--target-variant`-style GFX name (e.g. `"gfx1101"`) into a `GfxArch`.
    /// Recognizes exactly the six names below — no fuzzy matching, no case-insensitivity —
    /// each confirmed real and byte-compatible with every one of this backend's ~250
    /// instruction forms via `llvm-mc-18 -triple=amdgcn-amd-amdhsa -mcpu=<name>
    /// -show-encoding` (every distinct mnemonic `enc.rs` emits, checked against every
    /// variant) plus `llvm-mc-18 ... -filetype=obj` piped through `readelf -h` (the resulting
    /// `e_flags`, cross-checked against `mach_code()` below). Anything else — including a real
    /// GFX10/GFX12/CDNA name confirmed *incompatible* by that same method (see `lower.rs`'s
    /// own module header) and plain nonsense — returns `None`; callers turn that into a clean
    /// `E093` refusal, never a guess. `gfx1152` is real silicon but is deliberately not a
    /// variant here: LLVM 18 (the only toolchain available anywhere in this project) does not
    /// recognize it as a processor at all ("not a recognized processor for this target"), so it
    /// cannot be verified, and this project's own doctrine is to never claim an unverifiable
    /// target.
    pub fn parse(name: &str) -> Option<GfxArch> {
        match name {
            "gfx1100" => Some(GfxArch::Gfx1100),
            "gfx1101" => Some(GfxArch::Gfx1101),
            "gfx1102" => Some(GfxArch::Gfx1102),
            "gfx1103" => Some(GfxArch::Gfx1103),
            "gfx1150" => Some(GfxArch::Gfx1150),
            "gfx1151" => Some(GfxArch::Gfx1151),
            _ => None,
        }
    }

    /// `EF_AMDGPU_MACH_AMDGCN_GFX*`. The whole `e_flags` value for this target: no
    /// XNACK/SRAMECC feature bits are set for any of the six (RDNA3 supports neither),
    /// matching the reference exactly for `gfx1100` (`e_flags == 0x41`, no bits outside the
    /// 8-bit mach-code field) and confirmed the same way for the other five (`llvm-mc-18
    /// -mcpu=<name> -filetype=obj` + `readelf -h`): `gfx1101` = `0x46`, `gfx1102` = `0x47`,
    /// `gfx1103` = `0x44`, `gfx1150` = `0x43`, `gfx1151` = `0x4a`.
    fn mach_code(self) -> u32 {
        match self {
            GfxArch::Gfx1100 => 0x41,
            GfxArch::Gfx1101 => 0x46,
            GfxArch::Gfx1102 => 0x47,
            GfxArch::Gfx1103 => 0x44,
            GfxArch::Gfx1150 => 0x43,
            GfxArch::Gfx1151 => 0x4a,
        }
    }

    fn target_triple(self) -> &'static str {
        match self {
            GfxArch::Gfx1100 => "amdgcn-amd-amdhsa--gfx1100",
            GfxArch::Gfx1101 => "amdgcn-amd-amdhsa--gfx1101",
            GfxArch::Gfx1102 => "amdgcn-amd-amdhsa--gfx1102",
            GfxArch::Gfx1103 => "amdgcn-amd-amdhsa--gfx1103",
            GfxArch::Gfx1150 => "amdgcn-amd-amdhsa--gfx1150",
            GfxArch::Gfx1151 => "amdgcn-amd-amdhsa--gfx1151",
        }
    }
}

/// Everything `write_hsaco` needs for a single-kernel HSACO object. One kernel per object is
/// this writer's whole scope for now, the same simplification `ElfObjectSpec` makes for a
/// plain ELF object's `.text` symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HsacoSpec {
    pub gfx_arch: GfxArch,
    pub kernel_name: String,
    /// Raw machine code for `.text` — every byte of the kernel body, entry point first.
    pub code: Vec<u8>,
    /// Total kernarg segment size in bytes (0 if the kernel takes no arguments).
    pub kernarg_segment_size: u32,
    /// Required alignment of the kernarg segment, in bytes.
    pub kernarg_segment_align: u32,
    pub group_segment_fixed_size: u32,
    pub private_segment_fixed_size: u32,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub vgpr_spill_count: u32,
    pub sgpr_spill_count: u32,
    /// Whether the kernel body expects the kernarg segment's base pointer in a user SGPR pair
    /// (`s[0:1]`, by this project's own convention). Setting the kernarg segment (via
    /// `with_kernarg_segment`) turns this on automatically; call sites with a nonstandard
    /// calling convention can still override it directly.
    pub enable_sgpr_kernarg_segment_ptr: bool,
    /// Whether the kernel body expects its X/Y/Z workgroup (block) index preloaded into the
    /// next user SGPR(s) after the kernarg pointer (`s2`, `s3`, `s4` in order, skipping any
    /// axis left disabled — real hardware only reserves an SGPR for an axis actually
    /// requested). Set via `with_workgroup_ids`.
    pub enable_sgpr_workgroup_id_x: bool,
    pub enable_sgpr_workgroup_id_y: bool,
    pub enable_sgpr_workgroup_id_z: bool,
}

impl HsacoSpec {
    /// A spec for a kernel with no arguments and no register/segment usage worth reporting —
    /// exactly what a bare `s_endpgm` body needs. `kernarg_segment_align` defaults to 8 (the
    /// standard kernarg segment alignment) even though an empty segment has nothing to align.
    pub fn new(gfx_arch: GfxArch, kernel_name: impl Into<String>, code: Vec<u8>) -> HsacoSpec {
        HsacoSpec {
            gfx_arch,
            kernel_name: kernel_name.into(),
            code,
            kernarg_segment_size: 0,
            kernarg_segment_align: 8,
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            vgpr_count: 0,
            sgpr_count: 0,
            vgpr_spill_count: 0,
            sgpr_spill_count: 0,
            enable_sgpr_kernarg_segment_ptr: false,
            enable_sgpr_workgroup_id_x: false,
            enable_sgpr_workgroup_id_y: false,
            enable_sgpr_workgroup_id_z: false,
        }
    }

    #[must_use]
    pub fn with_kernarg_segment(mut self, size: u32, align: u32) -> HsacoSpec {
        self.kernarg_segment_size = size;
        self.kernarg_segment_align = align;
        self.enable_sgpr_kernarg_segment_ptr = size > 0;
        self
    }

    #[must_use]
    pub fn with_segments(mut self, group_fixed_size: u32, private_fixed_size: u32) -> HsacoSpec {
        self.group_segment_fixed_size = group_fixed_size;
        self.private_segment_fixed_size = private_fixed_size;
        self
    }

    /// Requests the workgroup index (x, y, z) be preloaded into a user SGPR per enabled axis,
    /// in order, right after the kernarg-pointer pair — the real hardware/kernel-descriptor
    /// mechanism `Op::BidX`/`BidY`/`BidZ` lower against (see `lower.rs`'s own header).
    #[must_use]
    pub fn with_workgroup_ids(mut self, x: bool, y: bool, z: bool) -> HsacoSpec {
        self.enable_sgpr_workgroup_id_x = x;
        self.enable_sgpr_workgroup_id_y = y;
        self.enable_sgpr_workgroup_id_z = z;
        self
    }

    #[must_use]
    pub fn with_register_counts(
        mut self,
        vgpr_count: u32,
        sgpr_count: u32,
        vgpr_spill_count: u32,
        sgpr_spill_count: u32,
    ) -> HsacoSpec {
        self.vgpr_count = vgpr_count;
        self.sgpr_count = sgpr_count;
        self.vgpr_spill_count = vgpr_spill_count;
        self.sgpr_spill_count = sgpr_spill_count;
        self
    }
}

fn round_up(x: u64, align: u64) -> u64 {
    if align <= 1 {
        x
    } else {
        x.div_ceil(align) * align
    }
}

fn kd_bytes(spec: &HsacoSpec) -> [u8; 64] {
    let granulated_vgpr = if spec.vgpr_count == 0 {
        0
    } else {
        spec.vgpr_count.div_ceil(8) - 1
    };
    let rsrc1 = granulated_vgpr & 0x3f;

    let mut rsrc2 = 0u32;
    if spec.private_segment_fixed_size > 0 {
        rsrc2 |= 1 << 0;
    }
    let user_sgpr_count: u32 = if spec.enable_sgpr_kernarg_segment_ptr {
        2
    } else {
        0
    };
    rsrc2 |= (user_sgpr_count & 0x1f) << 1;
    if spec.enable_sgpr_workgroup_id_x {
        rsrc2 |= 1 << 7;
    }
    if spec.enable_sgpr_workgroup_id_y {
        rsrc2 |= 1 << 8;
    }
    if spec.enable_sgpr_workgroup_id_z {
        rsrc2 |= 1 << 9;
    }

    let mut code_properties: u16 = 1 << 10; // ENABLE_WAVEFRONT_SIZE32, always set: wave32-only.
    if spec.enable_sgpr_kernarg_segment_ptr {
        code_properties |= 1 << 3; // ENABLE_SGPR_KERNARG_SEGMENT_PTR
    }

    let mut kd = [0u8; 64];
    kd[0..4].copy_from_slice(&spec.group_segment_fixed_size.to_le_bytes());
    kd[4..8].copy_from_slice(&spec.private_segment_fixed_size.to_le_bytes());
    kd[8..12].copy_from_slice(&spec.kernarg_segment_size.to_le_bytes());
    // 12..16 reserved, 16..24 kernel_code_entry_byte_offset: left zero, resolved by the
    // R_AMDGPU_REL64 relocation the caller emits into .rela.rodata (see module header).
    // 24..44 reserved.
    kd[44..48].copy_from_slice(&0u32.to_le_bytes()); // compute_pgm_rsrc3
    kd[48..52].copy_from_slice(&rsrc1.to_le_bytes());
    kd[52..56].copy_from_slice(&rsrc2.to_le_bytes());
    kd[56..58].copy_from_slice(&code_properties.to_le_bytes());
    // 58..60 kernarg_preload, 60..64 reserved: left zero.
    kd
}

mod mp {
    //! A hand-rolled MessagePack encoder covering exactly the value shapes
    //! `NT_AMDGPU_METADATA` needs: fixmap/map16, fixarray/array16, fixstr/str8/str16, unsigned
    //! integers, and bool. Not a general MessagePack implementation.

    pub fn push_map_header(buf: &mut Vec<u8>, len: usize) {
        if len < 16 {
            buf.push(0x80 | len as u8);
        } else {
            buf.push(0xde);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        }
    }

    pub fn push_array_header(buf: &mut Vec<u8>, len: usize) {
        if len < 16 {
            buf.push(0x90 | len as u8);
        } else {
            buf.push(0xdc);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        }
    }

    pub fn push_str(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        if bytes.len() < 32 {
            buf.push(0xa0 | bytes.len() as u8);
        } else if bytes.len() < 256 {
            buf.push(0xd9);
            buf.push(bytes.len() as u8);
        } else {
            buf.push(0xda);
            buf.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        }
        buf.extend_from_slice(bytes);
    }

    pub fn push_uint(buf: &mut Vec<u8>, v: u64) {
        if v < 0x80 {
            buf.push(v as u8);
        } else if v <= 0xff {
            buf.push(0xcc);
            buf.push(v as u8);
        } else if v <= 0xffff {
            buf.push(0xcd);
            buf.extend_from_slice(&(v as u16).to_be_bytes());
        } else if v <= 0xffff_ffff {
            buf.push(0xce);
            buf.extend_from_slice(&(v as u32).to_be_bytes());
        } else {
            buf.push(0xcf);
            buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    pub fn push_bool(buf: &mut Vec<u8>, b: bool) {
        buf.push(if b { 0xc3 } else { 0xc2 });
    }
}

fn note_metadata(spec: &HsacoSpec) -> Vec<u8> {
    let mut m = Vec::new();

    // Per-kernel map, 15 keys, alphabetical (matches the reference exactly).
    let mut kernel = Vec::new();
    mp::push_map_header(&mut kernel, 15);
    mp::push_str(&mut kernel, ".args");
    mp::push_array_header(&mut kernel, 0); // no per-argument entries yet; see module header.
    mp::push_str(&mut kernel, ".group_segment_fixed_size");
    mp::push_uint(&mut kernel, spec.group_segment_fixed_size as u64);
    mp::push_str(&mut kernel, ".kernarg_segment_align");
    mp::push_uint(&mut kernel, spec.kernarg_segment_align as u64);
    mp::push_str(&mut kernel, ".kernarg_segment_size");
    mp::push_uint(&mut kernel, spec.kernarg_segment_size as u64);
    mp::push_str(&mut kernel, ".max_flat_workgroup_size");
    mp::push_uint(&mut kernel, 1024);
    mp::push_str(&mut kernel, ".name");
    mp::push_str(&mut kernel, &spec.kernel_name);
    mp::push_str(&mut kernel, ".private_segment_fixed_size");
    mp::push_uint(&mut kernel, spec.private_segment_fixed_size as u64);
    mp::push_str(&mut kernel, ".sgpr_count");
    mp::push_uint(&mut kernel, spec.sgpr_count as u64);
    mp::push_str(&mut kernel, ".sgpr_spill_count");
    mp::push_uint(&mut kernel, spec.sgpr_spill_count as u64);
    mp::push_str(&mut kernel, ".symbol");
    mp::push_str(&mut kernel, &format!("{}.kd", spec.kernel_name));
    mp::push_str(&mut kernel, ".uses_dynamic_stack");
    mp::push_bool(&mut kernel, false);
    mp::push_str(&mut kernel, ".vgpr_count");
    mp::push_uint(&mut kernel, spec.vgpr_count as u64);
    mp::push_str(&mut kernel, ".vgpr_spill_count");
    mp::push_uint(&mut kernel, spec.vgpr_spill_count as u64);
    mp::push_str(&mut kernel, ".wavefront_size");
    mp::push_uint(&mut kernel, 32);
    mp::push_str(&mut kernel, ".workgroup_processor_mode");
    mp::push_uint(&mut kernel, 1);

    mp::push_map_header(&mut m, 3);
    mp::push_str(&mut m, "amdhsa.kernels");
    mp::push_array_header(&mut m, 1);
    m.extend_from_slice(&kernel);
    mp::push_str(&mut m, "amdhsa.target");
    mp::push_str(&mut m, spec.gfx_arch.target_triple());
    mp::push_str(&mut m, "amdhsa.version");
    mp::push_array_header(&mut m, 2);
    mp::push_uint(&mut m, 1);
    mp::push_uint(&mut m, 2);
    m
}

fn note_section_bytes(spec: &HsacoSpec) -> Vec<u8> {
    let desc = note_metadata(spec);
    let name = b"AMDGPU\0";
    let mut note = Vec::new();
    note.extend_from_slice(&(name.len() as u32).to_le_bytes());
    note.extend_from_slice(&(desc.len() as u32).to_le_bytes());
    note.extend_from_slice(&NT_AMDGPU_METADATA.to_le_bytes());
    note.extend_from_slice(name);
    while note.len() % 4 != 0 {
        note.push(0);
    }
    note.extend_from_slice(&desc);
    while note.len() % 4 != 0 {
        note.push(0);
    }
    note
}

struct StrTab {
    bytes: Vec<u8>,
}

impl StrTab {
    fn new() -> StrTab {
        StrTab { bytes: vec![0] }
    }

    fn push(&mut self, s: &str) -> u32 {
        let off = self.bytes.len() as u32;
        self.bytes.extend_from_slice(s.as_bytes());
        self.bytes.push(0);
        off
    }
}

fn sym_entry(name: u32, info: u8, other: u8, shndx: u16, value: u64, size: u64) -> [u8; 24] {
    let mut b = [0u8; 24];
    b[0..4].copy_from_slice(&name.to_le_bytes());
    b[4] = info;
    b[5] = other;
    b[6..8].copy_from_slice(&shndx.to_le_bytes());
    b[8..16].copy_from_slice(&value.to_le_bytes());
    b[16..24].copy_from_slice(&size.to_le_bytes());
    b
}

#[allow(clippy::too_many_arguments)]
fn shdr(
    name: u32,
    typ: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
) -> [u8; 64] {
    let mut b = [0u8; 64];
    b[0..4].copy_from_slice(&name.to_le_bytes());
    b[4..8].copy_from_slice(&typ.to_le_bytes());
    b[8..16].copy_from_slice(&flags.to_le_bytes());
    // 16..24 sh_addr: always 0, this object is never given a load address of its own.
    b[24..32].copy_from_slice(&offset.to_le_bytes());
    b[32..40].copy_from_slice(&size.to_le_bytes());
    b[40..44].copy_from_slice(&link.to_le_bytes());
    b[44..48].copy_from_slice(&info.to_le_bytes());
    b[48..56].copy_from_slice(&addralign.to_le_bytes());
    b[56..64].copy_from_slice(&entsize.to_le_bytes());
    b
}

/// Builds a HSACO: an ET_REL ELF64 object carrying `spec`'s kernel body, kernel descriptor,
/// entry-point relocation, and `NT_AMDGPU_METADATA` note. Deterministic — the same `HsacoSpec`
/// always serializes to the same bytes, on any host, any number of times, since every field
/// below is a pure function of `spec` (no timestamps, no host paths, no hashmap iteration).
pub fn write_hsaco(spec: &HsacoSpec) -> Result<Vec<u8>, Diag> {
    let kd = kd_bytes(spec);
    let note = note_section_bytes(spec);

    let mut strtab = StrTab::new();
    let name_text = strtab.push(".text");
    let name_rodata = strtab.push(".rodata");
    let name_rela = strtab.push(".rela.rodata");
    let name_note = strtab.push(".note");
    let name_symtab = strtab.push(".symtab");
    let name_strtab = strtab.push(".strtab");
    let sym_name_kernel = strtab.push(&spec.kernel_name);
    let sym_name_kd = strtab.push(&format!("{}.kd", spec.kernel_name));

    // Section order: [0] null [1] .strtab [2] .text [3] .rodata [4] .rela.rodata [5] .note
    // [6] .symtab. `.strtab` doubles as both the section-name and symbol-name string table
    // (one string table, not two), the same economy the reference object makes.
    const SH_STRTAB: u16 = 1;
    const SH_TEXT: u16 = 2;
    const SH_RODATA: u16 = 3;
    const SH_SYMTAB: u16 = 6;
    const SH_NUM: u16 = 7;

    let sym_kernel = sym_entry(
        sym_name_kernel,
        (STB_GLOBAL << 4) | STT_FUNC,
        STV_PROTECTED,
        SH_TEXT,
        0,
        spec.code.len() as u64,
    );
    let sym_kd = sym_entry(
        sym_name_kd,
        (STB_GLOBAL << 4) | STT_OBJECT,
        STV_DEFAULT,
        SH_RODATA,
        0,
        kd.len() as u64,
    );
    let mut symtab = Vec::with_capacity(72);
    symtab.extend_from_slice(&[0u8; 24]); // index 0: the mandatory null symbol.
    symtab.extend_from_slice(&sym_kernel); // index 1.
    symtab.extend_from_slice(&sym_kd); // index 2.

    // The kernel_code_entry_byte_offset field sits at .rodata+0x10; see module header for why
    // this exact (r_offset, addend) pair resolves it to the kernel symbol's own address.
    let mut rela = Vec::with_capacity(24);
    rela.extend_from_slice(&0x10u64.to_le_bytes()); // r_offset
    let r_info = ((1u64) << 32) | R_AMDGPU_REL64 as u64; // symbol index 1 == sym_kernel
    rela.extend_from_slice(&r_info.to_le_bytes());
    rela.extend_from_slice(&0x10i64.to_le_bytes()); // r_addend

    let text_off = round_up(64, 256);
    let text_end = text_off + spec.code.len() as u64;
    let rodata_off = round_up(text_end, 64);
    let rodata_end = rodata_off + kd.len() as u64;
    let rela_off = round_up(rodata_end, 8);
    let rela_end = rela_off + rela.len() as u64;
    let note_off = round_up(rela_end, 4);
    let note_end = note_off + note.len() as u64;
    let symtab_off = round_up(note_end, 8);
    let symtab_end = symtab_off + symtab.len() as u64;
    let strtab_off = symtab_end;
    let strtab_end = strtab_off + strtab.bytes.len() as u64;
    let shoff = round_up(strtab_end, 8);
    let file_size = shoff + SH_NUM as u64 * 64;

    let mut out = vec![0u8; file_size as usize];

    // ELF header.
    out[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    out[4] = 2; // ELFCLASS64
    out[5] = 1; // ELFDATA2LSB
    out[6] = 1; // EV_CURRENT
    out[7] = ELFOSABI_AMDGPU_HSA;
    out[8] = ELFABIVERSION_AMDGPU_HSA_V5;
    // 9..16 e_ident padding: left zero.
    out[16..18].copy_from_slice(&ET_REL.to_le_bytes());
    out[18..20].copy_from_slice(&EM_AMDGPU.to_le_bytes());
    out[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
                                                      // 24..32 e_entry, 32..40 e_phoff: left zero, no program headers.
    out[40..48].copy_from_slice(&shoff.to_le_bytes());
    out[48..52].copy_from_slice(&spec.gfx_arch.mach_code().to_le_bytes());
    out[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
                                                       // 54..56 e_phentsize, 56..58 e_phnum: left zero.
    out[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    out[60..62].copy_from_slice(&SH_NUM.to_le_bytes());
    out[62..64].copy_from_slice(&SH_STRTAB.to_le_bytes());

    out[text_off as usize..text_end as usize].copy_from_slice(&spec.code);
    out[rodata_off as usize..rodata_end as usize].copy_from_slice(&kd);
    out[rela_off as usize..rela_end as usize].copy_from_slice(&rela);
    out[note_off as usize..note_end as usize].copy_from_slice(&note);
    out[symtab_off as usize..symtab_end as usize].copy_from_slice(&symtab);
    out[strtab_off as usize..strtab_end as usize].copy_from_slice(&strtab.bytes);

    let headers: [[u8; 64]; SH_NUM as usize] = [
        shdr(0, 0, 0, 0, 0, 0, 0, 0, 0),
        shdr(
            name_strtab,
            SHT_STRTAB,
            0,
            strtab_off,
            strtab.bytes.len() as u64,
            0,
            0,
            1,
            0,
        ),
        shdr(
            name_text,
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            text_off,
            spec.code.len() as u64,
            0,
            0,
            256,
            0,
        ),
        shdr(
            name_rodata,
            SHT_PROGBITS,
            SHF_ALLOC,
            rodata_off,
            kd.len() as u64,
            0,
            0,
            64,
            0,
        ),
        shdr(
            name_rela,
            SHT_RELA,
            SHF_INFO_LINK,
            rela_off,
            rela.len() as u64,
            SH_SYMTAB as u32,
            SH_RODATA as u32,
            8,
            24,
        ),
        shdr(
            name_note,
            SHT_NOTE,
            SHF_ALLOC,
            note_off,
            note.len() as u64,
            0,
            0,
            4,
            0,
        ),
        shdr(
            name_symtab,
            SHT_SYMTAB,
            0,
            symtab_off,
            symtab.len() as u64,
            SH_STRTAB as u32,
            1, // one local symbol (the mandatory null entry at index 0).
            8,
            24,
        ),
    ];
    for (i, h) in headers.iter().enumerate() {
        let off = shoff as usize + i * 64;
        out[off..off + 64].copy_from_slice(h);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object::read::{Object as ReadObject, ObjectSection, ObjectSymbol};

    fn endpgm_spec() -> HsacoSpec {
        // s_endpgm; encoding: [0x00,0x00,0xb0,0xbf] (see enc.rs's own sopp_fixed_form_opcodes)
        HsacoSpec::new(GfxArch::Gfx1100, "endpgm_test", crate::enc::s_endpgm())
    }

    #[test]
    fn parse_round_trips_every_real_variant_to_its_verified_mach_code() {
        let cases = [
            ("gfx1100", GfxArch::Gfx1100, 0x41),
            ("gfx1101", GfxArch::Gfx1101, 0x46),
            ("gfx1102", GfxArch::Gfx1102, 0x47),
            ("gfx1103", GfxArch::Gfx1103, 0x44),
            ("gfx1150", GfxArch::Gfx1150, 0x43),
            ("gfx1151", GfxArch::Gfx1151, 0x4a),
        ];
        for (name, arch, mach_code) in cases {
            assert_eq!(GfxArch::parse(name), Some(arch), "{name}");
            assert_eq!(arch.mach_code(), mach_code, "{name}");
        }
    }

    #[test]
    fn parse_rejects_confirmed_incompatible_and_nonsense_names() {
        for name in ["gfx1030", "gfx1200", "gfx942", "not-a-target"] {
            assert_eq!(GfxArch::parse(name), None, "{name}");
        }
    }

    #[test]
    fn produces_a_valid_elf64_object() {
        let bytes = write_hsaco(&endpgm_spec()).expect("write succeeds");
        assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F']);
        assert_eq!(bytes[4], 2, "ELFCLASS64");
        assert_eq!(bytes[5], 1, "ELFDATA2LSB");
        assert_eq!(bytes[7], ELFOSABI_AMDGPU_HSA);
        assert_eq!(bytes[8], ELFABIVERSION_AMDGPU_HSA_V5);
        assert_eq!(u16::from_le_bytes([bytes[16], bytes[17]]), ET_REL);
        assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), EM_AMDGPU);
        assert_eq!(
            u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]),
            0x41,
            "e_flags carries EF_AMDGPU_MACH_AMDGCN_GFX1100"
        );

        let file = object::read::File::parse(&*bytes).expect("parses as an object file");
        assert_eq!(file.format(), object::BinaryFormat::Elf);

        let text = file.section_by_name(".text").expect(".text present");
        assert_eq!(text.data().unwrap(), &[0x00, 0x00, 0xb0, 0xbf]);

        let rodata = file.section_by_name(".rodata").expect(".rodata present");
        assert_eq!(
            rodata.data().unwrap().len(),
            64,
            "kernel descriptor is 64 bytes"
        );

        let note = file.section_by_name(".note").expect(".note present");
        assert!(
            note.data().unwrap().len() > 12,
            "note carries a real descriptor"
        );

        let kernel_sym = file
            .symbols()
            .find(|s| s.name() == Ok("endpgm_test"))
            .expect("kernel symbol present");
        assert_eq!(kernel_sym.size(), 4);

        let kd_sym = file
            .symbols()
            .find(|s| s.name() == Ok("endpgm_test.kd"))
            .expect("kernel descriptor symbol present");
        assert_eq!(kd_sym.size(), 64);
    }

    #[test]
    fn kernel_descriptor_carries_wavefront_size32() {
        let bytes = write_hsaco(&endpgm_spec()).unwrap();
        let file = object::read::File::parse(&*bytes).unwrap();
        let rodata = file.section_by_name(".rodata").unwrap().data().unwrap();
        let code_properties = u16::from_le_bytes([rodata[56], rodata[57]]);
        assert_eq!(
            code_properties & (1 << 10),
            1 << 10,
            "ENABLE_WAVEFRONT_SIZE32 must always be set for gfx1100"
        );
    }

    #[test]
    fn kernel_code_entry_relocation_resolves_to_the_kernel_symbol() {
        let bytes = write_hsaco(&endpgm_spec()).unwrap();
        let file = object::read::File::parse(&*bytes).unwrap();
        let rodata = file.section_by_name(".rodata").unwrap();
        let (_, reloc) = rodata
            .relocations()
            .next()
            .expect("one relocation on .rodata");
        assert_eq!(reloc.addend(), 0x10);
    }

    #[test]
    fn same_spec_produces_byte_identical_output() {
        let spec = endpgm_spec();
        let a = write_hsaco(&spec).unwrap();
        let b = write_hsaco(&spec).unwrap();
        assert_eq!(
            a, b,
            "determinism: same spec must yield byte-identical output"
        );
    }

    #[test]
    fn kernarg_segment_enables_kernarg_sgpr_pointer() {
        let spec = HsacoSpec::new(GfxArch::Gfx1100, "with_args", crate::enc::s_endpgm())
            .with_kernarg_segment(16, 8);
        let bytes = write_hsaco(&spec).unwrap();
        let file = object::read::File::parse(&*bytes).unwrap();
        let rodata = file.section_by_name(".rodata").unwrap().data().unwrap();
        let code_properties = u16::from_le_bytes([rodata[56], rodata[57]]);
        assert_eq!(code_properties & (1 << 3), 1 << 3);
        let kernarg_size = u32::from_le_bytes([rodata[8], rodata[9], rodata[10], rodata[11]]);
        assert_eq!(kernarg_size, 16);
    }

    #[test]
    fn workgroup_ids_set_the_matching_rsrc2_bits() {
        let spec = HsacoSpec::new(GfxArch::Gfx1100, "with_bid", crate::enc::s_endpgm())
            .with_workgroup_ids(true, false, true);
        let bytes = write_hsaco(&spec).unwrap();
        let file = object::read::File::parse(&*bytes).unwrap();
        let rodata = file.section_by_name(".rodata").unwrap().data().unwrap();
        let rsrc2 = u32::from_le_bytes([rodata[52], rodata[53], rodata[54], rodata[55]]);
        assert_eq!(rsrc2 & (1 << 7), 1 << 7, "workgroup id x enabled");
        assert_eq!(rsrc2 & (1 << 8), 0, "workgroup id y left disabled");
        assert_eq!(rsrc2 & (1 << 9), 1 << 9, "workgroup id z enabled");
    }
}
