/* virt32.ld — places a bare-metal RV32IM image inside qemu-system-riscv32's "-M virt" board
 * DRAM region, for run_kernel.sh's execution harness (P12-T2). 0x80000000 is the "virt"
 * machine's real, fixed DRAM base (see QEMU's hw/riscv/virt.c memmap) — a kernel linked
 * anywhere else in the board's other MMIO ranges (the boot ROM at 0x1000, VIRT_TEST at
 * 0x100000, UART at 0x10000000, ...) never gets a live CPU pointed at it, since "-kernel"
 * with an ELF file boots straight to the ELF's own entry address with no relocation. This
 * was empirically confirmed: a build linked at newlib's own default addresses (~0x10000,
 * meant for a different bare-metal target entirely) loads cleanly but the core never
 * executes a single instruction of it — that address isn't backed by RAM on this board.
 *
 * .init_array/.fini_array are their own output sections (not folded into .data) because
 * start.S's call to __libc_init_array walks __init_array_start.._end by symbol, and that
 * symbol pair only exists if something in the script defines it — libsemihost.a's own
 * ":tt" console-fd setup (init_semihosting) is registered as a constructor in exactly this
 * array, so omitting this section silently drops semihosting console I/O entirely (no
 * console fd ever gets opened) without any link or run error to point at why.
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
  . = ORIGIN(RAM);
  .text : { *(.text.init) *(.text*) } > RAM
  .rodata : { *(.rodata*) } > RAM

  . = ALIGN(4);
  PROVIDE(__global_pointer$ = . + 0x800);

  .init_array : {
    __init_array_start = .;
    *(.init_array*)
    __init_array_end = .;
  } > RAM
  .fini_array : {
    __fini_array_start = .;
    *(.fini_array*)
    __fini_array_end = .;
  } > RAM

  .data : { *(.data*) *(.sdata*) } > RAM

  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss*) *(.sbss*) *(COMMON)
    __bss_end = .;
  } > RAM
  _end = .;

  /* 16 KiB stack, fixed and never checked for overflow — every proof kernel this harness
   * runs is small enough that this is not a real constraint (see run_kernel.sh's header).
   */
  . = ALIGN(16);
  . = . + 0x4000;
  _stack_top = .;
}
