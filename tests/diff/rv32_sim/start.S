/* start.S — the real entry point for run_kernel.sh's bare-metal images, replacing
 * riscv64-elf-newlib's own crt0.o rather than patching around it.
 *
 * That crt0.o's own `_start` (verified by disassembly: `riscv64-elf-objdump -dr
 * .../rv32im/ilp32/crt0.o`) never initializes `sp` or `gp` — it assumes something upstream
 * of it already did, which is true on whatever board that multilib's crt0 actually targets
 * but not true of qemu-system-riscv32's "-M virt" board booted via "-bios none -kernel": the
 * CPU resets straight into the ELF's entry address with general-purpose registers zeroed.
 * Without this file, `_start` reads argc off `sp==0` and every later stack access faults
 * silently (no output, no logged guest error, just a dead core) — this was the actual first
 * failure mode hit standing this harness up, before gp/sp init was traced as the cause.
 *
 * Otherwise this mirrors crt0.o's own body exactly (zero .bss, run __libc_init_array, call
 * main, call exit) so libc's semihosting console/file layer — which registers its own
 * ":tt" fd setup as a .init_array constructor — initializes the same way it would under a
 * normal startup.
 */
.section .text.init
.global _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, _stack_top

  la a0, __bss_start
  la a1, __bss_end
1:
  bgeu a0, a1, 2f
  sw zero, 0(a0)
  addi a0, a0, 4
  j 1b
2:
  call __libc_init_array
  call main
  call exit
3:
  j 3b
