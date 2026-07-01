/* Host-side proof for the hand-built `spill_heavy_sum` BIR fixture (see
 * crates/basalt-x86/tests/link_and_run_regalloc.rs), which deliberately forces the regalloc
 * backend's spill-slot load/store path: 5 int32 params, all int-class, land in
 * rdi/rsi/rdx/rcx/r8, so nthreads is the 6th integer register (r9). The function itself
 * computes p0+p1+p2+p3+p4.
 */
#include <stdint.h>
#include <stdio.h>

extern int32_t spill_heavy_sum(int32_t p0, int32_t p1, int32_t p2, int32_t p3, int32_t p4,
                                int64_t nthreads);

static int check(int32_t p0, int32_t p1, int32_t p2, int32_t p3, int32_t p4, int64_t nthreads) {
    int32_t expected = p0 + p1 + p2 + p3 + p4;
    int32_t got = spill_heavy_sum(p0, p1, p2, p3, p4, nthreads);
    if (got != expected) {
        fprintf(stderr, "FAIL: spill_heavy_sum(%d,%d,%d,%d,%d, nthreads=%lld) expected %d, got %d\n",
                p0, p1, p2, p3, p4, (long long)nthreads, expected, got);
        return 0;
    }
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= check(1, 2, 3, 4, 5, 1);
    ok &= check(1, 2, 3, 4, 5, 8);
    ok &= check(-10, 20, -30, 40, -50, 1);
    ok &= check(1000000, -2000000, 3000000, -4000000, 5000000, 4);

    if (!ok) {
        return 1;
    }
    printf("PASS: spill_heavy_sum returned the correct sum across all cases\n");
    return 0;
}
