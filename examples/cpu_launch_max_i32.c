/* Host-side proof for the hand-built `max_i32_via_phi` BIR fixture (see
 * crates/basalt-x86/tests/link_and_run_regalloc.rs): `if (a > b) m = a; else m = b; return m;`,
 * lowered to a real `phi` at the merge block. Both params are integer-class (a in edi, b in
 * esi), so nthreads is the third integer register (rdx).
 */
#include <stdint.h>
#include <stdio.h>

extern int32_t max_i32(int32_t a, int32_t b, int64_t nthreads);

static int check(int32_t a, int32_t b, int64_t nthreads) {
    int32_t expected = a > b ? a : b;
    int32_t got = max_i32(a, b, nthreads);
    if (got != expected) {
        fprintf(stderr, "FAIL: max_i32(%d, %d, nthreads=%lld) expected %d, got %d\n", a, b,
                (long long)nthreads, expected, got);
        return 0;
    }
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= check(3, 4, 1);
    ok &= check(4, 3, 1);
    ok &= check(-5, 10, 1000);
    ok &= check(-1, -1, 1);
    ok &= check(7, 7, 5);

    if (!ok) {
        return 1;
    }
    printf("PASS: max_i32 returned the correct maximum across all cases\n");
    return 0;
}
