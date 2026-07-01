/* Host-side proof for the hand-built `add_i32` BIR fixture (see
 * crates/basalt-x86/tests/link_and_run.rs), isolating the oracle's scalar-return ABI path from
 * everything the vector_add proof already covers: both params are integer-class (a in edi, b
 * in esi), so nthreads is the third integer register (rdx), and the i32 result comes back in
 * eax. Run at a few different thread counts to confirm the result a function computes from its
 * own params alone does not depend on how many redundant loop iterations the oracle runs it
 * for.
 */
#include <stdint.h>
#include <stdio.h>

extern int32_t add_i32(int32_t a, int32_t b, int64_t nthreads);

static int check(int32_t a, int32_t b, int64_t nthreads, int32_t expected) {
    int32_t got = add_i32(a, b, nthreads);
    if (got != expected) {
        fprintf(stderr, "FAIL: add_i32(%d, %d, nthreads=%lld) expected %d, got %d\n", a, b,
                (long long)nthreads, expected, got);
        return 0;
    }
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= check(3, 4, 1, 7);
    ok &= check(3, 4, 1000, 7);
    ok &= check(-5, 10, 1, 5);
    ok &= check(-1, -1, 1, -2);

    if (!ok) {
        return 1;
    }
    printf("PASS: add_i32 returned the correct sum across all cases\n");
    return 0;
}
