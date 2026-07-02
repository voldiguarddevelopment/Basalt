/* Host-side proof that the x86-64 oracle's emitted object for mymathhomework.cu links and
 * runs, and that its output matches the hand-computed result of the kernel's own arithmetic:
 * a = 2 + 3 = 5, b = a * 4 = 20, and the unused `a + b` temporary contributes nothing to the
 * result (it is never read).
 *
 * mymathhomework's only BIR param is (ptr.global), so under SysV it lands in rdi; the oracle's
 * calling convention always appends one trailing integer-class `nthreads` argument after a
 * function's own params, landing in rsi here and read back a full 8 bytes, hence int64_t.
 */
#include <stdint.h>
#include <stdio.h>

extern void mymathhomework(int *out, int64_t nthreads);

int main(void) {
    int out[1] = {-1};

    mymathhomework(out, (int64_t)1);

    int expected = 20;
    if (out[0] != expected) {
        fprintf(stderr, "FAIL: expected %d, got %d\n", expected, out[0]);
        return 1;
    }

    printf("PASS: mymathhomework produced %d\n", out[0]);
    return 0;
}
