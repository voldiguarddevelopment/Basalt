/* Host-side proof for tests/kernels/stress.cu: hand-computes the same eighteen-temporary
 * fold in plain C and compares it against the kernel's output. stress's BIR params are
 * (ptr.global, ptr.global, i32) — all integer-class, so under SysV they land in rdi/rsi/rdx
 * and the trailing nthreads argument (see the calling-convention note in
 * crates/basalt-x86/tests/link_and_run.rs) lands in rcx, read back a full 8 bytes hence
 * int64_t.
 */
#include <stdint.h>
#include <stdio.h>

extern void stress(const float *a, float *out, int n, int64_t nthreads);

int main(void) {
    float a[20];
    for (int i = 0; i < 20; i++) {
        a[i] = (float)(i + 1) * 0.5f - 3.0f;
    }
    float out[1] = {-1.0f};

    stress(a, out, 1, (int64_t)1);

    float t0 = a[0] * a[1] + a[2];
    float t1 = a[3] * a[4] + a[5];
    float t2 = a[6] * a[7] + a[8];
    float t3 = a[9] * a[10] + a[11];
    float t4 = a[12] * a[13] + a[14];
    float t5 = a[15] * a[16] + a[17];
    float t6 = a[18] * a[19] + a[0];
    float t7 = a[1] * a[2] + a[3];
    float t8 = a[4] * a[5] + a[6];
    float t9 = a[7] * a[8] + a[9];
    float t10 = a[10] * a[11] + a[12];
    float t11 = a[13] * a[14] + a[15];
    float t12 = a[16] * a[17] + a[18];
    float t13 = a[19] * a[0] + a[1];
    float t14 = a[2] * a[3] + a[4];
    float t15 = a[5] * a[6] + a[7];
    float t16 = a[8] * a[9] + a[10];
    float t17 = a[11] * a[12] + a[13];
    float expected = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 +
                     t14 + t15 + t16 + t17;

    if (out[0] != expected) {
        fprintf(stderr, "FAIL: stress produced %f, expected %f\n", (double)out[0],
                (double)expected);
        return 1;
    }

    printf("PASS: stress matched the hand-computed fold: %f\n", (double)expected);
    return 0;
}
