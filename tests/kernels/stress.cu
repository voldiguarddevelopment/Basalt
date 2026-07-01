// Register-pressure stress kernel: eighteen independent float temporaries, each folding
// three array elements together, are all computed before any of them is consumed — so at
// the point the final reduction starts, all eighteen are simultaneously live. That is far
// past either backend's real register budget (the regalloc backend's float pool is 5 XMM
// registers; see basalt-x86/src/regalloc.rs), forcing genuine spilling in both, while still
// being small enough to hand-verify arithmetically (see examples/cpu_launch_stress.c).
__global__ void stress(const float *a, float *out, int n) {
    int i = threadIdx.x;
    if (i < n) {
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
        float sum = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 +
                    t14 + t15 + t16 + t17;
        out[i] = sum;
    }
}
