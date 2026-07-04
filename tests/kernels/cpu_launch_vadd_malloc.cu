// Real end-to-end host+device compile proof (P13-T1c-ii): a genuine, fully self-contained host
// function that allocates its own device buffers rather than relying on a host-side test driver
// to pre-allocate them, closing the gap `cpu_launch_vadd_host.cu` (P13-T1c-i) documented and
// deliberately sidestepped. `vector_add` is `vector_add.cu`'s own unmodified kernel; `d_a`/`d_b`/
// `d_c` are ordinary local pointer variables (not arrays — a local *array*'s real multi-byte
// stack home remains the separate, general `Frame` gap P13-T1c-i flagged and this task does not
// touch), each a real `cudaMalloc`'d buffer copied into via real `cudaMemcpy`, read/written by
// the launched kernel, copied back, and `cudaFree`'d.
#include "vector_add.cu"

void launch_vector_add_malloc(const float *h_a, const float *h_b, float *h_c, int n) {
    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
