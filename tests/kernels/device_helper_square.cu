// Real end-to-end proof of a __device__ helper called from a __global__ kernel via a genuine
// `Op::Call` (P13-T-calls-i): `square` computes one element's result, `square_vector` calls it
// once per thread. Every element uses a different, checkable value (including negatives, to
// exercise the same signed multiply `square`'s own `mul` would use either way) so a wrong
// argument/return marshaling would show up as a mismatch, not an accidental match.
__device__ int square(int x) {
    return x * x;
}

__global__ void square_vector(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = square(in[i]);
    }
}
