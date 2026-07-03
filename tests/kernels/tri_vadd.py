# Masked Triton vector-add: c[i] = a[i] + b[i] for i in [0, n), same kernel
# `crates/basalt-x86/tests/triton_link_and_run.rs`'s `masked_triton_vector_add_links_and_runs`
# already proves correct through the real `parse -> check_triton -> lower_triton -> X86Oracle`
# pipeline (as `MASKED_VECTOR_ADD`) — kept byte-for-byte identical here so this file's own
# proof (via `--triton --cpu`/`--triton --nvidia-ptx`) is provably the same kernel, not a
# lookalike. `BLOCK_SIZE` need not divide `n`: `mask` guards every out-of-bounds lane on both
# the load and the store.
import triton
import triton.language as tl


@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
