# PhotonRT roadmap

## v0.2 — native vision validation

1. Validate `mobileclip_s1.pt -> mobileclip-s1.f32.mclip`.
2. Compare native 512-D embeddings with Apple's PyTorch reference.
3. Run full native `MobileCLIP -> Photon -> caption` path.

## v0.3 — CPU optimization

1. Multithreaded convolution.
2. AVX2/AVX-512 kernels on x86.
3. ARM NEON kernels.
4. GEMM packing for pointwise layers.
5. Operator fusion.

## v0.4 — model formats and compression

1. mmap-capable model loading.
2. F16 weights.
3. Q8.
4. Q4.
5. Quality regression suite.

## v0.5 — GPU

1. CUDA backend.
2. CUDA convolution/GEMM.
3. CUDA attention.
4. GPU/CPU backend parity.

## v1.0 — library release

1. Stable C API.
2. C++ API.
3. Python bindings.
4. Native JPEG/PNG decoding.
5. Installable packages.
6. Reproducible benchmarks.
7. Model cards and license notices.
