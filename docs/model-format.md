# PhotonRT v1 model format

The first runtime release uses a deliberately simple little-endian FP32 binary format.

```text
8 bytes   PHOTON01
u32       version
u32       vocab_size
u32       d_model
u32       num_layers
u32       num_heads
u32       ffn_hidden
u32       image_embedding_dim
u32       num_image_tokens
u32       max_length
f32       rope_theta
f32       rms_norm_eps
u32       BOS id
u32       EOS id
u32       tensor_count

for each tensor:
    u32   name length
    bytes name (UTF-8)
    u32   ndim
    u64[] shape
    u64   element count
    f32[] contiguous tensor data
```

The tokenizer sidecar format uses the `PHOTOKN1` magic and stores decoded token pieces so the native runtime can decode generated ids without requiring Rust/Python.

A future version should add GGUF import/export once FP32 CPU numerical parity is established.
