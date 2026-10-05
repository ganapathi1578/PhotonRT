# PhotonRT

PhotonRT is a CPU-first native inference runtime for the Photon image-captioning model and its MobileCLIP-S1 image encoder.

## Current pipeline

```text
image (preprocessed CHW float32)
        |
        v
MobileCLIP-S1 native CPU encoder
        |
       512-D
        |
        v
Photon projector
        |
      8 x 256
        |
        v
Photon decoder (6 layers)
        |
        v
caption
```

The Photon decoder is implemented in C++ with a KV-cache generation path. MobileCLIP-S1 is implemented as a reparameterized FastViT/MCI-1 CPU graph. The upstream MobileCLIP repository defines S1 as a 256-pixel image model with a 512-D embedding and MCI-1 image encoder; see the official Apple implementation for model details.

## Build (Windows + Visual Studio generator)

```bash
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

## Convert Photon checkpoint

```bash
python tools/convert_photon.py \
  --checkpoint "C:/path/to/nano_ep55.pt" \
  --tokenizer "C:/path/to/mobilecap_tokenizer.json" \
  --model-out photon.f32.photon \
  --tokenizer-out photon.tokenizer
```

## Convert MobileCLIP-S1

Use Apple's official MobileCLIP Python package for conversion:

```bash
python tools/convert_mobileclip.py \
  --checkpoint "C:/path/to/mobileclip_s1.pt" \
  --output mobileclip-s1.f32.mclip
```

The converter writes reparameterized inference weights and does not bundle the original Apple checkpoint.

## Prepare an exact-reference MobileCLIP input

```bash
python tools/preprocess_mobileclip.py \
  --image "C:/path/to/image.jpg" \
  --output image_chw.f32
```

## Run native MobileCLIP

```bash
./build/Release/mobileclip-embedding.exe \
  mobileclip-s1.f32.mclip \
  image_chw.f32
```

## Compare native MobileCLIP against PyTorch

```bash
python tools/compare_mobileclip.py \
  --checkpoint "C:/path/to/mobileclip_s1.pt" \
  --image-tensor image_chw.f32 \
  --native-embedding mobileclip_embedding.f32
```

## Run end-to-end C++ image captioning

```bash
./build/Release/image-caption.exe \
  mobileclip-s1.f32.mclip \
  photon.f32.photon \
  photon.tokenizer \
  image_chw.f32
```

## Important release note

PhotonRT can distribute its own runtime code under its chosen open-source license. Apple's MobileCLIP source code is MIT, while the pretrained MobileCLIP model weights are governed by Apple's model terms; do not bundle Apple's weights into a release without checking those terms.

## Status

- Photon C++ CPU reference runtime: working
- real Photon checkpoint conversion: working
- Photon generation: working
- native MobileCLIP-S1 CPU graph: implemented
- PyTorch/native MobileCLIP parity: next validation target
- optimized SIMD kernels: next
- quantization: next
- CUDA kernels: next
- JPEG/PNG native image decoding: next
- Python bindings: next
