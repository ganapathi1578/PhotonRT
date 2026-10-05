# MobileCLIP-S1 native CPU path

Photon uses the MobileCLIP-S1 image encoder to turn an image into a 512-D vector before the Photon decoder. The native encoder in this repository is a C++ CPU implementation of the **reparameterized** MobileCLIP-S1 image path.

## What is native

```text
C++ / PhotonRT
    image tensor (3x256x256, float32, CHW)
       -> MobileCLIP-S1
       -> 512-D embedding
       -> L2 normalization (optional)
```

The implementation contains the inference-time FastViT/MCI-1 graph: MobileOne/ReparamLargeKernel fused convolutions, RepMixer stages, BatchNorm folding, the final MHSA stage, SE blocks, CPE, global average pooling, and the 512-D image projection.

The official MCI-1 configuration is 4 / 12 / 20 / 4 blocks with channel sizes 64 / 128 / 256 / 512, and the MobileCLIP-S1 config exposes a 512-D image embedding and 256x256 input. See the upstream Apple sources for the exact model definition. 

## Export

The exporter uses the official Python implementation to load `mobileclip_s1.pt` and fold reparameterizable branches before saving `mobileclip-s1.f32.mclip`.

```bash
python tools/convert_mobileclip.py \
  --checkpoint "C:/path/to/mobileclip_s1.pt" \
  --output mobileclip-s1.f32.mclip
```

Do not commit or redistribute Apple's pretrained weights unless their applicable model terms permit it.

## Exact-reference preprocessing

For parity, use the upstream MobileCLIP preprocessing path first:

```bash
python tools/preprocess_mobileclip.py \
  --image "C:/path/to/image.jpg" \
  --output image_chw.f32
```

The released Python helper currently performs resize, center crop and `ToTensor()` for MobileCLIP v1; it does not add an ImageNet normalization step.

## Native embedding

```bash
./build/Release/mobileclip-embedding.exe \
  mobileclip-s1.f32.mclip \
  image_chw.f32
```

This writes `mobileclip_embedding.f32`.

## Parity test

```bash
python tools/compare_mobileclip.py \
  --checkpoint "C:/path/to/mobileclip_s1.pt" \
  --image-tensor image_chw.f32 \
  --native-embedding mobileclip_embedding.f32
```

Inspect:

- max absolute error
- mean absolute error
- cosine similarity
- both vector norms

## End-to-end native C++ path

```bash
./build/Release/image-caption.exe \
  mobileclip-s1.f32.mclip \
  photon.f32.photon \
  photon.tokenizer \
  image_chw.f32
```

The image decoder is deliberately not bundled yet. The runtime accepts a preprocessed tensor so that MobileCLIP numerical parity can be established before introducing JPEG/PNG decoding.
