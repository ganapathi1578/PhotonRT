# MobileCLIP in PhotonRT

## Role

MobileCLIP-S1 is PhotonRT's visual encoder.

Its output is the visual representation consumed by the Photon decoder pipeline.

```text
image
  |
  v
MobileCLIP-S1
  |
  v
512-D embedding
  |
  v
L2 normalization
  |
  v
Photon prefix projection
```

## Runtime artifact

The ONNX runtime file is:

```text
mobileclip-s1.onnx
```

## Python usage

Normal users do not need to invoke MobileCLIP directly.

```python
from photonrt import Captioner

model = Captioner.from_pretrained()

print(model.caption("image.jpg"))
```

## Native path

The C++ runtime loads MobileCLIP through ONNX Runtime.

The image-captioning path performs:

1. image loading
2. preprocessing
3. MobileCLIP inference
4. embedding normalization
5. Photon generation
6. token decoding

## Raw frame path

For streaming applications, the native binding supports:

```python
runtime.caption_rgb(frame, options)
runtime.caption_bgr(frame, options)
```

The expected frame format is:

```text
shape = (height, width, 3)
dtype = uint8
```

## BGR frames

OpenCV normally returns BGR images.

Use:

```python
runtime.caption_bgr(frame, options)
```

so the native path performs the required channel conversion.

## RGB frames

For RGB data:

```python
runtime.caption_rgb(frame, options)
```

## Performance

MobileCLIP is part of the end-to-end inference time.

Camera applications should benchmark:

```text
capture
preprocessing
MobileCLIP
Photon prefill
Photon decode
token decode
```

rather than assuming the decoder alone determines latency.
