# PhotonRT Python API Reference

## Public imports

```python
from photonrt import (
    Captioner,
    CaptionerNative,
    CaptionOptions,
    CameraCaptioner,
)
```

## `Captioner`

High-level image-captioning interface.

### Constructor

```python
Captioner(
    mobileclip_model,
    photon_prefill_model,
    photon_decode_model,
    tokenizer,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `mobileclip_model` | `str` / `Path` | MobileCLIP ONNX model |
| `photon_prefill_model` | `str` / `Path` | Photon prefill ONNX model |
| `photon_decode_model` | `str` / `Path` | Photon decode ONNX model |
| `tokenizer` | `str` / `Path` | Photon tokenizer |

### `from_pretrained`

```python
Captioner.from_pretrained(
    repo_id="ganapathi1578/PhotonRT",
    revision="main",
    cache_dir=None,
)
```

The public 0.2.0 workflow also supports:

```python
model = Captioner.from_pretrained()
```

The loader downloads the runtime model files from Hugging Face and then constructs the native runtime.

### `caption`

```python
model.caption(
    image,
    max_new_tokens=32,
)
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `image` | `str` / `Path` | required | Image path |
| `max_new_tokens` | `int` | `32` | Maximum generated tokens |

### Errors

Missing image:

```text
FileNotFoundError
```

Invalid generation length:

```text
ValueError
```

---

## `CaptionOptions`

```python
options = CaptionOptions()
options.max_new_tokens = 32
```

### Field

| Field | Default | Description |
|---|---:|---|
| `max_new_tokens` | `32` | Maximum generated token count |

---

## `CaptionerNative`

```python
runtime = CaptionerNative(
    mobileclip_model,
    photon_prefill_model,
    photon_decode_model,
    tokenizer,
)
```

### Methods

```python
runtime.caption(image, options)
runtime.caption_rgb(frame, options)
runtime.caption_bgr(frame, options)
```

### Frame contract

```text
dtype = uint8
shape = (height, width, 3)
```

---

## `CameraCaptioner`

```python
CameraCaptioner(
    source,
    mobileclip_model,
    photon_prefill_model,
    photon_decode_model,
    tokenizer,
    *,
    workers=2,
    queue_size=8,
    frame_stride=1,
    max_new_tokens=32,
    drop_oldest=True,
    capture_api=None,
)
```

### Constructor parameters

| Name | Default | Description |
|---|---:|---|
| `source` | required | OpenCV source |
| `mobileclip_model` | required | MobileCLIP model |
| `photon_prefill_model` | required | Photon prefill model |
| `photon_decode_model` | required | Photon decoder |
| `tokenizer` | required | Photon tokenizer |
| `workers` | `2` | Number of independent native workers |
| `queue_size` | `8` | Maximum pending task count |
| `frame_stride` | `1` | Process every Nth frame |
| `max_new_tokens` | `32` | Maximum caption token count |
| `drop_oldest` | `True` | Fresh-frame queue policy |
| `capture_api` | `None` | Optional OpenCV backend ID |

### Methods

```python
camera.start()
camera.stop()
camera.get_result(timeout=None)
camera.stats()
```

### Iterator

```python
for result in camera:
    ...
```

---

## `CaptionResult`

Fields:

```python
result.frame_id
result.capture_time_ns
result.caption
result.inference_ms
result.total_latency_ms
```

---

## Version

```python
import photonrt

print(photonrt.__version__)
```

Current release:

```text
0.2.0
```

---

## Model loading flow

```text
Captioner.from_pretrained()
        |
        v
Hugging Face snapshot/cache
        |
        +-- mobileclip-s1.onnx
        +-- photon_prefill.onnx
        +-- photon.onnx
        +-- photon.tokenizer
        |
        v
Native Captioner
```

