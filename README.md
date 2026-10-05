# PhotonRT

<p align="center">
  <h1 align="center">PhotonRT</h1>
  <p align="center">
    Efficient native C++ image-captioning runtime with Python bindings.
  </p>
</p>

<p align="center">

[![PyPI](https://img.shields.io/pypi/v/photonrt.svg)](https://pypi.org/project/photonrt/)
[![Python](https://img.shields.io/pypi/pyversions/photonrt.svg)](https://pypi.org/project/photonrt/)
[![License](https://img.shields.io/github/license/ganapathi1578/PhotonRT.svg)](LICENSE)

</p>

 PhotonRT is an optimized native C++ implementation of
  <strong>Photon: Efficient Prefix-Conditioned Image Captioning with Lightweight Transformer Decoding</strong>,
  designed for efficient real-world inference and deployment.<br>
  It provides a high-performance C++ runtime with Python bindings,
  ONNX Runtime execution, and support for both image captioning and
  real-time camera-stream captioning.<br>
  
<p align="center">

###### 📄 **Paper:** [Link](https://link.springer.com/article/10.1007/s10994-026-07072-4) | 💻 **Code:** [GitHub](https://github.com/visual-positioning/Photon-12M)

</p>

The Python package provides:

- `Captioner` for single-image captioning
- `CaptionerNative` for direct native bindings
- `CaptionOptions` for native generation settings
- `CameraCaptioner` for webcam/video/stream captioning
- automatic Hugging Face model retrieval for the high-level `Captioner` API

The repository also contains native C++ code, examples, tests, model-export tools, and deeper documentation.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hugging Face Models](#hugging-face-models)
- [Captioner API](#captioner-api)
- [Native API](#native-api)
- [Raw Frame API](#raw-frame-api)
- [Camera Streaming](#camera-streaming)
- [Configuration Reference](#configuration-reference)
- [Latency and Queue Behavior](#latency-and-queue-behavior)
- [Architecture](#architecture)
- [Building from Source](#building-from-source)
- [Repository Layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

---

## Installation

### Python

```bash
pip install photonrt
```

Check:

```bash
python -c "import photonrt; print(photonrt.__version__)"
```

For PhotonRT 0.2.0:

```text
0.2.0
```

### Camera streaming

Install OpenCV separately for the camera/video API:

```bash
pip install opencv-python
```

OpenCV is only needed by the Python camera capture layer. Single-image captioning through `Captioner` does not require OpenCV.

---

## Quick Start

### Caption one image

```python
from photonrt import Captioner

model = Captioner.from_pretrained()

caption = model.caption("image.jpg")

print(caption)
```

The first call downloads the required PhotonRT runtime files when they are not already cached.

### Control generation length

```python
from photonrt import Captioner

model = Captioner.from_pretrained()

caption = model.caption(
    "image.jpg",
    max_new_tokens=48,
)

print(caption)
```

`max_new_tokens` must be greater than zero.

---

## Hugging Face Models

The public PhotonRT model repository is:

```text
https://huggingface.co/ganapathi1578/PhotonRT
```

The runtime uses these artifacts:

| File | Purpose |
|---|---|
| `mobileclip-s1.onnx` | MobileCLIP-S1 image encoder |
| `photon_prefill.onnx` | Photon prefill graph |
| `photon.onnx` | Photon autoregressive decoder |
| `photon.tokenizer` | Photon tokenizer |
| `config.json` | Optional model/runtime metadata |

### Automatic loading

```python
from photonrt import Captioner

model = Captioner.from_pretrained()
```

### Explicit repository

```python
model = Captioner.from_pretrained(
    repo_id="ganapathi1578/PhotonRT",
    revision="main",
)
```

### Cache directory

```python
from pathlib import Path

model = Captioner.from_pretrained(
    repo_id="ganapathi1578/PhotonRT",
    revision="main",
    cache_dir=Path("D:/photonrt-cache"),
)
```

### Model loading parameters

| Parameter | Type | Meaning |
|---|---|---|
| `repo_id` | `str` | Hugging Face repository |
| `revision` | `str` | Branch, tag, or commit |
| `cache_dir` | `str` / `Path` / `None` | Optional cache location |

For reproducible deployments, prefer an immutable model revision rather than a moving branch.

---

# Captioner API

```python
from photonrt import Captioner
```

## Constructor

```python
Captioner(
    mobileclip_model,
    photon_prefill_model,
    photon_decode_model,
    tokenizer,
)
```

| Parameter | Type | Description |
|---|---|---|
| `mobileclip_model` | `str` / `Path` | Path to MobileCLIP ONNX model |
| `photon_prefill_model` | `str` / `Path` | Path to Photon prefill ONNX model |
| `photon_decode_model` | `str` / `Path` | Path to Photon decode ONNX model |
| `tokenizer` | `str` / `Path` | Path to Photon tokenizer |

Example:

```python
from photonrt import Captioner

model = Captioner(
    mobileclip_model="models/fp32/mobileclip-s1.onnx",
    photon_prefill_model="models/fp32/photon_prefill.onnx",
    photon_decode_model="models/fp32/photon.onnx",
    tokenizer="models/fp32/photon.tokenizer",
)
```

## `caption()`

```python
model.caption(
    image,
    max_new_tokens=32,
)
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `image` | `str` / `Path` | required | Image file |
| `max_new_tokens` | `int` | `32` | Maximum generated token count |

Example:

```python
caption = model.caption("image.jpg")
```

The high-level wrapper validates that the image exists and that `max_new_tokens > 0`. The actual image loading, preprocessing, MobileCLIP execution, Photon inference, and token decoding are performed by the native runtime.

---

# Native API

```python
from photonrt import CaptionerNative, CaptionOptions
```

## `CaptionOptions`

```python
options = CaptionOptions()
options.max_new_tokens = 32
```

| Field | Type | Default | Description |
|---|---|---:|---|
| `max_new_tokens` | integer | `32` | Maximum generated token count |

## Direct native inference

```python
from photonrt import CaptionerNative, CaptionOptions

runtime = CaptionerNative(
    "models/fp32/mobileclip-s1.onnx",
    "models/fp32/photon_prefill.onnx",
    "models/fp32/photon.onnx",
    "models/fp32/photon.tokenizer",
)

options = CaptionOptions()
options.max_new_tokens = 32

caption = runtime.caption(
    "image.jpg",
    options,
)

print(caption)
```

---

# Raw Frame API

The native binding also accepts raw NumPy image frames:

```python
runtime.caption_rgb(frame, options)
runtime.caption_bgr(frame, options)
```

Expected input:

```text
shape = (height, width, 3)
dtype = uint8
```

### OpenCV / BGR

```python
import cv2

frame = cv2.imread("image.jpg")

caption = runtime.caption_bgr(
    frame,
    options,
)

print(caption)
```

### RGB

```python
caption = runtime.caption_rgb(
    frame_rgb,
    options,
)
```

`caption_bgr()` is useful with OpenCV because OpenCV normally returns BGR images.

---

# Camera Streaming

PhotonRT provides a bounded-queue camera/video captioning API.

The current v1 design uses:

```text
1 capture thread
+
N inference worker threads
```

The Python camera layer captures frames with OpenCV, while each worker invokes the native C++ inference runtime.

```text
OpenCV VideoCapture
        |
        v
Frame sampling
        |
        v
Bounded task queue
        |
        +---- Worker 0 -> native PhotonRT
        +---- Worker 1 -> native PhotonRT
        +---- ...
        |
        v
Bounded result queue
        |
        v
CaptionResult
```

## Basic webcam example

```python
from photonrt import CameraCaptioner

camera = CameraCaptioner(
    source=0,
    mobileclip_model="models/fp32/mobileclip-s1.onnx",
    photon_prefill_model="models/fp32/photon_prefill.onnx",
    photon_decode_model="models/fp32/photon.onnx",
    tokenizer="models/fp32/photon.tokenizer",
    workers=1,
    queue_size=1,
    frame_stride=15,
    max_new_tokens=32,
)

try:
    for result in camera:
        print(
            f"frame={result.frame_id} "
            f"inference={result.inference_ms:.1f} ms "
            f"latency={result.total_latency_ms:.1f} ms "
            f"caption={result.caption}"
        )
finally:
    camera.stop()
```

## Camera source values

Default webcam:

```python
source=0
```

Second webcam:

```python
source=1
```

Video:

```python
source="video.mp4"
```

RTSP:

```python
source="rtsp://user:password@host/stream"
```

Any source accepted by OpenCV `VideoCapture` can be used.

---

# Configuration Reference

## Camera constructor

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

| Variable | Type | Default | Description |
|---|---|---:|---|
| `source` | OpenCV source | required | Camera index, file path, URL, or other OpenCV source |
| `mobileclip_model` | `str` / `Path` | required | MobileCLIP model |
| `photon_prefill_model` | `str` / `Path` | required | Photon prefill model |
| `photon_decode_model` | `str` / `Path` | required | Photon decoder |
| `tokenizer` | `str` / `Path` | required | Photon tokenizer |
| `workers` | `int` | `2` | Number of independent inference workers |
| `queue_size` | `int` | `8` | Maximum pending inference tasks |
| `frame_stride` | `int` | `1` | Process every Nth captured frame |
| `max_new_tokens` | `int` | `32` | Maximum caption token count |
| `drop_oldest` | `bool` | `True` | Replace older queued work when the task queue is full |
| `capture_api` | `int` / `None` | `None` | Optional OpenCV capture backend |

Validation:

```text
workers > 0
queue_size > 0
frame_stride > 0
max_new_tokens > 0
```

---

# Frame Stride

`frame_stride` controls temporal sampling.

```python
frame_stride=1
```

processes every captured frame.

```python
frame_stride=5
```

processes approximately every fifth frame.

```python
frame_stride=10
```

processes approximately every tenth frame.

```python
frame_stride=15
```

processes approximately every fifteenth frame.

For a nominal 30 FPS source:

| Stride | Approx. submitted rate |
|---:|---:|
| `1` | 30 FPS |
| `5` | 6 FPS |
| `10` | 3 FPS |
| `15` | 2 FPS |
| `30` | 1 FPS |

Actual rates depend on capture and inference performance.

---

# Workers

Each worker creates its own native PhotonRT runtime.

Examples:

```python
workers=1
workers=2
workers=4
```

More workers require more model/runtime memory.

More workers do not guarantee lower latency. Benchmark `workers=1` first.

---

# Queue Size

The inference queue is bounded.

```python
queue_size=1
```

keeps little pending work.

```python
queue_size=8
```

allows more frames to wait.

A large queue can increase the age of the frame being captioned.

For real-time applications, start with:

```python
queue_size=1
```

or:

```python
queue_size=2
```

---

# Dropping Policy

## `drop_oldest=True`

When the task queue is full:

```text
old queued frame -> dropped
new frame        -> inserted
```

This favors freshness.

## `drop_oldest=False`

When the task queue is full, the incoming task can be discarded instead.

For live streams, `drop_oldest=True` is generally the appropriate freshness-oriented setting.

---

# CaptionResult

```python
CaptionResult(
    frame_id,
    capture_time_ns,
    caption,
    inference_ms,
    total_latency_ms,
)
```

| Field | Type | Description |
|---|---|---|
| `frame_id` | `int` | Sequential capture-frame identifier |
| `capture_time_ns` | `int` | Monotonic capture timestamp |
| `caption` | `str` | Generated caption |
| `inference_ms` | `float` | Worker inference time |
| `total_latency_ms` | `float` | Capture-to-result latency |

Example:

```python
for result in camera:
    print(result.frame_id)
    print(result.caption)
    print(result.inference_ms)
    print(result.total_latency_ms)
```

---

# Camera Methods

## `start()`

```python
camera.start()
```

Starts capture and worker threads.

## `stop()`

```python
camera.stop()
```

Stops capture and joins the worker threads.

## `get_result()`

```python
result = camera.get_result(timeout=5.0)
```

`timeout` is specified in seconds.

## Iteration

```python
for result in camera:
    print(result.caption)
```

Iteration automatically starts the camera pipeline.

## Context manager

```python
with CameraCaptioner(
    0,
    MOBILECLIP,
    PREFILL,
    DECODE,
    TOKENIZER,
    workers=1,
    queue_size=1,
    frame_stride=15,
) as camera:
    for result in camera:
        print(result.caption)
```

---

# Camera Statistics

```python
stats = camera.stats()
```

Returns:

```python
{
    "captured_frames": ...,
    "submitted_frames": ...,
    "completed_frames": ...,
    "dropped_frames": ...,
    "queued_frames": ...,
}
```

| Key | Meaning |
|---|---|
| `captured_frames` | Frames read from the source |
| `submitted_frames` | Frames submitted to the inference queue |
| `completed_frames` | Results consumed by the application |
| `dropped_frames` | Frames discarded because of queue pressure |
| `queued_frames` | Currently pending tasks |

---

# Recommended Streaming Configurations

### Low latency

```python
workers=1
queue_size=1
frame_stride=15
max_new_tokens=32
drop_oldest=True
```

### Balanced

```python
workers=1
queue_size=2
frame_stride=10
max_new_tokens=32
drop_oldest=True
```

### Higher temporal sampling

```python
workers=1
queue_size=2
frame_stride=5
max_new_tokens=32
drop_oldest=True
```

The best configuration is hardware-dependent.

---

# Latency

Two timing values are exposed:

```text
inference_ms
total_latency_ms
```

`inference_ms` measures the worker's inference section.

`total_latency_ms` represents the elapsed time from frame capture to result production, including queue delay.

Therefore, a frame that waits in a queue can have much higher total latency than its model inference time.

---

# Architecture

PhotonRT's image pipeline is:

```text
Input image
    |
    v
Native preprocessing
    |
    v
MobileCLIP-S1
    |
    v
512-D normalized visual embedding
    |
    v
Photon prefix conditioning
    |
    v
Photon prefill
    |
    v
Autoregressive decode
    |
    v
Tokenizer decode
    |
    v
Caption
```

The model architecture is based on a frozen MobileCLIP-S1 encoder followed by a prefix projection module and a compact decoder-only Transformer.

See:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/mobileclip.md`](docs/mobileclip.md)
- [`docs/model-format.md`](docs/model-format.md)

---

# Building from Source

PhotonRT uses CMake for the C++ runtime and scikit-build-core/pybind11 for Python packaging.

Required development components:

- C++ compiler/toolchain
- CMake
- Python
- pybind11
- ONNX Runtime development package

## ONNX Runtime

Configure:

```text
PHOTON_ONNXRUNTIME_ROOT
```

to the ONNX Runtime SDK root.

Example:

```bash
export PHOTON_ONNXRUNTIME_ROOT="C:/path/to/onnxruntime"
```

The SDK must provide the required headers and library.

## CMake options

| Variable | Typical value | Purpose |
|---|---|---|
| `PHOTON_ENABLE_ONNXRUNTIME` | `ON` | Enable ONNX Runtime |
| `PHOTON_BUILD_PYTHON` | `ON` | Build Python extension |
| `PHOTON_BUILD_EXAMPLES` | `ON` / `OFF` | Build C++ examples |
| `PHOTON_BUILD_TESTS` | `ON` / `OFF` | Build tests |
| `PHOTON_BUILD_BENCHMARKS` | `ON` / `OFF` | Build benchmarks |
| `PHOTON_ONNXRUNTIME_ROOT` | path | ONNX Runtime SDK root |

Example:

```bash
cmake -S . -B build \
  -DPHOTON_ENABLE_ONNXRUNTIME=ON \
  -DPHOTON_BUILD_PYTHON=ON \
  -DPHOTON_BUILD_EXAMPLES=OFF \
  -DPHOTON_BUILD_TESTS=OFF \
  -DPHOTON_BUILD_BENCHMARKS=OFF \
  -DPHOTON_ONNXRUNTIME_ROOT="$PHOTON_ONNXRUNTIME_ROOT"
```

Build:

```bash
cmake --build build --config Release
```

---

# Packaging

Build distributions:

```bash
python -m build
```

Validate:

```bash
python -m twine check dist/*
```

Typical artifacts:

```text
photonrt-<version>.tar.gz
photonrt-<version>-cp*-cp*-win_amd64.whl
```

The wheel contains the native Python extension and the packaged ONNX Runtime DLL.

Large model artifacts are distributed separately through Hugging Face.

---

# Repository Layout

```text
PhotonRT/
├── CMakeLists.txt
├── CMakePresets.json
├── LICENSE
├── README.md
├── pyproject.toml
│
├── backends/
│   ├── cpu/
│   └── cuda/
│
├── benchmarks/
├── bindings/
│   └── python_module.cpp
│
├── include/
│   └── photon/
│
├── src/
│
├── vision/
│   ├── mobileclip/
│   └── photon/
│
├── python/
│   └── photonrt/
│       ├── __init__.py
│       ├── hub.py
│       └── stream.py
│
├── examples/
├── tests/
├── tools/
└── docs/
    ├── architecture.md
    ├── mobileclip.md
    ├── model-format.md
    ├── python-api.md
    ├── release.md
    ├── ROADMAP.md
    └── streaming.md
```

---

# Troubleshooting

## Import error

```bash
python -m pip show photonrt
```

Then:

```bash
python -c "import photonrt; print(photonrt.__file__)"
```

## Source tree imported instead of installed package

If `PYTHONPATH` points to the repository:

```bash
unset PYTHONPATH
```

Then check:

```bash
python -c "import photonrt; print(photonrt.__file__)"
```

## Hugging Face / SSL issue

Check:

```bash
echo "$SSL_CERT_FILE"
echo "$CURL_CA_BUNDLE"
```

Unexpected certificate overrides may cause HTTPS downloads to fail.

## ONNX Runtime build failure

Check:

```text
PHOTON_ONNXRUNTIME_ROOT
```

and verify that the SDK includes the required `include/` and `lib/onnxruntime.lib` files.

## Camera cannot open

```python
import cv2

cap = cv2.VideoCapture(0)
print(cap.isOpened())
cap.release()
```

For RTSP streams, verify the URL and the OpenCV backend available on the system.

---

# Development

Native tests are under:

```text
tests/
```

Examples are under:

```text
examples/
```

Utilities are under:

```text
tools/
```

Benchmarking and model-export workflows are documented separately.

---

# Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/model-format.md`](docs/model-format.md)
- [`docs/mobileclip.md`](docs/mobileclip.md)
- [`docs/python-api.md`](docs/python-api.md)
- [`docs/streaming.md`](docs/streaming.md)
- [`docs/build.md`](docs/build.md)
- [`docs/release.md`](docs/release.md)
- [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

# Current Release

```text
PhotonRT 0.2.0
```

PyPI:

https://pypi.org/project/photonrt/

Hugging Face:

https://huggingface.co/ganapathi1578/PhotonRT

GitHub:

https://github.com/ganapathi1578/PhotonRT

---

# License

PhotonRT is released under the MIT License.

See [`LICENSE`](LICENSE).

---

# Citation

If you use PhotonRT or the Photon model in research, cite the associated work.

```bibtex
@article{photon,
  title   = {Photon: Efficient Prefix-Conditioned Image Captioning with Lightweight Transformer Decoding},
  journal = {Machine Learning},
  year    = {2026}
}
```
