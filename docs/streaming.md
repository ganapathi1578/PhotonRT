# PhotonRT Streaming Guide

## Overview

`CameraCaptioner` provides camera, video-file, and network-stream captioning.

The current implementation uses OpenCV for capture and native PhotonRT for inference.

```text
Capture
  |
  v
frame_stride
  |
  v
bounded queue
  |
  +---- worker 0
  +---- worker 1
  +---- worker N
  |
  v
native PhotonRT
  |
  v
result queue
  |
  v
CaptionResult
```

## Installation

```bash
pip install photonrt opencv-python
```

## Basic webcam

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
    drop_oldest=True,
)

try:
    for result in camera:
        print(
            f"[{result.frame_id}] "
            f"{result.total_latency_ms:.1f} ms "
            f"{result.caption}"
        )
finally:
    camera.stop()
```

## Sources

### Webcam

```python
source=0
```

### Video

```python
source="input.mp4"
```

### RTSP

```python
source="rtsp://user:password@host/stream"
```

## Parameters

### `workers`

Each worker owns an independent native runtime.

```python
workers=1
workers=2
workers=4
```

Use the smallest number that meets the required throughput.

### `queue_size`

Controls the maximum number of pending tasks.

```python
queue_size=1
queue_size=2
queue_size=8
```

A large queue can increase latency because old frames wait longer.

### `frame_stride`

Temporal sampling.

```text
1  -> every frame
5  -> every 5th frame
10 -> every 10th frame
15 -> every 15th frame
```

At a nominal 30 FPS input:

```text
stride 1  -> ~30 submitted frames/s
stride 5  -> ~6 submitted frames/s
stride 10 -> ~3 submitted frames/s
stride 15 -> ~2 submitted frames/s
```

### `max_new_tokens`

Controls caption length.

Typical values:

```python
16
32
48
64
```

Smaller values generally reduce generation work.

### `drop_oldest`

For live streams:

```python
drop_oldest=True
```

prioritizes current frames.

## `CaptionResult`

```python
@dataclass(frozen=True)
class CaptionResult:
    frame_id: int
    capture_time_ns: int
    caption: str
    inference_ms: float
    total_latency_ms: float
```

## `get_result`

```python
result = camera.get_result(timeout=1.0)
```

`timeout` is in seconds.

## Statistics

```python
print(camera.stats())
```

Keys:

```text
captured_frames
submitted_frames
completed_frames
dropped_frames
queued_frames
```

## Real-time tuning

Start here:

```python
workers=1
queue_size=1
frame_stride=10
max_new_tokens=32
drop_oldest=True
```

For lower frame sampling:

```python
frame_stride=15
```

For higher frame sampling:

```python
frame_stride=5
```

Do not increase queue size just to hide inference latency. A queue can trade throughput for frame age.

