# PhotonRT Roadmap

This roadmap is intentionally lightweight and can be updated as runtime capabilities mature.

## 0.2.x

- stabilize Python package APIs
- document model artifacts
- improve camera-stream configuration and examples
- expand release/test coverage
- improve error reporting
- clean source-distribution contents

## Native runtime direction

Move more orchestration into the C++ core:

```text
Current
Python capture
    +
Python queues/workers
    +
C++ inference

Target
C++ capture
    +
C++ scheduling
    +
C++ queues
    +
C++ timestamps
    +
C++ model/cache management
    +
C++ inference
```

The Python layer can then remain a thin interface over a stable native API.

## Packaging

Planned improvements:

- automated wheel builds
- multiple supported CPython versions
- broader operating-system support
- CI validation on clean environments
- automated PyPI releases
- reproducible model revisions

## Streaming

Potential improvements:

- native capture backends
- better source abstraction
- configurable frame sampling policies
- explicit backpressure modes
- zero-copy or reduced-copy frame paths
- stream lifecycle/error APIs
- telemetry and profiling

## Performance

Potential areas:

- preprocessing optimization
- ONNX Runtime execution tuning
- memory reuse
- decoder optimization
- backend-specific acceleration
- worker scheduling
- reduced host/device copies where supported

## Model formats

Potential future runtime formats:

```text
FP32
FP16
INT8
```

Any additional format should preserve a clearly documented model contract and explicit compatibility requirements.

## API stability

The public API should prioritize a small stable surface:

```python
Captioner
Captioner.from_pretrained
Captioner.caption

CaptionerNative
CaptionOptions

CameraCaptioner
CaptionResult
```

New features should avoid unnecessary breaking changes.
