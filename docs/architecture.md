# PhotonRT Architecture

## Runtime architecture

PhotonRT is split into a native inference core and Python-facing interfaces.

```text
                     Python
                       |
              pybind11 extension
                       |
                       v
             +-------------------+
             | PhotonRT C++ Core |
             +-------------------+
                |      |      |
                |      |      +--> Tokenizer
                |      |
                |      +---------> Photon ONNX
                |
                +---------------> MobileCLIP ONNX
                         |
                         v
                    ONNX Runtime
```

## Image captioning pipeline

```text
Image
  |
  v
Image loading
  |
  v
Native preprocessing
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

## Model architecture

The Photon model uses:

- frozen MobileCLIP-S1 visual encoding
- a two-layer projection module
- eight visual prefix tokens
- a compact decoder-only Transformer
- RMSNorm
- RoPE-based causal self-attention
- SwiGLU feed-forward layers
- autoregressive decoding
- tokenizer-based output decoding

## Prefix conditioning

The visual encoder produces a 512-dimensional image representation.

The projection stage maps this representation to:

```text
8 prefix tokens
```

The visual prefix is combined with the text sequence before the decoder processes the autoregressive language-model computation.

## ONNX split

PhotonRT keeps the runtime graph split into:

```text
mobileclip-s1.onnx
photon_prefill.onnx
photon.onnx
photon.tokenizer
```

### MobileCLIP

Produces the normalized visual embedding used by Photon.

### Photon prefill

Initializes the language-model decoding state using the visual representation and the beginning-of-sequence token.

### Photon decode

Performs autoregressive token generation using the decoder state.

### Tokenizer

Provides token IDs and decoding rules required by the runtime.

## Python interface

The high-level Python interface:

```python
Captioner.from_pretrained()
```

handles model retrieval and runtime construction.

The low-level native binding:

```python
CaptionerNative(...)
```

provides direct access to the C++ runtime.

## Camera architecture

Current v1 streaming architecture:

```text
OpenCV capture thread
        |
        v
frame sampling
        |
        v
bounded task queue
        |
        +--> worker 0 --> native Captioner
        +--> worker 1 --> native Captioner
        +--> worker N --> native Captioner
        |
        v
bounded result queue
        |
        v
CaptionResult
```

Each worker owns an independent native runtime.

## Latency

The streaming API exposes:

```text
inference_ms
total_latency_ms
```

`inference_ms` measures the worker's inference region.

`total_latency_ms` measures elapsed time from frame capture to result production.

Consequently:

```text
total latency
    =
queue delay
+
inference
+
result handling
```

## Design goals

PhotonRT is intended to:

1. keep inference work in native code
2. provide a simple Python entry point
3. separate model artifacts from the package binaries
4. support real-time streaming with bounded queues
5. allow direct C++ use in addition to Python
