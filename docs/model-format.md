# PhotonRT Model Format

## Runtime artifacts

The current runtime expects:

```text
mobileclip-s1.onnx
photon_prefill.onnx
photon.onnx
photon.tokenizer
```

An optional:

```text
config.json
```

may also be present in the model repository.

## Model roles

| Artifact | Role |
|---|---|
| `mobileclip-s1.onnx` | Vision encoder |
| `photon_prefill.onnx` | Prefill / initial decoder state |
| `photon.onnx` | Autoregressive decoder |
| `photon.tokenizer` | Tokenization and special token information |
| `config.json` | Optional metadata |

## MobileCLIP contract

Conceptually:

```text
image
  ->
MobileCLIP-S1
  ->
512-D visual representation
  ->
L2 normalization
```

The resulting embedding is passed into Photon.

## Photon prefix contract

The Photon architecture expands the 512-dimensional visual representation into eight prefix tokens.

```text
512-D image embedding
        |
        v
projection module
        |
        v
8 visual prefix tokens
```

## Prefill

The prefill stage initializes the decoder state for the multimodal context.

Conceptually:

```text
image embedding + BOS
        |
        v
Photon prefill
        |
        +--> logits
        +--> decoder state / KV tensors
```

## Decode

The decode stage executes autoregressively.

Conceptually:

```text
previous token + decoder state
        |
        v
Photon decode
        |
        v
next-token logits
```

The runtime repeats this process until EOS or `max_new_tokens` is reached.

## Tokenizer

The tokenizer file provides the tokenization data and special-token IDs required by the native runtime.

The runtime needs BOS and EOS handling for generation.

## Hugging Face repository

The public model repository is:

```text
https://huggingface.co/ganapathi1578/PhotonRT
```

The high-level Python loader retrieves only the runtime artifacts it needs.

## Local development layout

The source tree may keep local development models under:

```text
models/
└── fp32/
    ├── mobileclip-s1.onnx
    ├── photon_prefill.onnx
    ├── photon.onnx
    └── photon.tokenizer
```

Model binaries do not need to be bundled into the source repository for normal Python package use when Hugging Face distribution is used.

## Reproducibility

For reproducible model deployment, specify an immutable Hugging Face revision:

```python
model = Captioner.from_pretrained(
    repo_id="ganapathi1578/PhotonRT",
    revision="<immutable-revision>",
)
```

Avoid relying on a mutable `main` branch when exact model reproducibility matters.
