# PhotonRT architecture

PhotonRT is a CPU-first native inference runtime for the Photon model described in Photon-12M.

## Current model

- MobileCLIP-S1 produces a 512-D normalized image embedding.
- The projector maps 512 -> 2048 -> 2048, then reshapes to 8 x 256 prefix tokens.
- Decoder: 6 pre-norm Transformer blocks.
- Hidden size: 256.
- Attention: 8 heads, head dimension 32.
- Position encoding: RoPE.
- FFN: SwiGLU, intermediate size 768.
- Vocabulary: 8,000.
- Generation begins from `[CLS]` (id 1) and stops on `[SEP]` (id 2).

## Attention mask

During prefill, image queries can attend to all 8 image prefix tokens. Text queries can attend to all image tokens and to previous/current text tokens causally. This matches the current Python reference mask.

During autoregressive decode, the new token attends to the complete KV cache.

## Runtime state

```text
Model weights
  -> PhotonModel
  -> prefill(image prefix + BOS)
  -> KV cache
  -> decode(next token)
  -> logits
  -> sampler
```
