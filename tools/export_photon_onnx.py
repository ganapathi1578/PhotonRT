#!/usr/bin/env python3
"""
Export the exact Photon / MobileCapModern checkpoint to ONNX with KV cache.

IMPORTANT:
This architecture intentionally follows mobilecap_ablation.py exactly:

    512 -> 2048 -> GELU -> 2048
    8 visual prefix tokens
    256 hidden dimension
    8 attention heads
    6 Transformer blocks
    SwiGLU FFN, hidden=768
    vocabulary=8000

The ONNX implementation only changes the execution strategy to:

    photon_prefill.onnx
        image embedding + BOS
        -> logits
        -> KV cache

    photon.onnx
        next token + KV cache
        -> logits
        -> updated KV cache

The learned weights and model mathematics are not changed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EXACT PHOTON CONFIG
# ============================================================

D_MODEL = 256
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_LAYERS = 6

FFN_HIDDEN = 768

VOCAB_SIZE = 8000

IMG_EMB_DIM = 512
NUM_IMG_TOKENS = 8

MAX_TEXT_LENGTH = 32
MAX_TOTAL_LENGTH = MAX_TEXT_LENGTH + NUM_IMG_TOKENS


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        var = x.pow(2).mean(
            -1,
            keepdim=True,
        )

        x_norm = (
            x
            * torch.rsqrt(
                var + self.eps
            )
        )

        return self.weight * x_norm


# ============================================================
# RoPE
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_seq_len: int = MAX_TOTAL_LENGTH,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                ).float()
                / dim
            )
        )

        positions = torch.arange(
            max_seq_len
        ).float()

        freqs = torch.einsum(
            "i,j->ij",
            positions,
            inv_freq,
        )

        emb = torch.cat(
            (
                freqs,
                freqs,
            ),
            dim=-1,
        )

        self.register_buffer(
            "cos_cached",
            emb.cos()[
                None,
                None,
                :,
                :
            ],
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            emb.sin()[
                None,
                None,
                :,
                :
            ],
            persistent=False,
        )

    @staticmethod
    def rotate_half(x):

        x1, x2 = x.chunk(
            2,
            dim=-1,
        )

        return torch.cat(
            (
                -x2,
                x1,
            ),
            dim=-1,
        )

    def apply(
        self,
        q,
        k,
        position_ids,
    ):
        """
        q/k:
            [B,H,T,D]

        position_ids:
            [B,T]
        """

        batch, steps = position_ids.shape

        flat_positions = (
            position_ids.reshape(-1)
        )

        cos = self.cos_cached[
            0,
            0,
        ].index_select(
            0,
            flat_positions,
        )

        sin = self.sin_cached[
            0,
            0,
        ].index_select(
            0,
            flat_positions,
        )

        cos = cos.view(
            batch,
            steps,
            -1,
        ).unsqueeze(1)

        sin = sin.view(
            batch,
            steps,
            -1,
        ).unsqueeze(1)

        q = (
            q * cos
            + self.rotate_half(q) * sin
        )

        k = (
            k * cos
            + self.rotate_half(k) * sin
        )

        return q, k


# ============================================================
# SwiGLU
# ============================================================

class SwiGLUFFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.w1 = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.w2 = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        self.w3 = nn.Linear(
            hidden_dim,
            d_model,
            bias=False,
        )

    def forward(self, x):

        return self.w3(
            F.silu(
                self.w1(x)
            )
            * self.w2(x)
        )


# ============================================================
# EXACT SELF-ATTENTION
# ============================================================

class ModernAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.n_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_len,
        )

    def project_qkv(
        self,
        x,
        position_ids,
    ):

        batch, steps, channels = (
            x.shape
        )

        q = self.q_proj(
            x
        ).view(
            batch,
            steps,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = self.k_proj(
            x
        ).view(
            batch,
            steps,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = self.v_proj(
            x
        ).view(
            batch,
            steps,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rope.apply(
            q,
            k,
            position_ids,
        )

        return q, k, v

    def forward(
        self,
        x,
        mask=None,
    ):

        batch, steps, channels = (
            x.shape
        )

        position_ids = torch.arange(
            steps,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(
            batch,
            steps,
        )

        q, k, v = self.project_qkv(
            x,
            position_ids,
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
        )

        out = (
            out
            .transpose(1, 2)
            .contiguous()
            .view(
                batch,
                steps,
                channels,
            )
        )

        return self.o_proj(out)


# ============================================================
# EXACT NANO LLAMA BLOCK
# ============================================================

class NanoLlamaBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.attn_norm = RMSNorm(
            D_MODEL
        )

        self.attn = ModernAttention(
            D_MODEL,
            NUM_HEADS,
            MAX_TOTAL_LENGTH,
        )

        self.ffn_norm = RMSNorm(
            D_MODEL
        )

        self.ffn = SwiGLUFFN(
            D_MODEL,
            FFN_HIDDEN,
        )

    def forward(
        self,
        x,
        mask=None,
    ):

        h = x + self.attn(
            self.attn_norm(x),
            mask,
        )

        out = h + self.ffn(
            self.ffn_norm(h)
        )

        return out


# ============================================================
# EXACT PHOTON MODEL
# ============================================================

class PhotonModel(nn.Module):

    def __init__(self):

        super().__init__()

        # EXACT:
        # 512 -> 2048 -> GELU -> 2048
        # then reshape to [B,8,256]
        self.proj = nn.Sequential(
            nn.Linear(
                IMG_EMB_DIM,
                D_MODEL * NUM_IMG_TOKENS,
            ),
            nn.GELU(),
            nn.Linear(
                D_MODEL * NUM_IMG_TOKENS,
                D_MODEL * NUM_IMG_TOKENS,
            ),
        )

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL,
        )

        self.layers = nn.ModuleList(
            [
                NanoLlamaBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.norm = RMSNorm(
            D_MODEL
        )

        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=False,
        )

        # EXACT WEIGHT TYING
        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def load_checkpoint(
        self,
        checkpoint_path,
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        state = checkpoint

        if isinstance(
            checkpoint,
            dict,
        ):

            for key in (
                "state_dict",
                "model_state_dict",
                "model",
            ):

                candidate = checkpoint.get(
                    key
                )

                if isinstance(
                    candidate,
                    dict,
                ):

                    state = candidate
                    break

        if not isinstance(
            state,
            dict,
        ):
            raise RuntimeError(
                "Checkpoint does not contain a state_dict"
            )

        cleaned = {}

        for key, value in state.items():

            if not isinstance(
                value,
                torch.Tensor,
            ):
                continue

            # Common wrappers
            while key.startswith(
                "module."
            ):

                key = key[
                    len("module.") :
                ]

            if key.startswith(
                "model."
            ):

                key = key[
                    len("model.") :
                ]

            cleaned[key] = value

        missing, unexpected = (
            self.load_state_dict(
                cleaned,
                strict=False,
            )
        )

        if missing or unexpected:

            print()
            print(
                "Checkpoint mismatch"
            )
            print(
                "=================="
            )

            if missing:

                print(
                    "Missing keys:"
                )

                for key in missing:
                    print(
                        "  ",
                        key,
                    )

            if unexpected:

                print(
                    "Unexpected keys:"
                )

                for key in unexpected:
                    print(
                        "  ",
                        key,
                    )

            raise RuntimeError(
                "Checkpoint does not match exact Photon architecture"
            )

        self.eval()

        return len(cleaned)

    def build_prefix(
        self,
        image_embedding,
        input_ids,
    ):

        batch = input_ids.shape[0]

        image_tokens = self.proj(
            image_embedding
        ).view(
            batch,
            NUM_IMG_TOKENS,
            D_MODEL,
        )

        text_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        x = torch.cat(
            [
                image_tokens,
                text_embeddings,
            ],
            dim=1,
        )

        return x

    def causal_prefix_mask(
        self,
        total_length,
        device,
    ):

        mask = torch.ones(
            (
                total_length,
                total_length,
            ),
            device=device,
            dtype=torch.bool,
        ).tril()

        # Every token can attend to
        # every visual prefix token.
        mask[
            :,
            :NUM_IMG_TOKENS,
        ] = True

        return mask

    def forward(
        self,
        image_embedding,
        input_ids,
    ):

        x = self.build_prefix(
            image_embedding,
            input_ids,
        )

        mask = self.causal_prefix_mask(
            x.shape[1],
            x.device,
        )

        mask = mask.view(
            1,
            1,
            x.shape[1],
            x.shape[1],
        )

        for layer in self.layers:

            x = layer(
                x,
                mask=mask,
            )

        x = self.norm(x)

        text_output = x[
            :,
            NUM_IMG_TOKENS:,
            :,
        ]

        return self.lm_head(
            text_output
        )


# ============================================================
# PREFILL GRAPH
# ============================================================

class PhotonPrefill(nn.Module):

    """
    Image + first token.

    Input:
        image_embedding [B,512]
        input_ids       [B,1]

    Sequence:
        [IMG x 8] + [BOS]

    Output:
        logits [B,1,8000]

        K/V cache:
            [B,8,9,32]
    """

    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        image_embedding,
        input_ids,
    ):

        x = self.model.build_prefix(
            image_embedding,
            input_ids,
        )

        batch, total, _ = (
            x.shape
        )

        position_ids = torch.arange(
            total,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(
            batch,
            total,
        )

        mask = self.model.causal_prefix_mask(
            total,
            x.device,
        )

        mask = mask.view(
            1,
            1,
            total,
            total,
        )

        presents = []

        for layer in self.model.layers:

            h = layer.attn_norm(x)

            q, k, v = (
                layer.attn.project_qkv(
                    h,
                    position_ids,
                )
            )

            scores = torch.matmul(
                q,
                k.transpose(
                    -2,
                    -1,
                ),
            )

            scores = (
                scores
                / math.sqrt(
                    HEAD_DIM
                )
            )

            scores = scores.masked_fill(
                ~mask,
                float("-inf"),
            )

            probabilities = torch.softmax(
                scores,
                dim=-1,
            )

            attention_output = (
                torch.matmul(
                    probabilities,
                    v,
                )
            )

            merged = (
                attention_output
                .transpose(1, 2)
                .contiguous()
                .view(
                    batch,
                    total,
                    D_MODEL,
                )
            )

            x = (
                x
                + layer.attn.o_proj(
                    merged
                )
            )

            x = (
                x
                + layer.ffn(
                    layer.ffn_norm(x)
                )
            )

            # Store ROTATED K and V.
            presents.append(k)
            presents.append(v)

        x = self.model.norm(x)

        logits = self.model.lm_head(
            x[
                :,
                NUM_IMG_TOKENS:,
                :,
            ]
        )

        return (
            logits,
            *presents,
        )


# ============================================================
# DECODE GRAPH
# ============================================================

class PhotonDecode(nn.Module):

    """
    One-token autoregressive decode.

    Input:
        input_ids    [B,1]
        position_ids [B,1]

        past_key_i
        past_value_i

    Output:
        logits [B,1,8000]

        present_key_i
        present_value_i
    """

    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        input_ids,
        position_ids,
        *past,
    ):

        x = self.model.token_embedding(
            input_ids
        )

        batch = input_ids.shape[0]

        presents = []

        for layer_idx, layer in enumerate(
            self.model.layers
        ):

            past_k = past[
                2 * layer_idx
            ]

            past_v = past[
                2 * layer_idx + 1
            ]

            h = layer.attn_norm(x)

            q, k, v = (
                layer.attn.project_qkv(
                    h,
                    position_ids,
                )
            )

            # Cache already contains all
            # previous rotated keys.
            k_all = torch.cat(
                [
                    past_k,
                    k,
                ],
                dim=2,
            )

            v_all = torch.cat(
                [
                    past_v,
                    v,
                ],
                dim=2,
            )

            scores = torch.matmul(
                q,
                k_all.transpose(
                    -2,
                    -1,
                ),
            )

            scores = (
                scores
                / math.sqrt(
                    HEAD_DIM
                )
            )

            probabilities = torch.softmax(
                scores,
                dim=-1,
            )

            attention_output = (
                torch.matmul(
                    probabilities,
                    v_all,
                )
            )

            merged = (
                attention_output
                .transpose(1, 2)
                .contiguous()
                .view(
                    batch,
                    1,
                    D_MODEL,
                )
            )

            x = (
                x
                + layer.attn.o_proj(
                    merged
                )
            )

            x = (
                x
                + layer.ffn(
                    layer.ffn_norm(x)
                )
            )

            presents.append(
                k_all
            )

            presents.append(
                v_all
            )

        x = self.model.norm(x)

        logits = self.model.lm_head(
            x
        )

        return (
            logits,
            *presents,
        )


# ============================================================
# EXPORT HELPERS
# ============================================================

def export_prefill(
    model,
    output_path,
    opset,
):

    wrapper = PhotonPrefill(
        model
    ).eval()

    dummy_image = torch.zeros(
        1,
        IMG_EMB_DIM,
        dtype=torch.float32,
    )

    dummy_ids = torch.zeros(
        1,
        1,
        dtype=torch.long,
    )

    input_names = [
        "image_embedding",
        "input_ids",
    ]

    output_names = [
        "logits",
    ]

    for layer in range(NUM_LAYERS):

        output_names.append(
            f"present_key_{layer}"
        )

        output_names.append(
            f"present_value_{layer}"
        )

    dynamic_axes = {
        "image_embedding": {
            0: "batch",
        },
        "input_ids": {
            0: "batch",
        },
        "logits": {
            0: "batch",
        },
    }

    for layer in range(NUM_LAYERS):

        dynamic_axes[
            f"present_key_{layer}"
        ] = {
            0: "batch",
        }

        dynamic_axes[
            f"present_value_{layer}"
        ] = {
            0: "batch",
        }

    print(
        f"Exporting prefill: {output_path}"
    )

    torch.onnx.export(
        wrapper,
        (
            dummy_image,
            dummy_ids,
        ),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
        external_data=False,
        dynamic_axes=dynamic_axes,
    )


def export_decode(
    model,
    output_path,
    opset,
):

    wrapper = PhotonDecode(
        model
    ).eval()

    dummy_ids = torch.zeros(
        1,
        1,
        dtype=torch.long,
    )

    dummy_position = torch.tensor(
        [[NUM_IMG_TOKENS + 1]],
        dtype=torch.long,
    )

    # Prefill = 8 image tokens + BOS.
    initial_cache_length = (
        NUM_IMG_TOKENS + 1
    )

    dummy_inputs = [
        dummy_ids,
        dummy_position,
    ]

    input_names = [
        "input_ids",
        "position_ids",
    ]

    output_names = [
        "logits",
    ]

    dynamic_axes = {
        "input_ids": {
            0: "batch",
        },
        "position_ids": {
            0: "batch",
        },
        "logits": {
            0: "batch",
        },
    }

    for layer in range(NUM_LAYERS):

        dummy_inputs.append(
            torch.zeros(
                1,
                NUM_HEADS,
                initial_cache_length,
                HEAD_DIM,
                dtype=torch.float32,
            )
        )

        dummy_inputs.append(
            torch.zeros(
                1,
                NUM_HEADS,
                initial_cache_length,
                HEAD_DIM,
                dtype=torch.float32,
            )
        )

        input_names.append(
            f"past_key_{layer}"
        )

        input_names.append(
            f"past_value_{layer}"
        )

        output_names.append(
            f"present_key_{layer}"
        )

        output_names.append(
            f"present_value_{layer}"
        )

        dynamic_axes[
            f"past_key_{layer}"
        ] = {
            0: "batch",
            2: "past_sequence",
        }

        dynamic_axes[
            f"past_value_{layer}"
        ] = {
            0: "batch",
            2: "past_sequence",
        }

        dynamic_axes[
            f"present_key_{layer}"
        ] = {
            0: "batch",
            2: "present_sequence",
        }

        dynamic_axes[
            f"present_value_{layer}"
        ] = {
            0: "batch",
            2: "present_sequence",
        }

    print(
        f"Exporting decode: {output_path}"
    )

    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
        external_data=False,
        dynamic_axes=dynamic_axes,
    )


# ============================================================
# VALIDATE ONNX
# ============================================================

def validate_onnx(
    path,
):

    import onnx

    model = onnx.load(
        str(path)
    )

    onnx.checker.check_model(
        model
    )

    print(
        f"onnx check : passed ({path})"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Export exact Photon checkpoint "
            "to ONNX with KV cache."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--prefill-output",
        default="photon_prefill.onnx",
    )

    parser.add_argument(
        "--decode-output",
        default="photon.onnx",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=17,
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model = PhotonModel()

    tensors = model.load_checkpoint(
        args.checkpoint
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Photon ONNX exporter"
    )
    print(
        "===================="
    )
    print(
        f"checkpoint tensors : {tensors}"
    )
    print(
        f"total parameters   : {total_params:,}"
    )
    print(
        f"trainable params   : {trainable_params:,}"
    )
    print(
        f"d_model            : {D_MODEL}"
    )
    print(
        f"heads               : {NUM_HEADS}"
    )
    print(
        f"head_dim            : {HEAD_DIM}"
    )
    print(
        f"layers              : {NUM_LAYERS}"
    )
    print(
        f"FFN hidden          : {FFN_HIDDEN}"
    )
    print(
        f"vocabulary          : {VOCAB_SIZE}"
    )
    print(
        f"image embedding     : {IMG_EMB_DIM}"
    )
    print(
        f"image tokens        : {NUM_IMG_TOKENS}"
    )
    print(
        f"max text length     : {MAX_TEXT_LENGTH}"
    )
    print(
        f"max total length    : {MAX_TOTAL_LENGTH}"
    )
    print()

    prefill_path = Path(
        args.prefill_output
    )

    decode_path = Path(
        args.decode_output
    )

    export_prefill(
        model,
        prefill_path,
        args.opset,
    )

    export_decode(
        model,
        decode_path,
        args.opset,
    )

    validate_onnx(
        prefill_path
    )

    validate_onnx(
        decode_path
    )

    # --------------------------------------------------------
    # Numerical smoke values
    # --------------------------------------------------------

    image = torch.zeros(
        1,
        IMG_EMB_DIM,
    )

    bos = torch.zeros(
        1,
        1,
        dtype=torch.long,
    )

    with torch.no_grad():

        prefill = PhotonPrefill(
            model
        ).eval()

        result = prefill(
            image,
            bos,
        )

        checksum = float(
            result[0].sum().item()
        )

    print()
    print(
        f"prefill logits checksum : "
        f"{checksum:.8e}"
    )

    print(
        "export complete"
    )


if __name__ == "__main__":
    main()