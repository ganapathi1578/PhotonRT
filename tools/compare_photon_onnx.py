#!/usr/bin/env python3
"""
End-to-end Photon ONNX vs PhotonRT C++ comparison.

Pipeline A:
    image
      -> official MobileCLIP preprocessing
      -> MobileCLIP ONNX
      -> Photon prefill ONNX
      -> Photon decode ONNX + KV cache
      -> caption

Pipeline B:
    image
      -> PhotonRT native image loader
      -> PhotonRT MobileCLIP preprocessing
      -> MobileCLIP ONNX
      -> PhotonRT native Photon
      -> caption

Tokenizer IDs are taken automatically from mobilecap_tokenizer.json:
    [UNK] = 0
    [CLS] = 1   <-- BOS
    [SEP] = 2   <-- EOS
    [PAD] = 3
    [MASK] = 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# Photon constants
# ============================================================

IMG_EMB_DIM = 512
NUM_IMG_TOKENS = 8
NUM_LAYERS = 6
NUM_HEADS = 8
HEAD_DIM = 32
VOCAB_SIZE = 8000
MAX_TEXT_LENGTH = 32


# ============================================================
# Tokenizer
# ============================================================

def load_special_token_ids(tokenizer_json: Path):
    """
    Read special-token IDs directly from the HuggingFace
    tokenizer JSON instead of passing them through CLI args.
    """

    with open(tokenizer_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}

    for item in data.get("added_tokens", []):
        token = item.get("content")
        token_id = item.get("id")

        if token is not None and token_id is not None:
            mapping[token] = int(token_id)

    required = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    missing = [x for x in required if x not in mapping]

    if missing:
        raise RuntimeError(
            "Missing tokenizer special tokens: "
            + ", ".join(missing)
        )

    return {
        "unk": mapping["[UNK]"],
        "bos": mapping["[CLS]"],
        "eos": mapping["[SEP]"],
        "pad": mapping["[PAD]"],
        "mask": mapping["[MASK]"],
    }


# ============================================================
# Exact MobileCLIP preprocessing
# ============================================================

def preprocess_mobileclip(image_path: Path) -> np.ndarray:
    """
    Use MobileCLIP's own official transform.

    This avoids accidentally introducing:
      - 224x224 input
      - ImageNet normalization
      - CLIP mean/std normalization
      - wrong resize behavior
    """

    try:
        import mobileclip
    except ImportError as exc:
        raise RuntimeError(
            "The 'mobileclip' Python package is required for the "
            "end-to-end parity test.\n"
            "Install/use the same environment used to export "
            "mobileclip-s1.onnx."
        ) from exc

    _, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s1",
        pretrained=None,
        device="cpu",
    )

    image = Image.open(image_path).convert("RGB")

    x = preprocess(image).float().contiguous()

    if tuple(x.shape) != (3, 256, 256):
        raise RuntimeError(
            f"Unexpected MobileCLIP tensor shape: {tuple(x.shape)}"
        )

    x = x.numpy()

    # NCHW
    x = np.expand_dims(x, axis=0)

    return np.ascontiguousarray(x, dtype=np.float32)


# ============================================================
# MobileCLIP ONNX
# ============================================================

def create_onnx_session(path: Path):
    return ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )


def run_mobileclip_onnx(
    session: ort.InferenceSession,
    image_tensor: np.ndarray,
) -> np.ndarray:

    inputs = session.get_inputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected one MobileCLIP input, got {len(inputs)}"
        )

    input_meta = inputs[0]

    print("\nMobileCLIP ONNX")
    print("==============")
    print("input name :", input_meta.name)
    print("input shape:", input_meta.shape)
    print("providers  :", session.get_providers())

    output = session.run(
        None,
        {
            input_meta.name: image_tensor,
        },
    )[0]

    embedding = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1)

    if embedding.size != IMG_EMB_DIM:
        raise RuntimeError(
            f"Expected 512-D MobileCLIP embedding, "
            f"got {embedding.size}"
        )

    return embedding


# ============================================================
# Utility comparison
# ============================================================

def compare_arrays(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.shape != b.shape:
        print(
            f"{name}: SHAPE MISMATCH "
            f"{a.shape} vs {b.shape}"
        )
        return False

    diff = np.abs(a - b)

    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na > 0.0 and nb > 0.0:
        cosine = float(np.dot(a, b) / (na * nb))
    else:
        cosine = 1.0

    print(
        f"{name:24s} "
        f"max_abs={max_abs:.6e} "
        f"mean_abs={mean_abs:.6e} "
        f"cosine={cosine:.10f}"
    )

    return True


# ============================================================
# Photon ONNX prefill
# ============================================================

def run_prefill(
    session: ort.InferenceSession,
    embedding: np.ndarray,
    bos_id: int,
):
    image_embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(1, IMG_EMB_DIM)

    input_ids = np.array(
        [[bos_id]],
        dtype=np.int64,
    )

    outputs = session.run(
        None,
        {
            "image_embedding": image_embedding,
            "input_ids": input_ids,
        },
    )

    logits = outputs[0]

    cache = {}

    for layer in range(NUM_LAYERS):
        cache[f"past_key_{layer}"] = outputs[
            1 + 2 * layer
        ]

        cache[f"past_value_{layer}"] = outputs[
            2 + 2 * layer
        ]

    return logits, cache


# ============================================================
# Photon ONNX decode
# ============================================================

def run_decode(
    session: ort.InferenceSession,
    input_id: int,
    position: int,
    cache: dict,
):
    input_ids = np.array(
        [[input_id]],
        dtype=np.int64,
    )

    position_ids = np.array(
        [[position]],
        dtype=np.int64,
    )

    feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }

    feed.update(cache)

    outputs = session.run(
        None,
        feed,
    )

    logits = outputs[0]

    new_cache = {}

    for layer in range(NUM_LAYERS):
        new_cache[f"past_key_{layer}"] = outputs[
            1 + 2 * layer
        ]

        new_cache[f"past_value_{layer}"] = outputs[
            2 + 2 * layer
        ]

    return logits, new_cache


# ============================================================
# Generate with Photon ONNX KV cache
# ============================================================

def generate_onnx(
    prefill_session: ort.InferenceSession,
    decode_session: ort.InferenceSession,
    embedding: np.ndarray,
    bos_id: int,
    eos_id: int,
):
    """
    Exact captioning flow:

        prefill:
            [image tokens] + [CLS]

        decode:
            token 1
            token 2
            ...
            [SEP]
    """

    prefill_logits, cache = run_prefill(
        prefill_session,
        embedding,
        bos_id,
    )

    # Prefill's final position is [CLS].
    next_token = int(
        np.argmax(
            prefill_logits[0, -1]
        )
    )

    generated = []

    # Image tokens occupy positions 0..7.
    # [CLS] occupies position 8.
    #
    # Therefore the first generated token is position 9.
    position = NUM_IMG_TOKENS + 1

    for _ in range(MAX_TEXT_LENGTH):

        if next_token == eos_id:
            break

        generated.append(next_token)

        logits, cache = run_decode(
            decode_session,
            next_token,
            position,
            cache,
        )

        next_token = int(
            np.argmax(
                logits[0, -1]
            )
        )

        position += 1

    return generated


# ============================================================
# Decode tokenizer JSON
# ============================================================

def decode_token_ids(
    tokenizer_json: Path,
    token_ids: list[int],
):
    """
    Decode generated IDs using the exact tokenizer JSON.
    """

    try:
        from tokenizers import Tokenizer
    except ImportError:
        return " ".join(
            str(x)
            for x in token_ids
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_json)
    )

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


# ============================================================
# Run C++ PhotonRT
# ============================================================

def run_cpp(
    cpp_exe: Path,
    mobileclip: Path,
    photon: Path,
    cpp_tokenizer: Path,
    image: Path,
):
    """
    The native executable MUST receive photon.tokenizer,
    NOT mobilecap_tokenizer.json.
    """

    cmd = [
        str(cpp_exe),
        str(mobileclip),
        str(photon),
        str(cpp_tokenizer),
        str(image),
    ]

    print("\nC++ PhotonRT")
    print("============")
    print("command:")

    print(
        " ".join(
            f'"{x}"'
            for x in cmd
        )
    )

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.stdout:
        print("\nstdout:")
        print(result.stdout.rstrip())

    if result.stderr:
        print("\nstderr:")
        print(result.stderr.rstrip())

    return result.returncode, result.stdout.strip()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Photon ONNX KV-cache generation "
            "against the native PhotonRT C++ pipeline."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--mobileclip",
        default="mobileclip-s1.onnx",
    )

    parser.add_argument(
        "--prefill",
        default="photon_prefill.onnx",
    )

    parser.add_argument(
        "--decode",
        default="photon.onnx",
    )

    # Python tokenizer JSON.
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="mobilecap_tokenizer.json",
    )

    # Native tokenizer binary.
    parser.add_argument(
        "--cpp-tokenizer",
        default="photon.tokenizer",
        help="Native Photon tokenizer file",
    )

    parser.add_argument(
        "--cpp",
        default="build/Release/image-caption.exe",
    )

    parser.add_argument(
        "--cpp-photon",
        default="photon.f32.photon",
    )

    args = parser.parse_args()

    image = Path(args.image)
    mobileclip_path = Path(args.mobileclip)
    prefill_path = Path(args.prefill)
    decode_path = Path(args.decode)
    tokenizer_json = Path(args.tokenizer)
    cpp_tokenizer = Path(args.cpp_tokenizer)
    cpp_exe = Path(args.cpp)
    cpp_photon = Path(args.cpp_photon)

    # --------------------------------------------------------
    # File validation
    # --------------------------------------------------------

    required = [
        image,
        mobileclip_path,
        prefill_path,
        decode_path,
        tokenizer_json,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    print("\nPhoton end-to-end comparison")
    print("============================")

    print("image       :", image)
    print("mobileclip  :", mobileclip_path)
    print("prefill     :", prefill_path)
    print("decode      :", decode_path)
    print("tokenizer   :", tokenizer_json)
    print("cpp tokenizer:", cpp_tokenizer)
    print("cpp photon   :", cpp_photon)

    # --------------------------------------------------------
    # Tokenizer IDs
    # --------------------------------------------------------

    special = load_special_token_ids(
        tokenizer_json
    )

    bos_id = special["bos"]
    eos_id = special["eos"]

    print("\nTokenizer")
    print("=========")
    print("[UNK] :", special["unk"])
    print("[CLS] :", bos_id)
    print("[SEP] :", eos_id)
    print("[PAD] :", special["pad"])
    print("[MASK]:", special["mask"])

    # --------------------------------------------------------
    # Exact MobileCLIP preprocessing
    # --------------------------------------------------------

    print("\nMobileCLIP preprocessing")
    print("========================")

    x = preprocess_mobileclip(image)

    print("shape :", x.shape)
    print("dtype :", x.dtype)
    print("min   :", float(x.min()))
    print("max   :", float(x.max()))
    print("mean  :", float(x.mean()))

    # --------------------------------------------------------
    # MobileCLIP ONNX
    # --------------------------------------------------------

    mobileclip_session = create_onnx_session(
        mobileclip_path
    )

    embedding = run_mobileclip_onnx(
        mobileclip_session,
        x,
    )

    print("\nMobileCLIP embedding")
    print("====================")
    print("shape :", embedding.shape)
    print("norm  :", float(np.linalg.norm(embedding)))
    print("min   :", float(embedding.min()))
    print("max   :", float(embedding.max()))
    print("first 8:")
    print(
        np.array2string(
            embedding[:8],
            precision=8,
        )
    )

    # --------------------------------------------------------
    # Photon ONNX
    # --------------------------------------------------------

    prefill_session = create_onnx_session(
        prefill_path
    )

    decode_session = create_onnx_session(
        decode_path
    )

    print("\nPhoton ONNX generation")
    print("======================")

    token_ids = generate_onnx(
        prefill_session,
        decode_session,
        embedding,
        bos_id,
        eos_id,
    )

    caption = decode_token_ids(
        tokenizer_json,
        token_ids,
    )

    print("token IDs:")
    print(token_ids)

    print("\nONNX caption:")
    print(caption)

    # --------------------------------------------------------
    # C++ production pipeline
    # --------------------------------------------------------

    if not cpp_exe.exists():
        print(
            "\nC++ skipped:"
            f" executable not found: {cpp_exe}"
        )

    elif not cpp_tokenizer.exists():
        print(
            "\nC++ skipped:"
            f" native tokenizer not found: {cpp_tokenizer}"
        )

        print(
            "\nIMPORTANT:"
        )
        print(
            "Use photon.tokenizer here, NOT "
            "mobilecap_tokenizer.json."
        )

    elif not cpp_photon.exists():
        print(
            "\nC++ skipped:"
            f" Photon model not found: {cpp_photon}"
        )

    else:

        return_code, cpp_output = run_cpp(
            cpp_exe,
            mobileclip_path,
            cpp_photon,
            cpp_tokenizer,
            image,
        )

        # ----------------------------------------------------
        # Extract C++ caption
        # ----------------------------------------------------

        cpp_caption = cpp_output.strip()

        # Remove possible diagnostic lines if emitted
        # through stdout.
        lines = [
            line.strip()
            for line in cpp_caption.splitlines()
            if line.strip()
        ]

        if lines:
            cpp_caption = lines[-1]

        print("\nC++ caption:")
        print(cpp_caption)

        print(
            "\nC++ return code:",
            return_code,
        )

        # ----------------------------------------------------
        # Final comparison
        # ----------------------------------------------------

        print("\nFinal comparison")
        print("================")

        print("ONNX:")
        print(caption)

        print("\nC++:")
        print(cpp_caption)

        print(
            "\nEXACT CAPTION MATCH:",
            caption == cpp_caption,
        )

    print("\n" + "=" * 72)
    print("TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()