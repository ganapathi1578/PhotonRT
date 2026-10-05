#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# Constants
# ============================================================

NUM_LAYERS = 6
NUM_IMG_TOKENS = 8
IMG_EMB_DIM = 512
MAX_TEXT_LENGTH = 32


# ============================================================
# Tokenizer
# ============================================================

def load_special_tokens(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    special = {}

    for item in data.get("added_tokens", []):
        content = item.get("content")
        token_id = item.get("id")

        if content is not None and token_id is not None:
            special[content] = int(token_id)

    for token in ["[CLS]", "[SEP]"]:
        if token not in special:
            raise RuntimeError(
                f"Missing {token} in tokenizer JSON"
            )

    return special


def decode_tokens(tokenizer_path: Path, ids):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
    )


# ============================================================
# Exact preprocessing used by PhotonRT
# ============================================================

def create_preprocess():
    """
    Matches the native PhotonRT MobileCLIP preprocessing:

        Resize(short edge -> 256)
        CenterCrop(256)
        ToTensor()

    No ImageNet/CLIP normalization.
    """

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose([
        transforms.Resize(
            256,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])


# ============================================================
# ONNX Runtime session
# ============================================================

def make_session(path: Path, threads: int):

    opts = ort.SessionOptions()

    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    opts.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1

    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True

    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


# ============================================================
# Statistics
# ============================================================

def stats(values):
    if not values:
        return {
            "avg": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "max": 0.0,
        }

    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "p50": statistics.median(values),
        "max": max(values),
    }


def print_stats(name, values):
    s = stats(values)

    print(
        f"{name:<28}"
        f"avg={s['avg']:8.3f} ms   "
        f"min={s['min']:8.3f} ms   "
        f"p50={s['p50']:8.3f} ms   "
        f"max={s['max']:8.3f} ms"
    )


# ============================================================
# Photon ONNX generation
# ============================================================

def photon_generate(
    prefill_session,
    decode_session,
    embedding,
    bos_id,
    eos_id,
    timing,
):

    image_embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(1, IMG_EMB_DIM)

    input_ids = np.array(
        [[bos_id]],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Prefill
    # --------------------------------------------------------

    t0 = time.perf_counter()

    outputs = prefill_session.run(
        None,
        {
            "image_embedding": image_embedding,
            "input_ids": input_ids,
        },
    )

    timing["prefill"].append(
        (time.perf_counter() - t0) * 1000.0
    )

    logits = outputs[0]

    cache = {}

    for layer in range(NUM_LAYERS):

        cache[
            f"past_key_{layer}"
        ] = outputs[1 + 2 * layer]

        cache[
            f"past_value_{layer}"
        ] = outputs[2 + 2 * layer]

    # --------------------------------------------------------
    # First token selection
    # --------------------------------------------------------

    next_token = int(
        np.argmax(
            logits[0, -1]
        )
    )

    generated = []

    # image tokens = positions 0..7
    # BOS          = position 8
    # first decode = position 9

    position = NUM_IMG_TOKENS + 1

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decode_onnx_total = 0.0
    decode_wall_total = 0.0

    for _ in range(MAX_TEXT_LENGTH):

        if next_token == eos_id:
            break

        generated.append(next_token)

        token_input = np.array(
            [[next_token]],
            dtype=np.int64,
        )

        position_ids = np.array(
            [[position]],
            dtype=np.int64,
        )

        feed = {
            "input_ids": token_input,
            "position_ids": position_ids,
        }

        feed.update(cache)

        # Entire decode step.
        wall_start = time.perf_counter()

        onnx_start = time.perf_counter()

        outputs = decode_session.run(
            None,
            feed,
        )

        onnx_elapsed = (
            time.perf_counter() - onnx_start
        ) * 1000.0

        wall_elapsed = (
            time.perf_counter() - wall_start
        ) * 1000.0

        decode_onnx_total += onnx_elapsed
        decode_wall_total += wall_elapsed

        timing["decode_onnx"].append(
            onnx_elapsed
        )

        timing["decode_step"].append(
            wall_elapsed
        )

        logits = outputs[0]

        new_cache = {}

        for layer in range(NUM_LAYERS):

            new_cache[
                f"past_key_{layer}"
            ] = outputs[1 + 2 * layer]

            new_cache[
                f"past_value_{layer}"
            ] = outputs[2 + 2 * layer]

        cache = new_cache

        next_token = int(
            np.argmax(
                logits[0, -1]
            )
        )

        position += 1

    timing["decode_total"].append(
        decode_onnx_total
    )

    timing["decode_wall_total"].append(
        decode_wall_total
    )

    return generated


# ============================================================
# Main benchmark
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--tokenizer",
        required=True,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    mobileclip_path = Path(args.mobileclip)
    prefill_path = Path(args.prefill)
    decode_path = Path(args.decode)
    tokenizer_path = Path(args.tokenizer)

    for p in [
        image_path,
        mobileclip_path,
        prefill_path,
        decode_path,
        tokenizer_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    if args.iterations <= 0:
        raise ValueError(
            "--iterations must be > 0"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    special = load_special_tokens(
        tokenizer_path
    )

    bos_id = special["[CLS]"]
    eos_id = special["[SEP]"]

    print()
    print("Photon end-to-end ONNX benchmark")
    print("=" * 72)
    print("image       :", image_path)
    print("mobileclip  :", mobileclip_path)
    print("prefill     :", prefill_path)
    print("decode      :", decode_path)
    print("threads     :", args.threads)
    print("warmup      :", args.warmup)
    print("iterations  :", args.iterations)
    print("BOS [CLS]   :", bos_id)
    print("EOS [SEP]   :", eos_id)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocess = create_preprocess()

    # --------------------------------------------------------
    # Load ONNX sessions
    # --------------------------------------------------------

    print()
    print("Loading ONNX models...")

    mobileclip_session = make_session(
        mobileclip_path,
        args.threads,
    )

    prefill_session = make_session(
        prefill_path,
        args.threads,
    )

    decode_session = make_session(
        decode_path,
        args.threads,
    )

    print("MobileCLIP providers:",
          mobileclip_session.get_providers())

    print("Photon prefill providers:",
          prefill_session.get_providers())

    print("Photon decode providers:",
          decode_session.get_providers())

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print("Warmup...")

    for _ in range(args.warmup):

        image = Image.open(
            image_path
        ).convert("RGB")

        x = preprocess(image)

        x = np.ascontiguousarray(
            np.asarray(
                x,
                dtype=np.float32,
            )[None, ...]
        )

        embedding = mobileclip_session.run(
            ["embedding"],
            {
                mobileclip_session.get_inputs()[0].name: x
            },
        )[0]

        dummy_timing = {
            "prefill": [],
            "decode_onnx": [],
            "decode_step": [],
            "decode_total": [],
            "decode_wall_total": [],
        }

        photon_generate(
            prefill_session,
            decode_session,
            embedding,
            bos_id,
            eos_id,
            dummy_timing,
        )

    # --------------------------------------------------------
    # Measurement storage
    # --------------------------------------------------------

    load_times = []
    preprocess_times = []
    mobileclip_times = []
    prefill_times = []
    decode_step_times = []
    decode_onnx_times = []
    decode_total_times = []
    end_to_end_times = []

    token_counts = []

    last_ids = []
    last_caption = ""

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print()
    print("Benchmarking...")

    for i in range(args.iterations):

        e2e_start = time.perf_counter()

        # --------------------------------------------
        # Image loading
        # --------------------------------------------

        t0 = time.perf_counter()

        image = Image.open(
            image_path
        ).convert("RGB")

        # Force actual image decoding.
        image.load()

        t1 = time.perf_counter()

        load_times.append(
            (t1 - t0) * 1000.0
        )

        # --------------------------------------------
        # Preprocessing
        # --------------------------------------------

        t0 = time.perf_counter()

        x = preprocess(image)

        x = np.ascontiguousarray(
            np.asarray(
                x,
                dtype=np.float32,
            )[None, ...]
        )

        t1 = time.perf_counter()

        preprocess_times.append(
            (t1 - t0) * 1000.0
        )

        # --------------------------------------------
        # MobileCLIP
        # --------------------------------------------

        input_name = (
            mobileclip_session
            .get_inputs()[0]
            .name
        )

        t0 = time.perf_counter()

        embedding = mobileclip_session.run(
            ["embedding"],
            {
                input_name: x
            },
        )[0]

        t1 = time.perf_counter()

        mobileclip_times.append(
            (t1 - t0) * 1000.0
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------
        # Photon
        # --------------------------------------------

        timing = {
            "prefill": [],
            "decode_onnx": [],
            "decode_step": [],
            "decode_total": [],
            "decode_wall_total": [],
        }

        token_ids = photon_generate(
            prefill_session,
            decode_session,
            embedding,
            bos_id,
            eos_id,
            timing,
        )

        prefill_times.append(
            timing["prefill"][0]
        )

        decode_step_times.extend(
            timing["decode_step"]
        )

        decode_onnx_times.extend(
            timing["decode_onnx"]
        )

        decode_total_times.append(
            timing["decode_total"][0]
        )

        token_counts.append(
            len(token_ids)
        )

        # --------------------------------------------
        # End-to-end
        # --------------------------------------------

        e2e_end = time.perf_counter()

        end_to_end_times.append(
            (e2e_end - e2e_start) * 1000.0
        )

        last_ids = token_ids
        last_caption = decode_tokens(
            tokenizer_path,
            token_ids,
        )

        if (i + 1) % 10 == 0:
            print(
                f"  {i + 1}/{args.iterations}"
            )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print()
    print("Stage timing")
    print("------------")

    print_stats(
        "Image load",
        load_times,
    )

    print_stats(
        "MobileCLIP preprocess",
        preprocess_times,
    )

    print_stats(
        "MobileCLIP ONNX",
        mobileclip_times,
    )

    print_stats(
        "Photon prefill",
        prefill_times,
    )

    print_stats(
        "Photon decode / ONNX call",
        decode_onnx_times,
    )

    print_stats(
        "Photon decode / full step",
        decode_step_times,
    )

    print_stats(
        "Photon decode / sequence",
        decode_total_times,
    )

    print_stats(
        "END-TO-END",
        end_to_end_times,
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    avg_load = statistics.mean(
        load_times
    )

    avg_prep = statistics.mean(
        preprocess_times
    )

    avg_mobileclip = statistics.mean(
        mobileclip_times
    )

    avg_prefill = statistics.mean(
        prefill_times
    )

    avg_decode = statistics.mean(
        decode_total_times
    )

    avg_e2e = statistics.mean(
        end_to_end_times
    )

    avg_tokens = statistics.mean(
        token_counts
    )

    print()
    print("Aggregate")
    print("---------")

    print(
        f"image load           : {avg_load:.3f} ms"
    )

    print(
        f"preprocess           : {avg_prep:.3f} ms"
    )

    print(
        f"MobileCLIP           : {avg_mobileclip:.3f} ms"
    )

    print(
        f"Photon prefill       : {avg_prefill:.3f} ms"
    )

    print(
        f"Photon decode        : {avg_decode:.3f} ms"
    )

    print(
        f"avg tokens           : {avg_tokens:.2f}"
    )

    print(
        f"sum of stages        : "
        f"{avg_load + avg_prep + avg_mobileclip + avg_prefill + avg_decode:.3f} ms"
    )

    print(
        f"measured END-TO-END  : {avg_e2e:.3f} ms"
    )

    print(
        f"end-to-end FPS       : "
        f"{1000.0 / avg_e2e:.3f}"
    )

    if avg_decode > 0.0:
        print(
            f"decode tokens/sec    : "
            f"{avg_tokens / (avg_decode / 1000.0):.3f}"
        )

    # --------------------------------------------------------
    # Final correctness
    # --------------------------------------------------------

    print()
    print("Correctness")
    print("-----------")

    print(
        "token IDs :",
        last_ids,
    )

    print(
        "caption   :",
        last_caption,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()