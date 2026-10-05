"""Create the exact 1x3x256x256 float32 input used by Apple's MobileCLIP v1 API."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import mobileclip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("image_chw.f32"))
    args = ap.parse_args()

    _, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s1", pretrained=None, device="cpu"
    )
    image = Image.open(args.image).convert("RGB")
    x = preprocess(image).float().contiguous()
    if tuple(x.shape) != (3, 256, 256):
        raise RuntimeError(f"Unexpected MobileCLIP input shape: {tuple(x.shape)}")

    # ToTensor() is already in CHW float32 [0,1] format. The official current
    # MobileCLIP v1 helper does resize + center crop + ToTensor and does not add
    # an ImageNet Normalize transform.
    x.numpy().tofile(args.output)
    print(f"Wrote {args.output}")
    print(f"shape={tuple(x.shape)}, min={x.min().item():.6f}, max={x.max().item():.6f}")


if __name__ == "__main__":
    main()
