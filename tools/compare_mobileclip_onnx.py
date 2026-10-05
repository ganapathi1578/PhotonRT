import argparse
import numpy as np
import torch
import timm
import onnxruntime as ort
from PIL import Image


IMAGE_SIZE = 256


def preprocess_image(path):
    image = Image.open(path).convert("RGB")

    # MobileCLIP preprocessing used by the native pipeline:
    # resize shortest side to 256, then center crop 256x256.
    w, h = image.size

    if w < h:
        new_w = IMAGE_SIZE
        new_h = round(h * IMAGE_SIZE / w)
    else:
        new_h = IMAGE_SIZE
        new_w = round(w * IMAGE_SIZE / h)

    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2

    image = image.crop(
        (left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)
    )

    x = np.asarray(image).astype(np.float32) / 255.0

    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))

    # IMPORTANT:
    # Match PhotonRT's MobileCLIP normalization.
    mean = np.array(
        [0.48145466, 0.4578275, 0.40821073],
        dtype=np.float32,
    )

    std = np.array(
        [0.26862954, 0.26130258, 0.27577711],
        dtype=np.float32,
    )

    x = (x - mean[:, None, None]) / std[:, None, None]

    return x[None, ...]


def load_timm(checkpoint):
    model = timm.create_model(
        "fastvit_mci1",
        pretrained=True,
        pretrained_cfg_overlay={"file": checkpoint},
    )

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------

    x = preprocess_image(args.image)

    print("input shape:", x.shape)
    print("input dtype:", x.dtype)

    # ------------------------------------------------------------
    # PyTorch / timm
    # ------------------------------------------------------------

    model = load_timm(args.checkpoint)

    with torch.no_grad():
        ref = model(torch.from_numpy(x)).cpu()

        # Match the normalized ONNX output.
        ref = torch.nn.functional.normalize(ref, dim=-1)

        ref = ref.numpy()
    # ------------------------------------------------------------
    # ONNX Runtime
    # ------------------------------------------------------------

    session = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    out = session.run(
        None,
        {
            input_name: x,
        },
    )[0]

    # ------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------

    diff = np.abs(ref - out)

    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    ref_flat = ref.reshape(-1)
    out_flat = out.reshape(-1)

    cosine = float(
        np.dot(ref_flat, out_flat)
        /
        (
            np.linalg.norm(ref_flat)
            *
            np.linalg.norm(out_flat)
        )
    )

    print()
    print("MobileCLIP ONNX parity")
    print("======================")
    print(f"reference shape : {ref.shape}")
    print(f"onnx shape      : {out.shape}")
    print(f"max_abs_error   : {max_abs:.9e}")
    print(f"mean_abs_error  : {mean_abs:.9e}")
    print(f"cosine          : {cosine:.10f}")
    print(f"ref_norm        : {np.linalg.norm(ref_flat):.10f}")
    print(f"onnx_norm       : {np.linalg.norm(out_flat):.10f}")


if __name__ == "__main__":
    main()