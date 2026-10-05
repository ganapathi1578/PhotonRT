import argparse
import time

import torch
from PIL import Image
import mobileclip


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Force CPU for a fair comparison.
    device = "cpu"

    print("Device:", device)

    model, _, preprocess = (
        mobileclip.create_model_and_transforms(
            "mobileclip_s1",
            pretrained=args.checkpoint,
        )
    )

    model = model.to(device).eval()

    image = Image.open(
        args.image
    ).convert("RGB")

    x = preprocess(image).unsqueeze(0)

    # Warmup.
    with torch.inference_mode():
        _ = model.encode_image(x)

    times = []

    for i in range(args.iterations):
        t0 = time.perf_counter()

        with torch.inference_mode():
            embedding = model.encode_image(x)

        # Force completion on CPU.
        embedding = embedding.contiguous()

        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0

        times.append(ms)

        print(
            f"iteration {i + 1}: "
            f"{ms:.3f} ms"
        )

    print()
    print(
        f"average: "
        f"{sum(times) / len(times):.3f} ms"
    )


if __name__ == "__main__":
    main()