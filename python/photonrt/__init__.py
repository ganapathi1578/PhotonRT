from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from . import _photonrt


__version__ = "0.1.0"


PathLike = Union[str, Path]


class Captioner:
    """
    Native PhotonRT image captioner.

    The actual inference is performed entirely in C++.
    """

    def __init__(
        self,
        mobileclip_model: PathLike,
        photon_prefill_model: PathLike,
        photon_decode_model: PathLike,
        tokenizer: PathLike,
    ):
        self._native = _photonrt.Captioner(
            str(mobileclip_model),
            str(photon_prefill_model),
            str(photon_decode_model),
            str(tokenizer),
        )

    def caption(
        self,
        image: PathLike,
        max_new_tokens: int = 32,
    ) -> str:
        """
        Caption one image.

        Image loading, preprocessing, MobileCLIP,
        Photon prefill/decode and tokenization all
        happen inside the native C++ runtime.
        """

        image = Path(image)

        if not image.exists():
            raise FileNotFoundError(
                f"Image not found: {image}"
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be > 0"
            )

        options = _photonrt.CaptionOptions()

        options.max_new_tokens = int(
            max_new_tokens
        )

        return self._native.caption(
            str(image),
            options,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: Optional[PathLike] = None,
    ) -> "Captioner":
        """
        Download the PhotonRT model files from Hugging Face
        and construct the native C++ runtime.
        """

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for "
                "Captioner.from_pretrained(). "
                "Install it with: pip install huggingface_hub"
            ) from exc

        allow_patterns = [
            "mobileclip-s1.onnx",
            "photon_prefill.onnx",
            "photon.onnx",
            "photon.tokenizer",
            "config.json",
        ]

        kwargs = {
            "repo_id": repo_id,
            "revision": revision,
            "allow_patterns": allow_patterns,
        }

        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        model_dir = Path(
            snapshot_download(**kwargs)
        )

        required = [
            "mobileclip-s1.onnx",
            "photon_prefill.onnx",
            "photon.onnx",
            "photon.tokenizer",
        ]

        paths = {}

        for filename in required:
            path = model_dir / filename

            if not path.exists():
                raise RuntimeError(
                    f"Missing model file in HF repository: "
                    f"{filename}"
                )

            paths[filename] = path

        return cls(
            mobileclip_model=paths[
                "mobileclip-s1.onnx"
            ],
            photon_prefill_model=paths[
                "photon_prefill.onnx"
            ],
            photon_decode_model=paths[
                "photon.onnx"
            ],
            tokenizer=paths[
                "photon.tokenizer"
            ],
        )


CaptionerNative = _photonrt.Captioner
CaptionOptions = _photonrt.CaptionOptions


__all__ = [
    "Captioner",
    "CaptionerNative",
    "CaptionOptions",
    "__version__",
]