from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from . import _photonrt
from .hub import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    download_model,
)

__version__ = "0.2.0"


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
        repo_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        cache_dir: Optional[PathLike] = None,
    ) -> "Captioner":
        """
        Download or load a cached PhotonRT model.

        Parameters
        ----------
        repo_id:
            Hugging Face repository containing PhotonRT model files.

        revision:
            Hugging Face branch, tag, or commit.

        cache_dir:
            Optional custom Hugging Face cache directory.
        """

        paths = download_model(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )

        return cls(
            mobileclip_model=paths["mobileclip"],
            photon_prefill_model=paths["prefill"],
            photon_decode_model=paths["decode"],
            tokenizer=paths["tokenizer"],
        )


from .stream import CameraCaptioner, CaptionResult


CaptionerNative = _photonrt.Captioner
CaptionOptions = _photonrt.CaptionOptions


__all__ = [
    "Captioner",
    "CaptionerNative",
    "CaptionOptions",
    "CameraCaptioner",
    "CaptionResult",
    "DEFAULT_MODEL_ID",
    "DEFAULT_REVISION",
    "__version__",
]   