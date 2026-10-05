from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

DEFAULT_MODEL_ID = "ganapathi1578/PhotonRT"
DEFAULT_REVISION = "main"

PathLike = Union[str, Path]

MODEL_FILES = {
    "mobileclip": "mobileclip-s1.onnx",
    "prefill": "photon_prefill.onnx",
    "decode": "photon.onnx",
    "tokenizer": "photon.tokenizer",
}


def download_model(
    repo_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: Optional[PathLike] = None,
) -> Dict[str, Path]:
    """
    Download or resolve a cached PhotonRT model.

    Returns paths to the native runtime model files.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download PhotonRT models. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "allow_patterns": list(MODEL_FILES.values()),
    }

    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    model_dir = Path(
        snapshot_download(**kwargs)
    )

    paths = {}

    for key, filename in MODEL_FILES.items():
        path = model_dir / filename

        if not path.is_file():
            raise RuntimeError(
                f"PhotonRT model is missing required file: "
                f"{filename}\n"
                f"Repository: {repo_id}\n"
                f"Revision: {revision}"
            )

        paths[key] = path

    return paths
