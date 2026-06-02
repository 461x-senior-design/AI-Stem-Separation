# src/stemmy/hub.py
"""Default model resolution via Hugging Face Hub.

The published wheel does not bundle model weights (PyPI per-file limit is 100 MB).
Instead, the default model is hosted on Hugging Face and downloaded on first use,
then cached under ~/.cache/huggingface/hub/.

Revision is pinned so that reuploads to the HF repo do not silently change the
output for installed versions of stemmy.
"""

from __future__ import annotations

DEFAULT_REPO_ID: str = "jscervantes/stemmy-0.1"
DEFAULT_FILENAME: str = "default.pth"
DEFAULT_REVISION: str = "62a5951e7e9519eefdfd5e7b43ee0e63357be787"


def get_default_model(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    revision: str = DEFAULT_REVISION,
) -> str:
    """Return a local filesystem path to the default model checkpoint.

    Downloads from Hugging Face Hub on first call; returns the cached path
    thereafter. Raises ImportError if `huggingface_hub` is not installed.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
