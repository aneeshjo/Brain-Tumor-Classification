from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "aneeshjose/brain-tumor-cnn"
MODEL_NAME = "model.keras"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_NAME


def download_model_if_needed() -> str:
    """
    Download the trained model from Hugging Face if it
    does not already exist locally.

    Returns
    -------
    str
        Local path to the downloaded model.
    """

    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_NAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    return str(MODEL_PATH)