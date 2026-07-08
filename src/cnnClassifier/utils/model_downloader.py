from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.keras"


def download_model_if_needed() -> str:
    """
    Download the trained model from Hugging Face
    if it does not already exist locally.

    Returns
    -------
    str
        Local path to the model.
    """

    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    downloaded_path = hf_hub_download(
        repo_id="aneeshjose/brain-tumor-cnn",
        filename="model.keras",
        local_dir="models",
        local_dir_use_symlinks=False,
    )

    return downloaded_path