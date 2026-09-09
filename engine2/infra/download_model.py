"""Download the TabFM 1.0.0 PyTorch checkpoint (regression + classification heads, ~12 GB) into models/.

Idempotent: re-running only fetches what is missing. No Hugging Face account needed.
"""

from huggingface_hub import snapshot_download

from infra.config import MODEL_DIR

REPO_ID = "google/tabfm-1.0.0-pytorch"


def download():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        local_dir=MODEL_DIR,
        allow_patterns=["config.json", "LICENSE", "README.md", "regression/*", "classification/*"],
    )
    print("model ready:", MODEL_DIR)


if __name__ == "__main__":
    download()
