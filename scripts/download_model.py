from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("GEMMA_MODEL_ID", "mlx-community/gemma-4-e4b-it-8bit")
REVISION = os.getenv("GEMMA_MODEL_REVISION", "main")


def main() -> int:
    path = snapshot_download(repo_id=MODEL_ID, revision=REVISION, repo_type="model")
    print(f"Downloaded {MODEL_ID}@{REVISION}")
    print(Path(path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
