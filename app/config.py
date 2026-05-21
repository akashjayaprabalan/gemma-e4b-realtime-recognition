from __future__ import annotations

import os

MODEL_ID = os.getenv("GEMMA_MODEL_ID", "mlx-community/gemma-4-e4b-it-8bit")
MODEL_REVISION = os.getenv("GEMMA_MODEL_REVISION", "main")
MAX_IMAGE_EDGE = int(os.getenv("GEMMA_MAX_IMAGE_EDGE", "896"))
JPEG_QUALITY = int(os.getenv("GEMMA_JPEG_QUALITY", "86"))
MAX_UPLOAD_BYTES = int(os.getenv("GEMMA_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_TOKENS = int(os.getenv("GEMMA_MAX_TOKENS", "220"))

RECOGNITION_PROMPT = os.getenv(
    "GEMMA_RECOGNITION_PROMPT",
    (
        "Analyze this live webcam frame for real-time image recognition. "
        "Respond only with a compact JSON object with keys: "
        "caption (one short sentence), labels (up to 8 short tags), "
        "objects (up to 8 visible objects), text_seen (visible readable text), "
        "alerts (notable safety or state changes, empty when none). "
        "Do not use markdown."
    ),
)
