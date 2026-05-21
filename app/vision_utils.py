from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.config import JPEG_QUALITY, MAX_IMAGE_EDGE

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ImageInputError(ValueError):
    """Raised when an uploaded frame cannot be decoded as an image."""


def normalize_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(value, dict):
        for key in ("text", "generated_text", "output"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    return str(value).strip()


def parse_model_output(raw_output: str) -> dict[str, Any]:
    text = (raw_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    match = JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return _coerce_payload(payload, raw_output)

    return {
        "caption": text,
        "labels": [],
        "objects": [],
        "text_seen": [],
        "alerts": [],
        "raw_output": raw_output,
    }


def prepare_frame(data: bytes) -> tuple[Path, dict[str, int]]:
    try:
        with Image.open(_BytesReader(data)) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
            width, height = image.size

            handle = tempfile.NamedTemporaryFile(
                prefix="gemma-frame-", suffix=".jpg", delete=False
            )
            path = Path(handle.name)
            with handle:
                image.save(handle, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError("Upload must be a readable image.") from exc

    return path, {
        "original_width": original_width,
        "original_height": original_height,
        "width": width,
        "height": height,
    }


def _coerce_payload(payload: dict[str, Any], raw_output: str) -> dict[str, Any]:
    return {
        "caption": _string(payload.get("caption") or payload.get("description")),
        "labels": _string_list(payload.get("labels")),
        "objects": _string_list(payload.get("objects")),
        "text_seen": _string_list(payload.get("text_seen") or payload.get("text")),
        "alerts": _string_list(payload.get("alerts")),
        "raw_output": raw_output,
    }


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [_string(item) for item in value if _string(item)]
    return [_string(value)] if _string(value) else []


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self.data) - self.offset
        chunk = self.data[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.data) + offset
        else:
            raise ValueError("invalid whence")
        self.offset = max(0, min(self.offset, len(self.data)))
        return self.offset

    def tell(self) -> int:
        return self.offset
