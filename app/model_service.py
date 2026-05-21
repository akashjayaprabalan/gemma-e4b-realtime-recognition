from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from app.config import MAX_TOKENS, MODEL_ID, MODEL_REVISION, RECOGNITION_PROMPT
from app.vision_utils import normalize_result, parse_model_output


class ModelService:
    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION) -> None:
        self.model_id = model_id
        self.revision = revision
        self._model: Any | None = None
        self._processor: Any | None = None
        self._generate: Any | None = None
        self._apply_chat_template: Any | None = None
        self._load_error: str | None = None
        self._lock = asyncio.Lock()

    def status(self) -> dict[str, Any]:
        if self._model is not None:
            state = "ready"
        elif self._lock.locked():
            state = "loading"
        elif self._load_error:
            state = "error"
        else:
            state = "cold"
        return {
            "state": state,
            "model_id": self.model_id,
            "revision": self.revision,
            "loaded": self._model is not None,
            "error": self._load_error,
        }

    async def recognize(self, image_path: Path) -> dict[str, Any]:
        async with self._lock:
            started = time.perf_counter()
            try:
                output = await asyncio.to_thread(self._recognize_sync, image_path)
            except Exception as exc:
                self._load_error = str(exc)
                raise
            latency_ms = round((time.perf_counter() - started) * 1000)

        payload = parse_model_output(output)
        payload["latency_ms"] = latency_ms
        payload["model_id"] = self.model_id
        return payload

    def _recognize_sync(self, image_path: Path) -> str:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._generate is not None
        assert self._apply_chat_template is not None

        prompt = self._apply_chat_template(
            self._processor,
            self._model.config,
            RECOGNITION_PROMPT,
            num_images=1,
        )
        result = self._generate(
            model=self._model,
            processor=self._processor,
            prompt=prompt,
            image=[str(image_path)],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        return normalize_result(result)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template

            self._model, self._processor = load(self.model_id, revision=self.revision)
            self._generate = generate
            self._apply_chat_template = apply_chat_template
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            raise


model_service = ModelService()
