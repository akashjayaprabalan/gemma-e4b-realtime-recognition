from __future__ import annotations

import platform
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import MAX_UPLOAD_BYTES
from app.model_service import model_service
from app.vision_utils import ImageInputError, prepare_frame

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Gemma 4 E4B Image Recognition", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "app": "ok",
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "model": model_service.status(),
    }


@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...)) -> JSONResponse:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")

    try:
        frame_path, frame_info = prepare_frame(data)
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await model_service.recognize(frame_path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Recognition failed: {exc}") from exc
    finally:
        frame_path.unlink(missing_ok=True)

    result["frame"] = frame_info
    return JSONResponse(result)
