# Gemma 4 E4B Real-Time Image Recognition

Local webcam recognition app for Apple Silicon using the MLX 8-bit Gemma 4 E4B-it model:

- Model: `mlx-community/gemma-4-e4b-it-8bit`
- Runtime: `mlx-vlm`
- UI: browser webcam capture with a 2-second sampling timer that skips overlapping requests
- API: FastAPI

## Setup

Use Python 3.13 on this Mac. The system `python3` is newer, so call `python3.13` explicitly.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Check the machine and Hub model:

```sh
.venv/bin/python scripts/preflight.py
```

Download the model into the Hugging Face cache:

```sh
.venv/bin/python scripts/download_model.py
```

Run the app:

```sh
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API

```sh
curl http://127.0.0.1:8000/api/health
curl -F "image=@sample.jpg" http://127.0.0.1:8000/api/recognize
```

`POST /api/recognize` returns:

```json
{
  "caption": "A person at a desk with a laptop.",
  "labels": ["person", "laptop", "desk"],
  "objects": ["person", "laptop", "desk"],
  "text_seen": [],
  "alerts": [],
  "raw_output": "...",
  "latency_ms": 2400,
  "model_id": "mlx-community/gemma-4-e4b-it-8bit"
}
```

## Notes

The app performs sampled live recognition, not full video-rate object detection. Warm Gemma 4 E4B 8-bit inference on this 16 GB Apple Silicon Mac took about 6 seconds on the included API smoke test, so the browser timer samples every 2 seconds but waits for the current recognition call to finish before sending another frame. Model weights stay in the Hugging Face cache and are never written into this repository.
