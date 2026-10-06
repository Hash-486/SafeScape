"""FastAPI inference server: serves /predict, /health, and the mobile-styled web frontend,
plus the multi-model endpoints behind the comparison dashboard at /compare."""
import csv
import json
import subprocess
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.inference import Predictor, MODELS, EXPORTED_DIR
from server.schemas import PredictResponse, PredictAllResponse, HealthResponse

app = FastAPI(title="SafeScape Inference Server")

DEFAULT_MODEL = "logmel_crnn"
OWNERS = {"mfcc_cnn": "Amruth Rohan KR", "logmel_crnn": "Harish Venkat VS",
          "transformer": "Harish Venkat VS"}
METRICS_DIR = ROOT / "reports" / "metrics"
TEST_CLIPS = ROOT / "demo_clips" / "test"

_predictors = {}


def exported(key):
    return (EXPORTED_DIR / key / "best_model.pt").exists()


def get_predictor(key=DEFAULT_MODEL):
    if key not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model '{key}'")
    if not exported(key):
        raise HTTPException(status_code=503, detail=f"model '{key}' is not exported")
    if key not in _predictors:
        _predictors[key] = Predictor(EXPORTED_DIR / key)
    return _predictors[key]


@app.on_event("startup")
def warm_up():
    # the first predict pays for model load and noisereduce's first call (~1.3s); do it
    # here so the dashboard's latency figures show inference, not cold start
    # a model without a bundle is skipped, not fatal: the others (and /predict) still serve
    for key in filter(exported, MODELS):
        get_predictor(key).predict(np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32), 16000)


def timed_predict(key, y, sr):
    p = get_predictor(key)
    t0 = time.perf_counter()
    # predict() resamples/denoises its input; each model must see the same raw clip
    out = p.predict(y.copy(), sr)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


def decode_audio_bytes(raw: bytes):
    """Transcode arbitrary browser-recorded audio (webm/opus, ogg, wav, ...) to
    16kHz mono PCM float32 via a bundled static ffmpeg binary (no system install needed)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore')[:500]}")
    y, sr = sf.read(io.BytesIO(proc.stdout), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


async def read_upload(file):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio upload")
    try:
        y, sr = decode_audio_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {e}")
    if len(y) == 0:
        raise HTTPException(status_code=400, detail="decoded audio is empty")
    return y, sr


@app.get("/health", response_model=HealthResponse)
def health():
    p = get_predictor()
    return HealthResponse(status="ok", model_name=p.model_name)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), model: str = DEFAULT_MODEL):
    get_predictor(model)  # unknown key -> 404 before paying for the ffmpeg decode
    y, sr = await read_upload(file)
    return PredictResponse(**timed_predict(model, y, sr))


@app.post("/predict/all", response_model=PredictAllResponse)
async def predict_all(file: UploadFile = File(...)):
    y, sr = await read_upload(file)
    return PredictAllResponse(results={k: PredictResponse(**timed_predict(k, y, sr))
                                       for k in filter(exported, MODELS)})


@app.get("/models")
def models():
    out = []
    for key, name in MODELS.items():
        # absent until scripts/run_v2_all_models.sh has evaluated this model
        mpath = METRICS_DIR / f"{key}_v2.json"
        out.append({
            "key": key, "name": name, "owner": OWNERS[key],
            "metrics": json.loads(mpath.read_text()) if mpath.exists() else None,
            "bundle_kb": (EXPORTED_DIR / key / "best_model.pt").stat().st_size / 1024
                         if exported(key) else None,
        })
    return {"models": out}


@app.get("/testclips")
def testclips():
    with open(TEST_CLIPS / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    return {"clips": [{"file": r["file"], "label": r["label"], "url": f"/clips/{r['file']}"}
                      for r in rows]}


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/clips", StaticFiles(directory=str(TEST_CLIPS)), name="clips")
app.mount("/figures", StaticFiles(directory=str(ROOT / "reports" / "figures")), name="figures")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/compare")
def compare():
    return FileResponse(str(STATIC_DIR / "compare.html"))
