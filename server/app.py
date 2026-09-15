"""FastAPI inference server: serves /predict, /health, and the mobile-styled web frontend."""
import subprocess
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.inference import Predictor
from server.schemas import PredictResponse, HealthResponse

app = FastAPI(title="SafeScape Inference Server")

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


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


@app.get("/health", response_model=HealthResponse)
def health():
    p = get_predictor()
    return HealthResponse(status="ok", model_name=p.model_name)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio upload")
    try:
        y, sr = decode_audio_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {e}")
    if len(y) == 0:
        raise HTTPException(status_code=400, detail="decoded audio is empty")
    p = get_predictor()
    result = p.predict(y, sr)
    return PredictResponse(**result)


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
