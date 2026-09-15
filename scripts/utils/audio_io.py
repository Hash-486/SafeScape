"""Audio loading, resampling, denoising, and fixed-length windowing."""
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
HOP_SAMPLES = WINDOW_SAMPLES // 2  # 50% overlap


def load_and_resample(path, sr=SAMPLE_RATE):
    y, orig_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y


def denoise(y, sr=SAMPLE_RATE):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr, stationary=False).astype(np.float32)
    except Exception:
        return y


def window_signal(y, win=WINDOW_SAMPLES, hop=HOP_SAMPLES, min_energy=1e-4):
    """Split a 1D signal into fixed-length windows with 50% overlap.
    Drops near-silent windows (below min_energy RMS) and zero-pads the last one.
    """
    n = len(y)
    if n == 0:
        return []
    windows = []
    start = 0
    if n <= win:
        w = np.zeros(win, dtype=np.float32)
        w[:n] = y
        if np.sqrt(np.mean(w ** 2)) >= min_energy:
            windows.append(w)
        return windows
    while start < n:
        end = start + win
        if end <= n:
            w = y[start:end]
        else:
            w = np.zeros(win, dtype=np.float32)
            w[: n - start] = y[start:n]
        if np.sqrt(np.mean(w ** 2)) >= min_energy:
            windows.append(w.astype(np.float32))
        start += hop
    return windows


def save_window(y, path, sr=SAMPLE_RATE):
    sf.write(path, y, sr, subtype="PCM_16")


def load_window(path, sr=SAMPLE_RATE):
    y, file_sr = sf.read(path, always_2d=False)
    y = y.astype(np.float32)
    assert file_sr == sr, f"{path} has sr={file_sr}, expected {sr}"
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    elif len(y) > WINDOW_SAMPLES:
        y = y[:WINDOW_SAMPLES]
    return y
