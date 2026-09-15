"""Loads the exported champion model bundle and reproduces training preprocessing exactly."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.utils import audio_io, features as feat_mod
from scripts.utils.models import build_model
from scripts.utils.labels import LABELS, HAZARD_LABELS

EXPORTED_DIR = ROOT / "models" / "exported"


class Predictor:
    def __init__(self, exported_dir=EXPORTED_DIR):
        exported_dir = Path(exported_dir)
        with open(exported_dir / "label_map.json") as f:
            self.label_map = json.load(f)
        with open(exported_dir / "preprocess_config.json") as f:
            self.cfg = json.load(f)

        self.model_name = self.cfg["model_name"]
        self.feature_type = self.cfg["feature_type"]
        self.sample_rate = self.cfg["sample_rate"]
        self.window_samples = self.cfg["window_samples"]

        self.device = torch.device("cpu")  # serving target is CPU per proposal's low-cost/on-device framing
        model_kwargs = {"hidden_size": self.cfg["hidden_size"]} if self.model_name == "logmel_crnn" else {}
        self.model = build_model(self.model_name, **model_kwargs)
        self.model.eval()
        if self.cfg.get("quantized"):
            import torch.nn as nn
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        # dynamic-quantized GRU state contains packed torch.ScriptObject params that
        # weights_only=True can't unpickle; safe here since this is a bundle we export
        # ourselves (see 11_quantize_export.py), never an externally-sourced checkpoint.
        state = torch.load(exported_dir / "best_model.pt", map_location=self.device,
                            weights_only=not self.cfg.get("quantized"))
        self.model.load_state_dict(state)
        self.model.eval()

    def _prep_waveform(self, y, orig_sr):
        if orig_sr != self.sample_rate:
            import librosa
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sample_rate)
        y = audio_io.denoise(y, sr=self.sample_rate)
        n = self.window_samples
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        else:
            y = y[-n:]  # most recent window
        return y.astype(np.float32)

    def _extract_features(self, y):
        if self.feature_type == "mfcc":
            f = feat_mod.mfcc_features(y, sr=self.sample_rate)
        else:
            f = feat_mod.logmel_features(y, sr=self.sample_rate)
        f = feat_mod.normalize(f)
        return torch.from_numpy(f).unsqueeze(0).unsqueeze(0)  # (1,1,F,T)

    @torch.no_grad()
    def predict(self, y, orig_sr):
        y = self._prep_waveform(y, orig_sr)
        x = self._extract_features(y)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        return {
            "predicted_class": label,
            "confidence": float(probs[idx]),
            "is_hazard": label in HAZARD_LABELS,
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        }
