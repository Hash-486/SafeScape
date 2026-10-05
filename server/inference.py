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
# one bundle per key under models/exported/<key>/ (scripts/run_v2_all_models.sh)
MODELS = {"mfcc_cnn": "MFCC-CNN", "logmel_crnn": "log-mel CRNN", "transformer": "Distilled Transformer"}


class Predictor:
    def __init__(self, exported_dir=EXPORTED_DIR / "logmel_crnn"):
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

        # per-class logit bias fitted on the val split (scripts/12_calibrate_logits.py);
        # absent for bundles exported before calibration existed
        calib_path = exported_dir / "calibration.json"
        self.logit_bias = torch.zeros(len(LABELS))
        if calib_path.exists():
            with open(calib_path) as f:
                self.logit_bias = torch.tensor(json.load(f)["bias"], dtype=torch.float32)

    def _prep_waveform(self, y, orig_sr):
        if orig_sr != self.sample_rate:
            import librosa
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sample_rate)
        return audio_io.denoise(y, sr=self.sample_rate).astype(np.float32)

    def _extract_features(self, y):
        if self.feature_type == "mfcc":
            f = feat_mod.mfcc_features(y, sr=self.sample_rate)
        else:
            f = feat_mod.logmel_features(y, sr=self.sample_rate)
        return feat_mod.normalize(f)

    def _pick(self, probs):
        """Which window's verdict speaks for the whole clip.

        A missed hazard costs more than a false alarm per the proposal, so any window
        that independently calls a hazard makes the clip that hazard, most confident
        first. Averaging across windows instead would bury a 0.3s glass_break under
        the second of room tone that follows it.
        """
        top = probs.argmax(axis=1)
        hazards = [i for i, t in enumerate(top) if LABELS[t] in HAZARD_LABELS]
        pool = hazards or range(len(top))
        return max(pool, key=lambda i: probs[i, top[i]])

    @torch.no_grad()
    def predict(self, y, orig_sr):
        y = self._prep_waveform(y, orig_sr)
        # Judge every 1s window of the clip, not just the last one. The browser sends
        # 2s and the model's window is 1s, so the old y[-n:] discarded half of every
        # clip: a glass_break in the first half came back as distress_call 72%.
        # This is the same call 03_preprocess_audio.py used to cut the training set,
        # min_energy gate included, so serving sees windows of the kind it was taught.
        windows = audio_io.window_signal(y, win=self.window_samples,
                                         hop=self.window_samples // 2)
        if not windows:
            # Every window fell below the training set's energy floor. The model was
            # never shown silence and reads it as distress_call at 77%, so this answers
            # from the gate rather than asking it a question it cannot have an opinion on.
            probs = np.zeros(len(LABELS), dtype=np.float32)
            probs[LABELS.index("ambience")] = 1.0
        else:
            x = torch.from_numpy(
                np.stack([self._extract_features(w) for w in windows])).unsqueeze(1)
            per_window = F.softmax(self.model(x) + self.logit_bias, dim=1).numpy()
            probs = per_window[self._pick(per_window)]
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        return {
            "predicted_class": label,
            "confidence": float(probs[idx]),
            "is_hazard": label in HAZARD_LABELS,
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
            "model_name": self.model_name,
        }
