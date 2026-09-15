"""PyTorch Dataset over precomputed MFCC/log-mel .npy feature caches (see
04_extract_features.py), with fast spectrogram-domain augmentation (train split
only). Falls back to on-the-fly extraction from the wav if a cache miss occurs."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import audio_io, features as feat_mod, augment as aug_mod
from .labels import LABEL_TO_IDX, HAZARD_LABELS
from pathlib import Path

# hazard classes get augmented more often than ambience (programmatic minority upsampling)
HAZARD_AUG_P = 0.7
AMBIENCE_AUG_P = 0.3

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT / "data" / "processed" / "features"


def feature_path(window_path: str, feature_type: str) -> Path:
    p = Path(window_path)
    cls = p.parent.name
    return FEATURES_DIR / feature_type / cls / (p.stem + ".npy")


class SafeScapeDataset(Dataset):
    def __init__(self, manifest_csv, split, feature_type="mfcc", augment=None):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        assert len(self.df) > 0, f"no rows for split={split} in {manifest_csv}"
        self.feature_type = feature_type
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.df)

    def _load_feature(self, row):
        cache_path = feature_path(row["window_path"], self.feature_type)
        if cache_path.exists():
            return np.load(cache_path)
        # fallback: compute on the fly (cache miss)
        y = audio_io.load_window(row["window_path"])
        if self.feature_type == "mfcc":
            return feat_mod.mfcc_features(y)
        return feat_mod.logmel_features(y)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = self._load_feature(row)
        feat = feat_mod.normalize(feat)
        if self.augment:
            p = HAZARD_AUG_P if row["target_class"] in HAZARD_LABELS else AMBIENCE_AUG_P
            feat = aug_mod.augment_feature(feat, p=p)
        x = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0).float()  # (1, F, T)
        label = LABEL_TO_IDX[row["target_class"]]
        return x, label


def class_weights(manifest_csv, split="train"):
    df = pd.read_csv(manifest_csv)
    df = df[df["split"] == split]
    counts = df["target_class"].value_counts()
    from .labels import LABELS
    w = np.array([1.0 / max(counts.get(l, 1), 1) for l in LABELS], dtype=np.float32)
    w = w / w.sum() * len(LABELS)
    return torch.tensor(w, dtype=torch.float32)
