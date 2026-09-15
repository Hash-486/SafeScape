"""Precompute MFCC and log-mel features for every windowed clip to .npy cache.

This exists for SPEED: on-the-fly librosa feature extraction (~40-100ms/sample)
made a full epoch take ~20 minutes single-threaded, and Windows DataLoader
multiprocessing (spawn-based re-import of torch/librosa/numba per worker) proved
unreliable/slow to parallelize reliably under deadline pressure. Precomputing once
(~20-25 min total, one-time cost) turns every subsequent training run into fast
in-memory-cache-speed epochs. Train-time augmentation is correspondingly moved to
operate directly on the cached feature arrays (see utils/augment.py spec_augment /
add_feature_noise) instead of the waveform domain.
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import audio_io, features as feat_mod

from utils.paths import ROOT, WINDOWS_MANIFEST as MANIFEST, FEATURES_DIR


def feature_path(window_path: str, feature_type: str) -> Path:
    p = Path(window_path)
    # .../windows/<class>/<name>.wav -> .../features/<feature_type>/<class>/<name>.npy
    cls = p.parent.name
    return FEATURES_DIR / feature_type / cls / (p.stem + ".npy")


def main():
    import pandas as pd
    df = pd.read_csv(MANIFEST)

    for feature_type, fn in [("mfcc", feat_mod.mfcc_features), ("logmel", feat_mod.logmel_features)]:
        print(f"extracting {feature_type} for {len(df)} windows...")
        n_ok, n_skip = 0, 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=feature_type):
            out_path = feature_path(row["window_path"], feature_type)
            if out_path.exists():
                n_ok += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                y = audio_io.load_window(row["window_path"])
                feat = fn(y)
                np.save(out_path, feat)
                n_ok += 1
            except Exception as e:
                print(f"[skip] {row['window_path']}: {e}")
                n_skip += 1
        print(f"{feature_type}: {n_ok} ok, {n_skip} skipped")

    # sanity check: shape + finiteness on a random sample
    import random
    for feature_type in ["mfcc", "logmel"]:
        sample = df.sample(n=min(100, len(df)), random_state=0)
        shapes = set()
        n_bad = 0
        for _, row in sample.iterrows():
            p = feature_path(row["window_path"], feature_type)
            if not p.exists():
                continue
            arr = np.load(p)
            shapes.add(arr.shape)
            if not np.isfinite(arr).all():
                n_bad += 1
        print(f"{feature_type} sanity: shapes seen={shapes}, non-finite arrays={n_bad}")


if __name__ == "__main__":
    main()
