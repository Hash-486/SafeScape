"""Resample/mono/denoise each source file from manifest.csv, window into 1.0s/50%-overlap
clips, and write windows_manifest.csv with train/val/test assigned at the SOURCE-FILE level
(stratified per class) so that windows from the same source clip never leak across splits.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import audio_io

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_IN = ROOT / "data" / "processed" / "manifest.csv"
WINDOWS_DIR = ROOT / "data" / "processed" / "windows"
MANIFEST_OUT = ROOT / "data" / "processed" / "windows_manifest.csv"

RNG_SEED = 42


def stratified_split(df, seed=RNG_SEED, train_frac=0.70, val_frac=0.15):
    rng = np.random.RandomState(seed)
    splits = []
    for cls, group in df.groupby("target_class"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        assign = {}
        for i in idx[:n_train]:
            assign[i] = "train"
        for i in idx[n_train:n_train + n_val]:
            assign[i] = "val"
        for i in idx[n_train + n_val:]:
            assign[i] = "test"
        splits.append(pd.Series(assign))
    return pd.concat(splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoise", action="store_true", default=True)
    ap.add_argument("--no-denoise", dest="denoise", action="store_false")
    ap.add_argument("--limit", type=int, default=None, help="debug: only process first N source files")
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST_IN)
    if args.limit:
        df = df.groupby("target_class").head(args.limit).reset_index(drop=True)

    print("source-file class distribution:\n", df["target_class"].value_counts())

    split_series = stratified_split(df)
    df["split"] = df.index.map(split_series)

    assert df["split"].isna().sum() == 0

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="preprocessing"):
        src = row["filepath"]
        cls = row["target_class"]
        split = row["split"]
        try:
            y = audio_io.load_and_resample(src)
        except Exception as e:
            print(f"[skip] failed to load {src}: {e}")
            continue
        if args.denoise:
            y = audio_io.denoise(y)
        windows = audio_io.window_signal(y)
        out_dir = WINDOWS_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for w_i, w in enumerate(windows):
            out_path = out_dir / f"{idx}_{w_i}.wav"
            audio_io.save_window(w, out_path)
            rows.append({
                "window_path": str(out_path),
                "source_filepath": src,
                "target_class": cls,
                "split": split,
            })

    out_df = pd.DataFrame(rows)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(MANIFEST_OUT, index=False)

    print("\nwindow counts by class/split:")
    print(out_df.groupby(["target_class", "split"]).size().unstack(fill_value=0))

    # correctness check: no source file appears in more than one split
    leak_check = out_df.groupby("source_filepath")["split"].nunique()
    assert (leak_check == 1).all(), "LEAK: a source file's windows span multiple splits"
    print(f"\nOK: no source-file leakage across splits. wrote {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
