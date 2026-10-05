"""Mark test windows whose source audio also sits in train or val under another path.

The Kaggle scream dataset ships the same recordings in two folders, so the
source-file-level split put byte-identical copies on both sides of it: 230 of the
3220 v2 test windows, every one of them distress_call. Re-splitting means retraining
all three models, so instead those windows are relabelled split=test_dup and every
headline number is computed on what is left.

Writes windows_manifest_dedup.csv next to the windows manifest; nothing is moved.

Usage (from scripts/): SAFESCAPE_PROC_DIR=data/processed_v2 python 18_flag_duplicate_sources.py
"""
import hashlib

import pandas as pd

from utils.paths import WINDOWS_MANIFEST

OUT = WINDOWS_MANIFEST.with_name("windows_manifest_dedup.csv")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    df = pd.read_csv(WINDOWS_MANIFEST)
    digests = {p: md5(p) for p in df.source_filepath.unique()}
    h = df.source_filepath.map(digests)
    seen = set(h[df.split.isin(["train", "val"])])
    dup = (df.split == "test") & h.isin(seen)
    df.loc[dup, "split"] = "test_dup"
    df.to_csv(OUT, index=False)
    print(f"{dup.sum()} of {dup.sum() + (df.split == 'test').sum()} test windows duplicate train/val audio")
    print(df[dup].target_class.value_counts().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
