"""Cut a small, labelled set of clips from v2 TEST-split source files for exercising
the UI. The old demo_clips/ were copies of training audio, which says nothing about
how the models handle sound they have not seen.

Each clip is the 2s around the loudest second of its source, so the event itself is
in frame rather than whatever lead-in the recording happens to start with.

Run 18_flag_duplicate_sources.py first.

Usage (from scripts/): SAFESCAPE_PROC_DIR=data/processed_v2 python 17_make_ui_testset.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from utils import audio_io
from utils.labels import LABELS
from utils.paths import ROOT, WINDOWS_MANIFEST

# 18_flag_duplicate_sources.py's output: test windows whose audio is also in train/val
# are split=test_dup there, so sampling split=test never picks a byte-identical copy of training audio
DEDUP_MANIFEST = WINDOWS_MANIFEST.with_name("windows_manifest_dedup.csv")

PER_CLASS = 4
CLIP = 2 * audio_io.SAMPLE_RATE
OUT = ROOT / "demo_clips" / "test"


def loudest_clip(y):
    if len(y) <= CLIP:
        return np.pad(y, (0, CLIP - len(y)))
    sec = audio_io.SAMPLE_RATE
    step = sec // 4
    rms = [np.sqrt(np.mean(y[i:i + sec] ** 2)) for i in range(0, len(y) - sec + 1, step)]
    centre = int(np.argmax(rms)) * step + sec // 2
    start = min(max(0, centre - CLIP // 2), len(y) - CLIP)
    return y[start:start + CLIP]


def main():
    df = pd.read_csv(DEDUP_MANIFEST)
    test = df[df.split == "test"].drop_duplicates("source_filepath")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in LABELS:
        picks = test[test.target_class == label].sample(PER_CLASS, random_state=7)
        for i, src in enumerate(picks.source_filepath):
            name = f"{label}_{i + 1}.wav"
            audio_io.save_window(loudest_clip(audio_io.load_and_resample(src)), OUT / name)
            rows.append({"file": name, "label": label, "source": Path(src).name})
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} clips to {OUT}")


if __name__ == "__main__":
    main()
