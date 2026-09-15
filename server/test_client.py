"""Smoke test: sends one known test-split wav per class to /predict and checks the
server correctly discriminates between classes (catches train/serve preprocessing
mismatches and models that collapse to always predicting one class)."""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils.labels import LABELS

SERVER = "http://127.0.0.1:8000"


def main():
    manifest_path = ROOT / "data" / "processed" / "windows_manifest.csv"
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"]

    r = requests.get(f"{SERVER}/health", timeout=10)
    print("health:", r.status_code, r.json())
    assert r.status_code == 200

    correct, total = 0, 0
    predicted_labels = set()
    for label in LABELS:
        rows = test_df[test_df["target_class"] == label]
        if len(rows) == 0:
            print(f"[skip] no test rows for {label}")
            continue
        wav_path = rows.iloc[0]["window_path"]
        with open(wav_path, "rb") as f:
            r = requests.post(f"{SERVER}/predict", files={"file": ("clip.wav", f, "audio/wav")}, timeout=30)
        r.raise_for_status()
        out = r.json()
        predicted_labels.add(out["predicted_class"])
        match = "OK" if out["predicted_class"] == label else "MISS"
        print(f"[{match}] true={label:14s} pred={out['predicted_class']:14s} conf={out['confidence']:.3f}")
        total += 1
        correct += int(out["predicted_class"] == label)

    print(f"\n{correct}/{total} correct on one-sample-per-class smoke test")
    print(f"distinct predicted labels across test: {predicted_labels}")
    if len(predicted_labels) <= 1:
        print("WARNING: model collapsed to a single predicted class — likely a bug, not a real result.")


if __name__ == "__main__":
    main()
