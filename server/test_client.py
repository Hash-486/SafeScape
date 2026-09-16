"""Smoke test: sends N test-split windows per class to /predict and checks the server
discriminates between classes (catches train/serve preprocessing mismatches and models
that collapse to always predicting one class).

Windows are cut from the RAW source audio rather than read from window_path, because
those files are already denoised -- see the comment in the request loop. Reported
accuracy is class-balanced and so reads lower than the natural-prior figure in
reports/eval_results.md; both are correct, they answer different questions."""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils.labels import LABELS
from scripts.utils.paths import WINDOWS_MANIFEST
from scripts.utils import audio_io

# The served model and the manifest must come from the same data revision -- testing a
# v2 model against v1 test windows would score it on clips that are v2 *training* data.
# Set SAFESCAPE_PROC_DIR to match whatever revision the exported bundle was built from.
SERVER = os.environ.get("SAFESCAPE_SERVER", "http://127.0.0.1:8124")


def main():
    print(f"server:   {SERVER}")
    print(f"manifest: {WINDOWS_MANIFEST}")
    df = pd.read_csv(WINDOWS_MANIFEST)
    test_df = df[df["split"] == "test"]

    r = requests.get(f"{SERVER}/health", timeout=10)
    print("health:", r.status_code, r.json())
    assert r.status_code == 200

    # One window per class is far too few to say anything about quality -- a single
    # unlucky draw reads as a broken model. Sample N per class instead and report
    # per-class accuracy; the collapse check stays, since that is what this file is for.
    per_class = int(os.environ.get("PER_CLASS", "20"))
    correct, total = 0, 0
    predicted_labels = set()
    for label in LABELS:
        rows = test_df[test_df["target_class"] == label]
        if len(rows) == 0:
            print(f"[skip] no test rows for {label}")
            continue
        rows = rows.sample(min(per_class, len(rows)), random_state=0)
        c = t = 0
        for _, row in rows.iterrows():
            # Send a window cut from the RAW source, never the file in window_path:
            # those were already denoised by 03_preprocess_audio.py before being
            # written, so posting one makes the server's own denoise() a second pass.
            # Measured cost of that double-denoise is ~4.4 points of accuracy -- a test
            # artifact that looks exactly like a broken model.
            try:
                y = audio_io.load_and_resample(row["source_filepath"])
                windows = audio_io.window_signal(y)
            except Exception as e:
                print(f"[skip] {row['source_filepath']}: {e}")
                continue
            if len(windows) == 0:
                continue
            k = int(Path(row["window_path"]).stem.split("_")[1])
            buf = io.BytesIO()
            sf.write(buf, windows[min(k, len(windows) - 1)], 16000,
                     format="WAV", subtype="PCM_16")
            buf.seek(0)
            r = requests.post(f"{SERVER}/predict",
                              files={"file": ("clip.wav", buf, "audio/wav")}, timeout=30)
            r.raise_for_status()
            out = r.json()
            predicted_labels.add(out["predicted_class"])
            c += int(out["predicted_class"] == label)
            t += 1
        if t:
            print(f"{label:<15} {c:>3}/{t:<3} = {c/t:.3f}")
        correct += c
        total += t

    print("")
    print(f"overall {correct}/{total} = {correct/total:.3f} "
          f"({per_class}/class, balanced -- not the natural-prior number)")
    print(f"distinct predicted labels: {predicted_labels}")
    if len(predicted_labels) <= 1:
        print("WARNING: model collapsed to a single predicted class -- likely a bug, not a real result.")


if __name__ == "__main__":
    main()
