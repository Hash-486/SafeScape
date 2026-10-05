"""Every exported model loads and answers on the held-out UI test clips, and on the
inputs a live demo will throw at it: silence and a sub-second clip.

Runs the Predictors directly, like test_clip_position.py, so a failure points at a
model bundle rather than at ffmpeg or the network."""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.inference import Predictor, MODELS, EXPORTED_DIR

SR = 16000
CLIPS = ROOT / "demo_clips" / "test"


def main():
    preds = {k: Predictor(EXPORTED_DIR / k) for k in MODELS}
    rows = list(csv.DictReader(open(CLIPS / "manifest.csv")))
    correct = {k: 0 for k in MODELS}

    print(f"{'clip':<20}{'truth':<15}" + "".join(f"{k:<16}" for k in MODELS))
    for r in rows:
        y, sr = sf.read(CLIPS / r["file"], dtype="float32")
        got = {k: p.predict(y.copy(), sr)["predicted_class"] for k, p in preds.items()}
        for k in MODELS:
            correct[k] += got[k] == r["label"]
        print(f"{r['file']:<20}{r['label']:<15}" + "".join(f"{got[k]:<16}" for k in MODELS))
    print("\naccuracy on UI test clips: " +
          ", ".join(f"{k} {correct[k]}/{len(rows)}" for k in MODELS))

    short, _ = sf.read(CLIPS / rows[0]["file"], dtype="float32")
    for k, p in preds.items():
        assert p.predict(np.zeros(2 * SR, dtype=np.float32), SR)["predicted_class"] == "ambience", k
        p.predict(short[: SR // 3].copy(), SR)
    print("edge inputs OK (silence -> ambience, 0.33s clip answered)")


if __name__ == "__main__":
    main()
