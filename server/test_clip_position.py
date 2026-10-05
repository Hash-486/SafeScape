"""Regression test: a hazard must be detected wherever it sits inside the clip.

The browser records 2s clips but the model's window is 1s, so serving has to decide
which second to look at. Taking only the last one (the original behaviour) discarded
half of every clip: a glass_break in the first half came back as distress_call 72%.

Runs against the Predictor directly rather than over HTTP -- the bug is in waveform
prep, and keeping ffmpeg and the network out of it makes a failure unambiguous."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.inference import Predictor

SR = 16000
CLIPS = ROOT / "demo_clips"


def main():
    p = Predictor()
    sil = np.zeros(SR, dtype=np.float32)
    failures = []

    print(f"{'clip':<15}{'position':<14}{'predicted':<16}{'conf':>7}")
    for wav in sorted(CLIPS.glob("*.wav")):
        expected = wav.stem
        y, sr = sf.read(wav, dtype="float32")
        assert sr == SR, f"{wav.name} is {sr} Hz, expected {SR}"

        for position, clip in (("first half", np.concatenate([y, sil])),
                               ("second half", np.concatenate([sil, y]))):
            out = p.predict(clip.copy(), SR)
            got = out["predicted_class"]
            print(f"{expected:<15}{position:<14}{got:<16}{out['confidence'] * 100:>6.1f}%")
            if got != expected:
                failures.append(f"{expected} in {position} -> {got}")

    print("")
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print(f"  {f}")
        return 1
    print("PASS -- position inside the clip does not change the verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
