"""Post-hoc per-class logit calibration for the champion model — no retraining.

Training used a class-weighted loss to keep hazard recall high, which pushes the decision
boundary toward the rare classes (glass_break/horn_skid) and costs precision there. This
fits one additive bias per class on the VALIDATION split (test stays untouched) by
coordinate ascent on macro-F1, and writes it to models/exported/calibration.json for the
server to apply before softmax.

Usage: python 12_calibrate_logits.py [--arch logmel_crnn] [--ckpt <path>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import SafeScapeDataset
from utils.models import build_model
from utils.labels import LABELS, HAZARD_LABELS
from utils.metrics import full_report

from utils.paths import ROOT, WINDOWS_MANIFEST as MANIFEST

EXPORTED_DIR = ROOT / "models" / "exported"
ARCH_TO_FEATURE = {"mfcc_cnn": "mfcc", "logmel_crnn": "logmel", "transformer": "logmel"}


def collect_logits(model, loader, device):
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            logits.append(out.cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(logits), np.concatenate(labels)


def score_bias(y_true, logits, bias, floors=None):
    """macro-F1, or -inf if any hazard class falls below its recall floor.

    Unconstrained macro-F1 buys precision on the rare classes by giving up recall on them,
    which is backwards for a safety system: the proposal's RQ4 states a missed hazard costs
    far more than a false alarm. Floors keep that property while taking free precision.
    """
    report, _ = full_report(y_true, (logits + bias).argmax(1))
    if floors:
        for label, floor in floors.items():
            if report[label]["recall"] < floor:
                return -np.inf
    return report["macro avg"]["f1-score"]


def search_bias(logits, y_true, floors=None, lo=-2.0, hi=2.0, step=0.1, passes=3):
    bias = np.zeros(len(LABELS), dtype=np.float64)
    best = score_bias(y_true, logits, bias, floors)
    print(f"uncalibrated val macro-F1: {best:.4f}")
    for p in range(passes):
        improved = False
        for c in range(len(LABELS)):
            trial = bias.copy()
            for v in np.arange(lo, hi + step / 2, step):
                trial[c] = v
                score = score_bias(y_true, logits, trial, floors)
                if score > best:
                    best, bias, improved = score, trial.copy(), True
        print(f"pass {p + 1}: val macro-F1 {best:.4f} bias {np.round(bias, 2).tolist()}")
        if not improved:
            break
    return bias, best


def summarize(y_true, logits, bias, title):
    report, _ = full_report(y_true, (logits + bias).argmax(1))
    print(f"\n--- {title} ---")
    print(f"accuracy={report['accuracy']:.4f} macro_f1={report['macro avg']['f1-score']:.4f} "
          f"macro_recall={report['macro avg']['recall']:.4f}")
    for label in LABELS:
        r = report[label]
        print(f"  {label:14s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="logmel_crnn", choices=list(ARCH_TO_FEATURE))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(EXPORTED_DIR / "calibration.json"))
    ap.add_argument("--objective", default="safety", choices=["safety", "macro_f1"],
                    help="safety: maximize macro-F1 subject to per-hazard recall floors")
    ap.add_argument("--recall-tolerance", type=float, default=0.02,
                    help="how far below uncalibrated recall each hazard class may fall")
    args = ap.parse_args()

    with open(EXPORTED_DIR / "preprocess_config.json") as f:
        hidden_size = json.load(f).get("hidden_size", 64)

    ckpt = args.ckpt or str(ROOT / "models" / "checkpoints" / f"{args.arch}_best.pt")
    device = torch.device("cpu")

    ds = SafeScapeDataset(MANIFEST, "val", feature_type=ARCH_TO_FEATURE[args.arch])
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    model_kwargs = {"hidden_size": hidden_size} if args.arch == "logmel_crnn" else {}
    model = build_model(args.arch, **model_kwargs).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    print(f"collecting logits over {len(ds)} val windows...")
    logits, y_true = collect_logits(model, loader, device)

    base = summarize(y_true, logits, np.zeros(len(LABELS)), "VAL uncalibrated")

    floors = None
    if args.objective == "safety":
        floors = {l: base[l]["recall"] - args.recall_tolerance for l in HAZARD_LABELS}
        print(f"\nrecall floors: { {k: round(v, 3) for k, v in floors.items()} }")

    bias, best = search_bias(logits, y_true, floors)
    summarize(y_true, logits, bias, f"VAL calibrated ({args.objective})")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump({"labels": LABELS, "bias": bias.tolist(), "fitted_on": "val",
                   "objective": args.objective, "recall_tolerance": args.recall_tolerance,
                   "arch": args.arch}, f, indent=2)
    print(f"\nwrote {out_path}  bias={np.round(bias, 2).tolist()}")
    print("re-run 10_evaluate.py --calibration <path> for held-out TEST numbers")


if __name__ == "__main__":
    main()
