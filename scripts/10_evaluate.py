"""Evaluate a trained checkpoint on the held-out TEST split: accuracy, per-class
precision/recall/F1, confusion matrix, CPU inference latency, model size, param count.
Appends a section to reports/eval_results.md.

Usage: python 10_evaluate.py --arch mfcc_cnn --ckpt models/checkpoints/mfcc_cnn_best.pt
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import SafeScapeDataset
from utils.models import build_model
from utils.labels import LABELS
from utils.metrics import full_report, report_to_markdown

from utils.paths import ROOT, WINDOWS_MANIFEST as MANIFEST
ARCH_TO_FEATURE = {"mfcc_cnn": "mfcc", "logmel_crnn": "logmel", "transformer": "logmel"}


def measure_latency_ms(model, sample_x, device, n=100):
    model.eval()
    x = sample_x.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mfcc_cnn", "logmel_crnn", "transformer"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hidden-size", type=int, default=64, help="only used by logmel_crnn")
    ap.add_argument("--tag", default=None, help="label for this run in the report, e.g. 'fp32' or 'quantized'")
    ap.add_argument("--cpu-only", action="store_true", help="force CPU (matches serving target)")
    ap.add_argument("--calibration", default=None,
                    help="path to calibration.json; applies its per-class logit bias before argmax")
    args = ap.parse_args()

    feature_type = ARCH_TO_FEATURE[args.arch]
    ckpt_path = args.ckpt or str(ROOT / "models" / "checkpoints" / f"{args.arch}_best.pt")
    tag = args.tag or args.arch

    device = torch.device("cpu") if args.cpu_only else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = SafeScapeDataset(MANIFEST, "test", feature_type=feature_type, augment=False)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model_kwargs = {"hidden_size": args.hidden_size} if args.arch == "logmel_crnn" else {}
    model = build_model(args.arch, **model_kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    bias = torch.zeros(len(LABELS), device=device)
    if args.calibration:
        import json
        with open(args.calibration) as f:
            bias = torch.tensor(json.load(f)["bias"], dtype=torch.float32, device=device)

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x) + bias
            all_true.extend(y.numpy().tolist())
            all_pred.extend(out.argmax(1).cpu().numpy().tolist())

    report, cm = full_report(all_true, all_pred)
    accuracy = report["accuracy"]

    sample_x, _ = test_ds[0]
    latency_ms = measure_latency_ms(model, sample_x, device)
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = Path(ckpt_path).stat().st_size

    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{tag} confusion matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig_path = ROOT / "reports" / "figures" / f"{tag}_confusion_matrix.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)

    md = report_to_markdown(report, title=f"{tag}")
    summary = (
        f"\n**accuracy:** {accuracy:.3f} | **distress_call recall:** {report['distress_call']['recall']:.3f} | "
        f"**params:** {n_params:,} | **size:** {size_bytes/1024:.1f} KB | **CPU latency:** {latency_ms:.2f} ms/clip\n"
        f"\n![confusion matrix]({fig_path.relative_to(ROOT)})\n"
    )

    out_path = ROOT / "reports" / "eval_results.md"
    with open(out_path, "a") as f:
        f.write(md + "\n" + summary + "\n---\n")

    print(f"accuracy={accuracy:.3f} macro_recall={report['macro avg']['recall']:.3f} "
          f"distress_recall={report['distress_call']['recall']:.3f} "
          f"params={n_params} size_kb={size_bytes/1024:.1f} latency_ms={latency_ms:.2f}")
    print(f"appended results to {out_path}")


if __name__ == "__main__":
    main()
