"""Generate presentation diagrams: end-to-end system pipeline + per-architecture detail.

Shapes and layer configs are transcribed from utils/models.py; the CRNN uses hidden_size
from models/exported/preprocess_config.json (128), not the class default.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a2e"
DIM = "#5a5a72"
PALETTE = {
    "input": "#e8eef7", "prep": "#dce9f5", "feat": "#d3e5ef",
    "model": "#f7e6e0", "out": "#e3f0e3", "serve": "#efe8f5",
}
EDGE = {
    "input": "#6b8cae", "prep": "#5b87b8", "feat": "#4a7fa0",
    "model": "#c08070", "out": "#7aa87a", "serve": "#9280b0",
}


def box(ax, xy, w, h, label, kind, sub=None, fs=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 fc=PALETTE[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x + w / 2, y + h / 2 + (0.075 if sub else 0), label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.115, sub, ha="center", va="center",
                fontsize=fs - 2.3, color=DIM)


def arrow(ax, p1, p2, label=None, fs=7.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color=DIM, shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.085, label, ha="center", va="bottom", fontsize=fs,
                color=DIM, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(-0.25, 10.25); ax.set_ylim(0, h / w * 10)
    ax.axis("off")
    return fig, ax


def system_diagram():
    fig, ax = canvas(13, 5.0)
    y = 2.45
    bw, bh = 1.42, 0.86
    xs = [0.18, 1.82, 3.46, 5.10, 6.74, 8.38]
    specs = [
        ("Microphone", "input", "1 s rolling\nwindow, 50% hop"),
        ("Preprocess", "prep", "16 kHz mono\ndenoise"),
        ("Features", "feat", "log-mel 64×101\n(or MFCC 40×101)"),
        ("CRNN", "model", "CNN + BiGRU\n516 KB int8"),
        ("Calibrate", "out", "per-class bias\nsoftmax → argmax"),
        ("Alert / UI", "serve", "FastAPI +\nmobile web app"),
    ]
    for x, (label, kind, sub) in zip(xs, specs):
        box(ax, (x, y), bw, bh, label, kind, sub)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + bw, y + bh / 2), (xs[i + 1], y + bh / 2))

    ax.text(5.0, 3.92, "SafeScape — end-to-end on-device pipeline",
            ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(5.0, 3.60, "audio never leaves the device · 5 classes: distress_call, glass_break, "
                       "horn_skid, alarm, ambience",
            ha="center", fontsize=9, color=DIM)

    ax.annotate("", xy=(0.9, 2.30), xytext=(9.1, 2.30),
                arrowprops=dict(arrowstyle="<->", color="#b0b0c0", lw=1.1))
    ax.text(5.0, 2.06, "measured end-to-end latency ≈ 6.6 ms / window  (≪ 1 s window → real-time)",
            ha="center", fontsize=8.6, color=DIM, style="italic")

    ax.text(5.0, 1.50, "Training-time pipeline (offline)", ha="center", fontsize=10,
            fontweight="bold", color=INK)
    tr = ["ESC-50 / RAVDESS /\nKaggle scream sets", "manifest +\nclass capping",
          "window + split\n(source-disjoint)", "train 3 architectures",
          "Optuna search", "quantize + export"]
    tw, tx0 = 1.46, 0.18
    for i, t in enumerate(tr):
        x = tx0 + i * 1.64
        ax.add_patch(FancyBboxPatch((x, 0.62), tw, 0.62,
                                     boxstyle="round,pad=0.02,rounding_size=0.05",
                                     fc="#f4f4f8", ec="#c5c5d5", lw=1.1))
        ax.text(x + tw / 2, 0.93, t, ha="center", va="center", fontsize=7.4, color=INK)
        if i < len(tr) - 1:
            arrow(ax, (x + tw, 0.93), (x + 1.64, 0.93))

    fig.tight_layout()
    p = OUT / "system_architecture.png"
    fig.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def crnn_diagram(hidden):
    fig, ax = canvas(13, 4.0)
    y, bw, bh = 1.52, 1.50, 0.92
    xs = [0.05 + i * 1.69 for i in range(6)]
    specs = [
        ("log-mel input", "input", "(1, 64, 101)"),
        ("Conv block 1", "prep", "Conv3×3→16\nBN, ReLU, pool(2,1)"),
        ("Conv block 2", "prep", "Conv3×3→32\nBN, ReLU, pool(2,1)"),
        ("reshape", "feat", "(T=101, 32×16)\n= (101, 512)"),
        (f"BiGRU", "model", f"hidden={hidden}\n→ (101, {hidden * 2})"),
        ("mean-pool\n+ FC", "out", f"({hidden * 2}) → 5 logits"),
    ]
    for x, (label, kind, sub) in zip(xs, specs):
        box(ax, (x, y), bw, bh, label, kind, sub, fs=9.4)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + bw, y + bh / 2), (xs[i + 1], y + bh / 2))

    ax.text(5.0, 2.92, f"Champion architecture — log-mel CRNN  (499,237 params · 89.8% test accuracy)",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(5.0, 2.66, "pooling is frequency-only (2,1) so the full 101-frame time axis reaches the GRU",
            ha="center", fontsize=9, color=DIM, style="italic")
    ax.text(5.0, 1.22, "Why this wins: convolutions learn local time–frequency texture (a scream's harmonic\n"
                       "structure, glass's broadband transient); the BiGRU then models how that texture evolves\n"
                       "across the second — which a frame-independent CNN cannot do.",
            ha="center", va="top", fontsize=8.8, color=INK)
    fig.tight_layout()
    p = OUT / "crnn_architecture.png"
    fig.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def compare_diagram():
    fig, ax = canvas(13, 4.6)
    rows = [
        ("MFCC-CNN", "#c08070", ["MFCC (40×101)", "4× Conv+BN+ReLU\n16→32→64→64", "AdaptiveAvgPool", "FC → 5"],
         "60,901 params · 248.6 KB · 0.44 ms · 66.6%"),
        ("log-mel CRNN", "#7aa87a", ["log-mel (64×101)", "2× Conv+BN+ReLU\n16→32", "BiGRU (128)", "mean-pool, FC → 5"],
         "499,237 params · 516.4 KB int8 · 11.6 ms · 89.8%"),
        ("Tiny Transformer", "#9280b0", ["log-mel (64×101)", "Linear proj → 64\n+ CLS + pos-embed",
                                          "2× Encoder layer\n4 heads", "CLS → FC → 5"],
         "84,293 params · 339.0 KB · 0.48 ms · 73.5%"),
    ]
    bw, bh, gap = 1.76, 0.72, 1.90
    for r, (name, color, stages, stat) in enumerate(rows):
        yy = 2.62 - r * 1.06
        ax.text(0.1, yy + bh / 2, name, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=color)
        for i, s in enumerate(stages):
            x = 2.20 + i * gap
            ax.add_patch(FancyBboxPatch((x, yy), bw, bh,
                                         boxstyle="round,pad=0.02,rounding_size=0.05",
                                         fc="#f6f6fa", ec=color, lw=1.4))
            ax.text(x + bw / 2, yy + bh / 2, s, ha="center", va="center", fontsize=7.8, color=INK)
            if i < len(stages) - 1:
                arrow(ax, (x + bw, yy + bh / 2), (x + gap, yy + bh / 2))
        ax.text(0.1, yy - 0.16, stat, ha="left", va="center", fontsize=7.4, color=DIM)

    ax.text(5.0, 3.86, "Three architectures, one dataset and split",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(5.0, 3.58, "each team member owns one branch · identical 1 s windows, identical train/val/test split",
            ha="center", fontsize=9, color=DIM)
    fig.tight_layout()
    p = OUT / "architecture_comparison.png"
    fig.savefig(p, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    with open(ROOT / "models" / "exported" / "preprocess_config.json") as f:
        hidden = json.load(f).get("hidden_size", 64)
    for p in (system_diagram(), crnn_diagram(hidden), compare_diagram()):
        print("wrote", p)
