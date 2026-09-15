"""Dynamic post-training quantization (qint8) of a trained checkpoint, with a
before/after size comparison, and exporting the chosen champion model bundle
(weights + label_map.json + preprocess_config.json) for the FastAPI server.

Usage:
  python 11_quantize_export.py quantize --arch logmel_crnn
  python 11_quantize_export.py export --arch logmel_crnn --quantized
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from utils.models import build_model
from utils.labels import LABEL_TO_IDX

ROOT = Path(__file__).resolve().parent.parent
ARCH_TO_FEATURE = {"mfcc_cnn": "mfcc", "logmel_crnn": "logmel", "transformer": "logmel"}
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000

QUANT_MODULES = {nn.Linear, nn.GRU}


def model_kwargs(arch, hidden_size):
    return {"hidden_size": hidden_size} if arch == "logmel_crnn" else {}


def build_quantized(arch, hidden_size):
    model = build_model(arch, **model_kwargs(arch, hidden_size))
    model.eval()
    return torch.quantization.quantize_dynamic(model, QUANT_MODULES, dtype=torch.qint8)


def cmd_quantize(args):
    fp32_ckpt = Path(args.ckpt) if args.ckpt else ROOT / "models" / "checkpoints" / f"{args.arch}_best.pt"
    model = build_model(args.arch, **model_kwargs(args.arch, args.hidden_size))
    model.load_state_dict(torch.load(fp32_ckpt, map_location="cpu", weights_only=True))
    model.eval()

    qmodel = torch.quantization.quantize_dynamic(model, QUANT_MODULES, dtype=torch.qint8)
    out_path = ROOT / "models" / "quantized" / f"{args.arch}_quant.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out_path)

    fp32_size = fp32_ckpt.stat().st_size / 1024
    quant_size = out_path.stat().st_size / 1024
    print(f"{args.arch}: fp32={fp32_size:.1f} KB -> quantized={quant_size:.1f} KB "
          f"({(1 - quant_size/fp32_size)*100:.1f}% smaller)")
    print(f"saved: {out_path}")


def cmd_export(args):
    if args.quantized:
        weights_path = ROOT / "models" / "quantized" / f"{args.arch}_quant.pt"
        model = build_quantized(args.arch, args.hidden_size)
    else:
        weights_path = Path(args.ckpt) if args.ckpt else ROOT / "models" / "checkpoints" / f"{args.arch}_best.pt"
        model = build_model(args.arch, **model_kwargs(args.arch, args.hidden_size))
        model.eval()

    # dynamic-quantized GRU state has packed torch.ScriptObject params incompatible
    # with weights_only=True; safe here since these are checkpoints we produced ourselves.
    state = torch.load(weights_path, map_location="cpu", weights_only=not args.quantized)
    model.load_state_dict(state)

    export_dir = ROOT / "models" / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), export_dir / "best_model.pt")

    with open(export_dir / "label_map.json", "w") as f:
        json.dump(LABEL_TO_IDX, f, indent=2)

    cfg = {
        "model_name": args.arch,
        "feature_type": ARCH_TO_FEATURE[args.arch],
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "quantized": bool(args.quantized),
        "hidden_size": args.hidden_size,
    }
    with open(export_dir / "preprocess_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"exported champion model ({args.arch}, quantized={args.quantized}) to {export_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("quantize")
    p1.add_argument("--arch", required=True, choices=["mfcc_cnn", "logmel_crnn", "transformer"])
    p1.add_argument("--hidden-size", type=int, default=64, help="only used by logmel_crnn")
    p1.add_argument("--ckpt", default=None, help="fp32 checkpoint to quantize")
    p1.set_defaults(func=cmd_quantize)

    p2 = sub.add_parser("export")
    p2.add_argument("--arch", required=True, choices=["mfcc_cnn", "logmel_crnn", "transformer"])
    p2.add_argument("--quantized", action="store_true")
    p2.add_argument("--hidden-size", type=int, default=64, help="only used by logmel_crnn")
    p2.add_argument("--ckpt", default=None, help="fp32 checkpoint to export (ignored with --quantized)")
    p2.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
