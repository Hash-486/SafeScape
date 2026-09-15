"""Train one architecture (mfcc_cnn or logmel_crnn) on the windowed/split manifest.

Usage: python 06_train.py --arch mfcc_cnn --epochs 25 --lr 1e-3 --batch-size 32 --dropout 0.3
"""
import argparse
from pathlib import Path

import torch

from utils.dataset import SafeScapeDataset, class_weights
from utils.models import build_model
from utils.train_utils import train_model

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "processed" / "windows_manifest.csv"

ARCH_TO_FEATURE = {"mfcc_cnn": "mfcc", "logmel_crnn": "logmel", "transformer": "logmel"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mfcc_cnn", "logmel_crnn", "transformer"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-size", type=int, default=64)  # only used by logmel_crnn
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    feature_type = ARCH_TO_FEATURE[args.arch]
    ckpt_path = args.ckpt or str(ROOT / "models" / "checkpoints" / f"{args.arch}_best.pt")
    log_csv_path = args.log_csv or str(ROOT / "reports" / f"{args.arch}_train_log.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} arch={args.arch} feature={feature_type}")

    train_ds = SafeScapeDataset(MANIFEST, "train", feature_type=feature_type)
    val_ds = SafeScapeDataset(MANIFEST, "val", feature_type=feature_type)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    kwargs = {"dropout": args.dropout}
    if args.arch == "logmel_crnn":
        kwargs["hidden_size"] = args.hidden_size
    model = build_model(args.arch, **kwargs).to(device)

    cw = class_weights(MANIFEST, split="train")

    model, best_recall = train_model(
        model, train_ds, val_ds, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, class_weights=cw,
        log_csv_path=log_csv_path, ckpt_path=ckpt_path, patience=args.patience,
        num_workers=args.num_workers,
    )
    print(f"DONE arch={args.arch} best_val_macro_recall={best_recall:.4f} ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
