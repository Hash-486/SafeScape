"""Stretch goal: train the tiny transformer via knowledge distillation from the
already-trained log-mel CRNN (teacher). Only run this after mfcc_cnn and logmel_crnn
are both trained and evaluated.

Usage: python 08_train_distilled_transformer.py --epochs 25 --lr 1e-3 --alpha 0.5 --temperature 4.0
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.dataset import SafeScapeDataset, class_weights
from utils.models import build_model
from utils.metrics import macro_recall

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "processed" / "windows_manifest.csv"
TEACHER_CKPT = ROOT / "models" / "checkpoints" / "logmel_crnn_best.pt"


def distill_loss(student_logits, teacher_logits, targets, alpha, T, class_weights=None):
    hard = F.cross_entropy(student_logits, targets, weight=class_weights)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on hard-label CE vs soft KD loss")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--teacher-hidden-size", type=int, default=128, help="must match the trained CRNN's winning hparam config")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert TEACHER_CKPT.exists(), f"train logmel_crnn first — missing teacher checkpoint {TEACHER_CKPT}"

    teacher = build_model("logmel_crnn", hidden_size=args.teacher_hidden_size).to(device)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_model("transformer").to(device)

    train_ds = SafeScapeDataset(MANIFEST, "train", feature_type="logmel")
    val_ds = SafeScapeDataset(MANIFEST, "val", feature_type="logmel")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    cw = class_weights(MANIFEST, split="train").to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    best_recall, best_state, no_improve = -1.0, None, 0
    log_rows = []
    ckpt_path = ROOT / "models" / "checkpoints" / "transformer_best.pt"
    log_csv_path = ROOT / "reports" / "transformer_train_log.csv"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = distill_loss(student_logits, teacher_logits, y, args.alpha, args.temperature, cw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        student.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = student(x)
                all_true.extend(y.numpy().tolist())
                all_pred.extend(out.argmax(1).cpu().numpy().tolist())
        val_recall = macro_recall(all_true, all_pred)
        dt = time.time() - t0
        log_rows.append([epoch, train_loss, val_recall, dt])
        print(f"epoch {epoch:3d} | distill_train_loss {train_loss:.4f} | val_recall {val_recall:.3f} | {dt:.1f}s")

        if val_recall > best_recall:
            best_recall = val_recall
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_recall={best_recall:.3f})")
                break

    with open(log_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "distill_train_loss", "val_recall", "seconds"])
        w.writerows(log_rows)

    print(f"DONE transformer (distilled) best_val_macro_recall={best_recall:.4f} ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
