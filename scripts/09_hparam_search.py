"""Small Optuna search (lr / batch size / dropout [/ hidden_size for CRNN]) over
val macro-recall, short-epoch trials to bound wall-clock. Logs every trial, then
retrains the winning config to full epochs for the final checkpoint.

Usage: python 09_hparam_search.py --arch mfcc_cnn --n-trials 8 --trial-epochs 6 --final-epochs 25
"""
import argparse
import csv
from pathlib import Path

import optuna
import torch

from utils.dataset import SafeScapeDataset, class_weights
from utils.models import build_model
from utils.train_utils import train_model

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "processed" / "windows_manifest.csv"
ARCH_TO_FEATURE = {"mfcc_cnn": "mfcc", "logmel_crnn": "logmel", "transformer": "logmel"}


def objective_factory(arch, feature_type, train_ds, val_ds, device, trial_epochs, cw, log_rows):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        kwargs = {"dropout": dropout}
        if arch == "logmel_crnn":
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
            kwargs["hidden_size"] = hidden_size
        else:
            hidden_size = None

        model = build_model(arch, **kwargs).to(device)
        _, best_recall = train_model(
            model, train_ds, val_ds, device,
            epochs=trial_epochs, batch_size=batch_size, lr=lr, class_weights=cw,
            patience=trial_epochs,  # no early stop within a short trial
            num_workers=0,
        )
        log_rows.append({
            "trial": trial.number, "lr": lr, "batch_size": batch_size,
            "dropout": dropout, "hidden_size": hidden_size, "val_macro_recall": best_recall,
        })
        return best_recall

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mfcc_cnn", "logmel_crnn", "transformer"])
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--trial-epochs", type=int, default=6)
    ap.add_argument("--final-epochs", type=int, default=25)
    args = ap.parse_args()

    feature_type = ARCH_TO_FEATURE[args.arch]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SafeScapeDataset(MANIFEST, "train", feature_type=feature_type)
    val_ds = SafeScapeDataset(MANIFEST, "val", feature_type=feature_type)
    cw = class_weights(MANIFEST, split="train")

    log_rows = []
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(
        objective_factory(args.arch, feature_type, train_ds, val_ds, device, args.trial_epochs, cw, log_rows),
        n_trials=args.n_trials,
    )

    log_path = ROOT / "reports" / "hparam_search_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "trial", "lr", "batch_size", "dropout", "hidden_size", "val_macro_recall"])
        if write_header:
            w.writeheader()
        for row in log_rows:
            row = {"arch": args.arch, **row}
            w.writerow(row)

    best = study.best_params
    print(f"BEST for {args.arch}: {best} -> val_macro_recall={study.best_value:.4f}")

    # retrain winning config to full epochs
    kwargs = {"dropout": best["dropout"]}
    if args.arch == "logmel_crnn":
        kwargs["hidden_size"] = best["hidden_size"]
    model = build_model(args.arch, **kwargs).to(device)
    ckpt_path = ROOT / "models" / "checkpoints" / f"{args.arch}_best.pt"
    log_csv_path = ROOT / "reports" / f"{args.arch}_train_log.csv"
    model, best_recall = train_model(
        model, train_ds, val_ds, device,
        epochs=args.final_epochs, batch_size=best["batch_size"], lr=best["lr"], class_weights=cw,
        log_csv_path=str(log_csv_path), ckpt_path=str(ckpt_path), num_workers=0,
    )
    print(f"FINAL {args.arch}: val_macro_recall={best_recall:.4f} ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
