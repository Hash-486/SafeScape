"""Shared training loop used by both architectures."""
import time
import csv
import torch
from torch.utils.data import DataLoader

from .metrics import macro_recall


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, all_true, all_pred = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(out.argmax(1).cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    recall = macro_recall(all_true, all_pred)
    return avg_loss, recall


def train_model(model, train_ds, val_ds, device, epochs=25, batch_size=32, lr=1e-3,
                 dropout_unused=None, weight_decay=1e-4, class_weights=None,
                 log_csv_path=None, ckpt_path=None, patience=6, num_workers=0):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=(device.type == "cuda"),
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_recall, best_state, no_improve = -1.0, None, 0
    rows = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_recall = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_recall = run_epoch(model, val_loader, device, criterion, optimizer=None)
        dt = time.time() - t0
        rows.append([epoch, train_loss, train_recall, val_loss, val_recall, dt])
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} recall {train_recall:.3f} | "
              f"val_loss {val_loss:.4f} recall {val_recall:.3f} | {dt:.1f}s")

        if val_recall > best_recall:
            best_recall = val_recall
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch} (best val_recall={best_recall:.3f})")
                break

    if log_csv_path:
        with open(log_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_recall", "val_loss", "val_recall", "seconds"])
            w.writerows(rows)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_recall
