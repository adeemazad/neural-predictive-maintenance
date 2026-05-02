"""
train.py — Train the FaultCNN classifier and/or the RUL LSTM regressor.

Usage:
  python train.py --model cnn     # fault classification
  python train.py --model rul     # remaining useful life
  python train.py --model both    # sequential

Experiments are tracked with MLflow (run `mlflow ui` to view results).
Checkpoints saved to models/checkpoints/.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error
)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.loader import build_synthetic_dataset, train_val_test_split, FAULT_LABELS
from src.models.cnn_classifier import FaultCNN
from src.models.lstm_rul import RULPredictor, generate_rul_dataset, RULLoss
from src.features.dsp import extract_features_batch

CKPT_DIR = ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── CNN Training ──────────────────────────────────────────────────────────────

def train_cnn(
    n_samples: int    = 400,
    epochs:    int    = 30,
    batch_size: int   = 64,
    lr:        float  = 1e-3,
    snr_db:    float  = 20.0,
    seed:      int    = 42,
):
    print("\n" + "="*60)
    print("  Training FaultCNN — Bearing Fault Classifier")
    print("="*60)

    device = get_device()
    print(f"  Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n  Generating synthetic dataset (n_per_class={n_samples}, SNR={snr_db}dB) …")
    X, y = build_synthetic_dataset(n_samples_per_class=n_samples, snr_db=snr_db, seed=seed)
    splits = train_val_test_split(X, y, seed=seed)

    def make_loader(split, shuffle):
        Xs, ys = splits[split]
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_dl = make_loader("train", shuffle=True)
    val_dl   = make_loader("val",   shuffle=False)
    test_dl  = make_loader("test",  shuffle=False)

    print(f"  Train: {len(splits['train'][0])}, "
          f"Val: {len(splits['val'][0])}, "
          f"Test: {len(splits['test'][0])}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FaultCNN(n_classes=4, window_size=2048).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        scheduler.step()

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                v_loss    += criterion(logits, yb).item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        train_acc = t_correct / t_total
        val_acc   = v_correct / v_total
        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "cnn_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={t_loss/t_total:.4f}  acc={train_acc:.3f}  "
                  f"| val_loss={v_loss/v_total:.4f}  acc={val_acc:.3f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(CKPT_DIR / "cnn_best.pt", map_location=device))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            all_preds.extend(model(xb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    test_acc = accuracy_score(all_true, all_preds)
    print(f"\n  ✓ Test accuracy : {test_acc:.4f}  (best val: {best_val_acc:.4f})")
    print("\n" + classification_report(
        all_true, all_preds,
        target_names=[FAULT_LABELS[i] for i in range(4)]
    ))

    np.save(CKPT_DIR / "cnn_history.npy", history)
    print(f"  Checkpoint saved → {CKPT_DIR}/cnn_best.pt")
    return model, history, (all_true, all_preds)


# ── LSTM Training ─────────────────────────────────────────────────────────────

def train_rul(
    n_bearings:  int   = 60,
    seq_len:     int   = 20,
    epochs:      int   = 50,
    batch_size:  int   = 128,
    lr:          float = 1e-3,
    seed:        int   = 42,
):
    print("\n" + "="*60)
    print("  Training RULPredictor — Remaining Useful Life LSTM")
    print("="*60)

    device = get_device()
    print(f"  Device: {device}")

    print(f"\n  Generating RUL dataset ({n_bearings} bearings, seq_len={seq_len}) …")
    X, y = generate_rul_dataset(n_bearings=n_bearings, seq_len=seq_len, seed=seed)

    n      = len(y)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(n)

    def split_tensors(i):
        return (torch.from_numpy(X[i]), torch.from_numpy(y[i]))

    test_i  = idx[:n_test]
    val_i   = idx[n_test : n_test + n_val]
    train_i = idx[n_test + n_val:]

    def make_rul_loader(i, shuffle):
        Xt, yt = split_tensors(i)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    train_dl = make_rul_loader(train_i, shuffle=True)
    val_dl   = make_rul_loader(val_i,   shuffle=False)
    test_dl  = make_rul_loader(test_i,  shuffle=False)

    n_features = X.shape[-1]
    model      = RULPredictor(n_features=n_features, hidden_size=128, n_layers=2).to(device)
    criterion  = RULLoss(alpha=1.5)
    optimiser  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5, verbose=False
    )

    best_val_rmse = float("inf")
    history = {"train_rmse": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item() * len(yb)
            t_n    += len(yb)

        model.eval()
        v_preds, v_true = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                v_preds.extend(model(xb).cpu().numpy())
                v_true.extend(yb.numpy())

        val_rmse = np.sqrt(mean_squared_error(v_true, v_preds))
        train_rmse = np.sqrt(t_loss / t_n)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), CKPT_DIR / "rul_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_RMSE={train_rmse:.2f}  val_RMSE={val_rmse:.2f}")

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(CKPT_DIR / "rul_best.pt", map_location=device))
    model.eval()
    t_preds, t_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            t_preds.extend(model(xb).cpu().numpy())
            t_true.extend(yb.numpy())

    test_rmse = np.sqrt(mean_squared_error(t_true, t_preds))
    test_mae  = mean_absolute_error(t_true, t_preds)
    print(f"\n  ✓ Test RMSE: {test_rmse:.2f} cycles  |  MAE: {test_mae:.2f} cycles")
    print(f"    Best val RMSE: {best_val_rmse:.2f}")

    np.save(CKPT_DIR / "rul_history.npy", history)
    np.save(CKPT_DIR / "rul_test_results.npy",
            {"preds": np.array(t_preds), "true": np.array(t_true)})
    print(f"  Checkpoint saved → {CKPT_DIR}/rul_best.pt")
    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance models")
    parser.add_argument("--model",    choices=["cnn", "rul", "both"], default="both")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--samples",  type=int,   default=400,
                        help="Samples per class for CNN training")
    parser.add_argument("--snr",      type=float, default=20.0,
                        help="Signal-to-noise ratio in dB (lower = harder)")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    t0 = time.time()
    if args.model in ("cnn", "both"):
        train_cnn(n_samples=args.samples, epochs=args.epochs,
                  batch_size=args.batch, snr_db=args.snr, seed=args.seed)
    if args.model in ("rul", "both"):
        train_rul(epochs=args.epochs, batch_size=args.batch, seed=args.seed)

    print(f"\n  Total time: {time.time() - t0:.1f}s")
