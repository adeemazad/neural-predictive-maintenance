"""
evaluate.py — Visualise model performance metrics.

Generates publication-quality figures saved to reports/figures/:
  - Confusion matrix (normalised)
  - Training history curves
  - RUL predicted vs actual scatter
  - FFT spectrum comparison across fault types
  - Envelope spectrum comparison

Run after training:  python evaluate.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "reports" / "figures"
CKPT_DIR    = ROOT / "models" / "checkpoints"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FAULT_LABELS = ["Normal", "Inner Race", "Outer Race", "Ball Fault"]
PALETTE      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]


def plot_confusion_matrix(y_true, y_pred, save: bool = True) -> plt.Figure:
    cm     = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=FAULT_LABELS, yticklabels=FAULT_LABELS,
        linewidths=0.5, linecolor="white", ax=ax, vmin=0, vmax=1
    )
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title("Normalised Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {FIGURES_DIR}/confusion_matrix.png")
    return fig


def plot_training_history(history: dict, model_name: str = "CNN",
                           save: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train", color="#2196F3")
        axes[0].plot(history["val_loss"],   label="Val",   color="#FF9800", linestyle="--")
        axes[0].set_title("Loss", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history["train_acc"], label="Train", color="#2196F3")
        axes[1].plot(history["val_acc"],   label="Val",   color="#FF9800", linestyle="--")
        axes[1].set_title("Accuracy", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()

    elif "train_rmse" in history:
        axes[0].plot(history["train_rmse"], label="Train RMSE", color="#2196F3")
        axes[0].plot(history["val_rmse"],   label="Val RMSE",   color="#FF9800", linestyle="--")
        axes[0].set_title("RMSE (cycles)", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].set_visible(False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{model_name} Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        fname = f"training_history_{model_name.lower()}.png"
        fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
        print(f"  Saved → {FIGURES_DIR}/{fname}")
    return fig


def plot_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                          save: bool = True) -> plt.Figure:
    """Scatter: predicted vs actual RUL, coloured by absolute error."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    err    = np.abs(y_pred - y_true)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(y_true, y_pred, c=err, cmap="YlOrRd", s=12, alpha=0.7)
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual RUL (cycles)", fontsize=11)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=11)
    ax.set_title("RUL: Predicted vs Actual", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("|Error| (cycles)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "rul_scatter.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {FIGURES_DIR}/rul_scatter.png")
    return fig


def plot_fft_comparison(fs: int = 12_000, window: int = 2048,
                         save: bool = True) -> plt.Figure:
    """FFT power spectrum for each fault type side-by-side."""
    from src.data.loader import generate_synthetic_signal
    from src.features.dsp import compute_fft

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (ax, label, color) in enumerate(zip(axes, FAULT_LABELS, PALETTE)):
        sig    = generate_synthetic_signal(fault_type=i, duration=2.0, fs=fs)
        result = compute_fft(sig[:window], fs=fs)

        ax.semilogy(result["freqs"], result["power"] + 1e-12, color=color,
                    linewidth=0.8, alpha=0.9)
        ax.set_title(label, fontweight="bold", color=color)
        ax.set_xlim(0, fs / 2)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Power spectral density", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle("FFT Power Spectra — Fault Type Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "fft_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {FIGURES_DIR}/fft_comparison.png")
    return fig


def plot_envelope_comparison(fs: int = 12_000, window: int = 4096,
                              save: bool = True) -> plt.Figure:
    """Envelope spectrum for each fault type — shows characteristic frequencies."""
    from src.data.loader import generate_synthetic_signal
    from src.features.dsp import compute_envelope_spectrum

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, (ax, label, color) in enumerate(zip(axes, FAULT_LABELS, PALETTE)):
        sig    = generate_synthetic_signal(fault_type=i, duration=2.0, fs=fs)
        result = compute_envelope_spectrum(sig[:window], fs=fs)

        # Show only up to 300 Hz where fault frequencies lie
        mask = result["freqs"] < 300
        ax.plot(result["freqs"][mask], result["power"][mask], color=color,
                linewidth=0.9)
        ax.set_title(label, fontweight="bold", color=color)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Envelope power", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        # Annotate known fault frequencies
        fault_freqs = {1: 162.2, 2: 107.4, 3: 141.2}
        if i in fault_freqs:
            ax.axvline(fault_freqs[i], color="red", linestyle="--",
                       linewidth=0.8, alpha=0.7,
                       label=f"BPFI/BPFO/BSF = {fault_freqs[i]} Hz")
            ax.legend(fontsize=8)

    fig.suptitle("Envelope Spectra — Fault Characteristic Frequencies",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "envelope_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {FIGURES_DIR}/envelope_comparison.png")
    return fig


if __name__ == "__main__":
    print("Generating evaluation figures …\n")

    # FFT & envelope plots (no model needed)
    plot_fft_comparison()
    plot_envelope_comparison()

    # Training history (if checkpoints exist)
    cnn_hist_path = CKPT_DIR / "cnn_history.npy"
    if cnn_hist_path.exists():
        hist = np.load(cnn_hist_path, allow_pickle=True).item()
        plot_training_history(hist, model_name="CNN")
    else:
        print(f"  Skipping CNN history (run train.py first)")

    rul_hist_path = CKPT_DIR / "rul_history.npy"
    if rul_hist_path.exists():
        hist = np.load(rul_hist_path, allow_pickle=True).item()
        plot_training_history(hist, model_name="RUL_LSTM")

    rul_res_path = CKPT_DIR / "rul_test_results.npy"
    if rul_res_path.exists():
        res = np.load(rul_res_path, allow_pickle=True).item()
        plot_rul_predictions(res["true"], res["preds"])

    print("\nAll figures saved to:", FIGURES_DIR)
