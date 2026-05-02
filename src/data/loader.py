"""
Data loading and synthetic signal generation for predictive maintenance.

CWRU dataset: https://engineering.case.edu/bearingdatacenter
Generates labelled vibration windows for training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import urllib.request
import os


# ── Fault class labels ──────────────────────────────────────────────────────
FAULT_LABELS = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
    3: "Ball Fault",
}

SAMPLE_RATE = 12_000   # Hz — CWRU 12kHz drive-end files
WINDOW_SIZE = 2_048    # samples per training window (~170ms)
STRIDE      = 512      # hop between windows


# ── Synthetic signal generator ───────────────────────────────────────────────

def _sine(t, freq, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t)


def generate_synthetic_signal(
    fault_type: int,
    duration: float = 2.0,
    fs: int = SAMPLE_RATE,
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthesise a bearing vibration signal for a given fault class.

    Parameters
    ----------
    fault_type : int  0=Normal, 1=Inner, 2=Outer, 3=Ball
    duration   : float  seconds of signal to generate
    fs         : int    sample rate (Hz)
    snr_db     : float  signal-to-noise ratio in dB
    rng        : numpy Generator (for reproducibility)

    Returns
    -------
    signal : np.ndarray  shape (N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Shaft rotation and harmonics (common to all)
    shaft_freq = 29.95  # Hz — typical test rig speed
    signal = (
        _sine(t, shaft_freq, amp=0.5)
        + _sine(t, 2 * shaft_freq, amp=0.2)
        + _sine(t, 3 * shaft_freq, amp=0.1)
    )

    if fault_type == 0:
        # Normal — just shaft harmonics + low noise
        pass

    elif fault_type == 1:
        # Inner race fault — BPFI ~162 Hz, modulated by shaft
        bpfi = 162.2
        mod  = 1.0 + 0.4 * np.sin(2 * np.pi * shaft_freq * t)
        signal += mod * (
            _sine(t, bpfi,     amp=0.8)
            + _sine(t, 2*bpfi, amp=0.3)
            + _sine(t, 3*bpfi, amp=0.1)
        )
        # Impulse train
        impulse_period = int(fs / bpfi)
        impulses = np.zeros_like(t)
        impulses[::impulse_period] = rng.uniform(0.5, 1.5, size=len(impulses[::impulse_period]))
        signal += np.convolve(impulses, np.exp(-np.linspace(0, 5, 80)), mode="same") * 0.6

    elif fault_type == 2:
        # Outer race fault — BPFO ~107 Hz, not modulated
        bpfo = 107.4
        signal += (
            _sine(t, bpfo,     amp=1.0)
            + _sine(t, 2*bpfo, amp=0.4)
            + _sine(t, 3*bpfo, amp=0.15)
        )
        impulse_period = int(fs / bpfo)
        impulses = np.zeros_like(t)
        impulses[::impulse_period] = rng.uniform(0.8, 1.2, size=len(impulses[::impulse_period]))
        signal += np.convolve(impulses, np.exp(-np.linspace(0, 4, 60)), mode="same") * 0.8

    elif fault_type == 3:
        # Ball fault — BSF ~141 Hz, modulated at 2× shaft (ball spins)
        bsf  = 141.2
        mod  = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * shaft_freq * t)
        signal += mod * (
            _sine(t, bsf,     amp=0.6)
            + _sine(t, 2*bsf, amp=0.2)
        )
    else:
        raise ValueError(f"fault_type must be 0–3, got {fault_type}")

    # Add AWGN at the requested SNR
    signal_power = np.mean(signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    signal += rng.normal(0, np.sqrt(noise_power), size=len(t))

    return signal.astype(np.float32)


def build_synthetic_dataset(
    n_samples_per_class: int = 300,
    window_size: int = WINDOW_SIZE,
    fs: int = SAMPLE_RATE,
    snr_db: float = 20.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a balanced dataset of vibration windows from synthetic signals.

    Returns
    -------
    X : np.ndarray  shape (N, 1, window_size)  — channel-first for Conv1d
    y : np.ndarray  shape (N,)                 — integer class labels
    """
    rng = np.random.default_rng(seed)
    windows, labels = [], []

    for fault_type in range(4):
        # Generate a long signal and slice into windows
        duration_needed = (n_samples_per_class * window_size) / fs + 2.0
        sig = generate_synthetic_signal(
            fault_type, duration=duration_needed, fs=fs, snr_db=snr_db, rng=rng
        )
        starts = np.arange(0, len(sig) - window_size, window_size)[:n_samples_per_class]
        for s in starts:
            windows.append(sig[s : s + window_size])
            labels.append(fault_type)

    X = np.stack(windows)[:, np.newaxis, :]   # (N, 1, W)
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Optional: load real CWRU .mat files ─────────────────────────────────────

def load_cwru_mat(path: str | Path, key: str = "X097_DE_time") -> np.ndarray:
    """
    Load a CWRU MATLAB .mat file and return the raw vibration array.

    Download files from: https://engineering.case.edu/bearingdatacenter/download-data-file
    Typical keys: 'X097_DE_time' (normal), 'X105_DE_time' (IR fault), etc.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")

    mat  = loadmat(str(path))
    keys = [k for k in mat.keys() if not k.startswith("_")]
    if key not in mat:
        key = keys[0]
    return mat[key].flatten().astype(np.float32)


def cwru_to_windows(
    signal: np.ndarray,
    label: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a 1-D signal into overlapping windows."""
    starts = range(0, len(signal) - window_size, stride)
    wins   = np.stack([signal[s : s + window_size] for s in starts])
    wins   = wins[:, np.newaxis, :]   # (N, 1, W)
    lbls   = np.full(len(wins), label, dtype=np.int64)
    return wins, lbls


# ── Simple train/val/test split ──────────────────────────────────────────────

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n   = len(y)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val":   (X[val_idx],   y[val_idx]),
        "test":  (X[test_idx],  y[test_idx]),
    }


if __name__ == "__main__":
    print("Building synthetic dataset …")
    X, y = build_synthetic_dataset(n_samples_per_class=200)
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {np.unique(y, return_counts=True)}")
    splits = train_val_test_split(X, y)
    for split, (Xs, ys) in splits.items():
        print(f"  {split:5s}  X={Xs.shape}  y={ys.shape}")
