"""
LSTM Remaining Useful Life (RUL) Regressor

Takes a sequence of statistical feature vectors extracted from consecutive
vibration windows and regresses a scalar: estimated cycles remaining.

Input shape : (batch, seq_len, n_features)
Output shape: (batch, 1)

Compatible with the NASA FEMTO-ST PRONOSTIA dataset and synthetic RUL data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class RULPredictor(nn.Module):
    """
    Bidirectional LSTM → FC regressor for remaining useful life prediction.

    Parameters
    ----------
    n_features  : int   number of input features per time step
    hidden_size : int   LSTM hidden state dimension
    n_layers    : int   number of stacked LSTM layers
    dropout     : float dropout between LSTM layers (ignored when n_layers=1)
    bidirectional: bool use BiLSTM (better context, 2× hidden dim)
    """

    def __init__(
        self,
        n_features:   int  = 14,
        hidden_size:  int  = 128,
        n_layers:     int  = 2,
        dropout:      float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.n_layers      = n_layers
        self.bidirectional = bidirectional
        directions         = 2 if bidirectional else 1

        self.input_norm = nn.LayerNorm(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_in = hidden_size * directions
        self.head = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Dropout(dropout),
            nn.Linear(fc_in, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),   # ensures positive RUL output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)  — feature sequence

        Returns
        -------
        rul : (batch, 1)  — predicted remaining useful life (positive)
        """
        x    = self.input_norm(x)
        out, _ = self.lstm(x)       # (B, T, directions*hidden)
        last = out[:, -1, :]        # use final time step
        return self.head(last)


# ── Synthetic RUL data generator ─────────────────────────────────────────────

def generate_rul_dataset(
    n_bearings:      int   = 40,
    max_life_cycles: int   = 200,
    seq_len:         int   = 20,
    n_features:      int   = 14,
    seed:            int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate degradation trajectories for N bearings.

    Each bearing has a randomly assigned lifetime. Features degrade
    gradually following a Weibull-like curve with additive noise.

    Returns
    -------
    X : (N_samples, seq_len, n_features)
    y : (N_samples, 1)   — RUL in cycles
    """
    rng = np.random.default_rng(seed)
    X_all, y_all = [], []

    for _ in range(n_bearings):
        life = rng.integers(max_life_cycles // 2, max_life_cycles)

        # Degradation index per cycle: 0 (new) → 1 (failed)
        t        = np.linspace(0, 1, life)
        k        = rng.uniform(2.0, 5.0)          # Weibull shape
        deg      = 1 - np.exp(-(t * 2.5) ** k)   # [0, 1]

        # Feature matrix: each feature degrades differently
        base     = rng.uniform(0.1, 1.0, size=(n_features,))
        amp      = rng.uniform(0.5, 3.0, size=(n_features,))
        noise    = rng.normal(0, 0.05, size=(life, n_features))
        features = base + amp * deg[:, None] + noise   # (life, n_features)

        # Normalise each feature to [0, 1] over this bearing's life
        f_min = features.min(axis=0, keepdims=True)
        f_max = features.max(axis=0, keepdims=True) + 1e-8
        features = (features - f_min) / (f_max - f_min)

        # Slide a window across the life
        for i in range(seq_len, life):
            seq = features[i - seq_len : i]     # (seq_len, n_features)
            rul = float(life - i)               # cycles remaining
            X_all.append(seq)
            y_all.append([rul])

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    # Normalise RUL to [0, max_life] for stable training
    # (keep raw values for evaluation, scale for model)
    idx = np.random.default_rng(seed + 1).permutation(len(y))
    return X[idx], y[idx]


# ── Training utility ──────────────────────────────────────────────────────────

class RULLoss(nn.Module):
    """
    Asymmetric RMSE: penalises under-prediction (predicting too much life
    remaining) more than over-prediction — conservative for safety.
    """

    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err  = pred - target
        loss = torch.where(err < 0, self.alpha * err ** 2, err ** 2)
        return loss.mean()


if __name__ == "__main__":
    model = RULPredictor(n_features=14, hidden_size=128, n_layers=2)
    print(model)

    dummy = torch.randn(16, 20, 14)
    out   = model(dummy)
    print(f"\nInput : {dummy.shape}")
    print(f"Output: {out.shape}  (all positive: {(out > 0).all().item()})")

    print("\nGenerating synthetic RUL dataset …")
    X, y = generate_rul_dataset(n_bearings=20, seq_len=20)
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  RUL range: {y.min():.0f} – {y.max():.0f} cycles")
