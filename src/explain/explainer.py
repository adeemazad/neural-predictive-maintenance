"""
Explainability module: SHAP feature importance + Grad-CAM for 1D-CNN.

Two complementary techniques:
  1. SHAP (SHapley Additive exPlanations) — explains which time-domain
     statistical features push the prediction toward each fault class.
  2. Grad-CAM — highlights which regions of the raw vibration signal
     the CNN focuses on (visualised as a heatmap over time).
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional


FAULT_LABELS = {0: "Normal", 1: "Inner Race", 2: "Outer Race", 3: "Ball Fault"}
FEATURE_NAMES = [
    "Mean", "Std", "RMS", "Peak", "Crest Factor", "Kurtosis", "Skewness",
    "Shape Factor", "Impulse Factor", "Clearance Factor",
    "Peak-to-Peak", "Variance", "Energy", "Zero Crossings",
]


# ── SHAP ──────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Wraps a trained sklearn-compatible classifier (e.g. RandomForestClassifier
    trained on statistical features) with SHAP's TreeExplainer.

    For the CNN, we use a surrogate: extract statistical features, then
    train a lightweight Random Forest on them and explain that.
    This gives interpretable feature-level explanations.
    """

    def __init__(self, model, feature_names: list[str] = FEATURE_NAMES):
        self.model         = model
        self.feature_names = feature_names
        self._explainer    = None

    def fit(self, X_background: np.ndarray):
        """
        Fit the SHAP explainer on background data.
        X_background: (N, n_features) statistical feature matrix.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        self._explainer = shap.TreeExplainer(self.model)
        return self

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values array of shape (N, n_features, n_classes)."""
        if self._explainer is None:
            raise RuntimeError("Call .fit() first")
        import shap
        sv = self._explainer.shap_values(X)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)   # (N, F, C)
        return sv

    def plot_summary(self, X: np.ndarray, max_display: int = 10,
                     save_path: Optional[str] = None) -> plt.Figure:
        """Bar summary plot of mean |SHAP| per feature."""
        import shap
        sv  = self.compute_shap_values(X)         # (N, F, C)
        mean_abs = np.mean(np.abs(sv), axis=(0, 2))  # (F,)

        fig, ax = plt.subplots(figsize=(8, 4))
        idx     = np.argsort(mean_abs)[-max_display:]
        colors  = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
        ax.barh(
            [self.feature_names[i] for i in idx],
            mean_abs[idx],
            color=colors
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature importance (SHAP)", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_class_beeswarm(self, X: np.ndarray, class_idx: int = 1,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Beeswarm plot for a single fault class."""
        sv = self.compute_shap_values(X)  # (N, F, C)
        sv_class = sv[:, :, class_idx]    # (N, F)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, name in enumerate(self.feature_names):
            vals = sv_class[:, i]
            feat = X[:, i]
            norm = (feat - feat.min()) / (feat.ptp() + 1e-8)
            ax.scatter(vals, [i] * len(vals),
                       c=norm, cmap="RdYlBu_r", s=8, alpha=0.6)

        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names, fontsize=9)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value", fontsize=11)
        ax.set_title(f"SHAP beeswarm — {FAULT_LABELS.get(class_idx, class_idx)}",
                     fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Grad-CAM for 1D-CNN ────────────────────────────────────────────────────────

class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for a 1D-CNN.

    Hooks into the last convolutional block and computes:
      CAM = ReLU(Σ_k  α_k · A_k)

    where α_k = global average pooled gradient of class score w.r.t. feature map k,
    and   A_k = feature map k from the target layer.
    """

    def __init__(self, model: torch.nn.Module):
        self.model        = model
        self._activations = None
        self._gradients   = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last Conv layer in model.features."""
        try:
            target_layer = list(self.model.features.children())[-1].conv
        except AttributeError:
            # Fallback: last Conv1d found anywhere
            for layer in reversed(list(self.model.modules())):
                if isinstance(layer, torch.nn.Conv1d):
                    target_layer = layer
                    break

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def compute(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        x         : (1, 1, window_size)  — single sample, batch dim=1
        class_idx : int or None  — if None uses the predicted class

        Returns
        -------
        cam : np.ndarray  shape (window_size,)  — upsampled heatmap in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        _, logits = self.model.get_feature_maps(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # α_k = global average of gradients over time dimension
        alpha = self._gradients.mean(dim=-1, keepdim=True)  # (1, C, 1)
        cam   = F.relu((alpha * self._activations).sum(dim=1))  # (1, T)
        cam   = cam.squeeze().cpu().numpy()

        # Upsample to input length
        from scipy.ndimage import zoom
        scale = x.shape[-1] / len(cam)
        cam   = zoom(cam, scale)

        # Normalise to [0, 1]
        cam  -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def plot(self, signal: np.ndarray, cam: np.ndarray,
             class_idx: int, fs: int = 12_000,
             save_path: Optional[str] = None) -> plt.Figure:
        """
        Overlay Grad-CAM heatmap on the raw vibration signal.
        Red regions = high attention, Blue = low attention.
        """
        t   = np.arange(len(signal)) / fs * 1000   # ms
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Signal coloured by CAM value
        from matplotlib.collections import LineCollection
        points  = np.array([t, signal]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        lc      = LineCollection(segs, cmap="RdYlBu_r",
                                 norm=mcolors.Normalize(0, 1))
        lc.set_array(cam[:-1])
        lc.set_linewidth(1.0)
        ax1.add_collection(lc)
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(signal.min() * 1.2, signal.max() * 1.2)
        ax1.set_ylabel("Acceleration (normalised)", fontsize=10)
        ax1.set_title(
            f"Grad-CAM — predicted: {FAULT_LABELS.get(class_idx, class_idx)}",
            fontsize=13, fontweight="bold"
        )

        # Heatmap bar
        ax2.imshow(cam[np.newaxis, :], aspect="auto", cmap="RdYlBu_r",
                   extent=[t[0], t[-1], 0, 1])
        ax2.set_yticks([])
        ax2.set_xlabel("Time (ms)", fontsize=10)
        ax2.set_ylabel("Attention", fontsize=9)

        cbar = fig.colorbar(lc, ax=[ax1, ax2], orientation="vertical", pad=0.01)
        cbar.set_label("Importance", fontsize=9)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ── Convenience: train surrogate RF + explain ─────────────────────────────────

def explain_with_surrogate(
    X_feat:  np.ndarray,
    y:       np.ndarray,
    X_test:  np.ndarray,
    n_estimators: int = 100,
    save_dir: Optional[str] = None,
) -> dict:
    """
    1. Train a Random Forest on statistical features (surrogate)
    2. Report its accuracy
    3. Build SHAP explainer and return figures

    Returns dict with 'fig_summary', 'fig_beeswarm', 'rf_model', 'shap_values'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_feat, y, cv=5, scoring="accuracy")
    rf.fit(X_feat, y)

    print(f"  Surrogate RF — CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    explainer = SHAPExplainer(rf, FEATURE_NAMES[:X_feat.shape[1]])
    explainer.fit(X_feat)

    figs = {}
    figs["fig_summary"]  = explainer.plot_summary(
        X_test, save_path=f"{save_dir}/shap_summary.png" if save_dir else None
    )
    figs["fig_beeswarm"] = explainer.plot_class_beeswarm(
        X_test, class_idx=1,
        save_path=f"{save_dir}/shap_beeswarm_inner.png" if save_dir else None
    )
    figs["rf_model"]    = rf
    figs["cv_scores"]   = cv_scores

    return figs


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.data.loader import build_synthetic_dataset, train_val_test_split
    from src.features.dsp import extract_features_batch

    print("Building dataset and extracting features …")
    X, y = build_synthetic_dataset(n_samples_per_class=100)
    splits = train_val_test_split(X, y)

    Xtr, ytr = splits["train"]
    Xte, yte = splits["test"]

    Xtr_feat = extract_features_batch(Xtr)
    Xte_feat = extract_features_batch(Xte)

    print("Training surrogate RF + computing SHAP …")
    figs = explain_with_surrogate(Xtr_feat, ytr, Xte_feat, save_dir="/tmp")
    print("Figures saved to /tmp/")
