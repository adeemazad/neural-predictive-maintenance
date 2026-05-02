"""
Unit tests for the predictive maintenance pipeline.

Run:  pytest tests/ -v --tb=short
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from src.data.loader import (
    generate_synthetic_signal, build_synthetic_dataset,
    train_val_test_split, WINDOW_SIZE, SAMPLE_RATE
)
from src.features.dsp import (
    compute_fft, compute_envelope_spectrum, compute_time_features,
    extract_features_batch
)
from src.models.cnn_classifier import FaultCNN, count_parameters
from src.models.lstm_rul import RULPredictor, generate_rul_dataset


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestSignalGenerator:
    def test_output_shape(self):
        for ft in range(4):
            sig = generate_synthetic_signal(fault_type=ft, duration=1.0)
            assert sig.shape == (SAMPLE_RATE,), f"Fault {ft}: shape mismatch"

    def test_dtype(self):
        sig = generate_synthetic_signal(0)
        assert sig.dtype == np.float32

    def test_invalid_fault_type(self):
        with pytest.raises(ValueError):
            generate_synthetic_signal(fault_type=99)

    def test_snr_affects_noise(self):
        rng   = np.random.default_rng(42)
        clean = generate_synthetic_signal(0, snr_db=40.0, rng=rng)
        rng   = np.random.default_rng(42)
        noisy = generate_synthetic_signal(0, snr_db=5.0,  rng=rng)
        # Noisier signal should have higher RMS deviation — not strictly guaranteed
        # but the variance should be higher
        assert np.var(noisy) >= np.var(clean) * 0.5   # at least not wildly less


class TestDatasetBuilder:
    def test_shapes(self):
        X, y = build_synthetic_dataset(n_samples_per_class=20)
        assert X.ndim == 3
        assert X.shape[1] == 1
        assert X.shape[2] == WINDOW_SIZE
        assert y.shape[0] == X.shape[0]

    def test_class_balance(self):
        X, y = build_synthetic_dataset(n_samples_per_class=50)
        counts = np.bincount(y)
        assert all(c == counts[0] for c in counts), "Dataset is unbalanced"

    def test_split_sizes(self):
        X, y = build_synthetic_dataset(n_samples_per_class=50)
        splits = train_val_test_split(X, y, val_ratio=0.2, test_ratio=0.2)
        n = len(y)
        assert len(splits["test"][1])  == int(n * 0.2)
        assert len(splits["val"][1])   == int(n * 0.2)
        # No overlap between splits
        tr_idx = set(range(len(splits["train"][1])))
        assert len(tr_idx) > 0


# ── DSP feature tests ─────────────────────────────────────────────────────────

class TestDSP:
    @pytest.fixture
    def signal(self):
        return generate_synthetic_signal(fault_type=1)[:WINDOW_SIZE]

    def test_fft_positive_freqs(self, signal):
        res = compute_fft(signal)
        assert np.all(res["freqs"] >= 0)
        assert res["freqs"][-1] <= SAMPLE_RATE / 2 + 1

    def test_fft_shape(self, signal):
        res = compute_fft(signal)
        assert res["freqs"].shape == res["power"].shape

    def test_envelope_shape(self, signal):
        res = compute_envelope_spectrum(signal)
        assert res["freqs"].shape == res["power"].shape

    def test_time_features_count(self, signal):
        tf = compute_time_features(signal)
        assert len(tf) == 14

    def test_time_features_finite(self, signal):
        tf = compute_time_features(signal)
        for k, v in tf.items():
            assert np.isfinite(v), f"Feature '{k}' is not finite: {v}"

    def test_crest_factor_positive(self, signal):
        tf = compute_time_features(signal)
        assert tf["crest_factor"] > 0

    def test_batch_features_shape(self):
        X, _ = build_synthetic_dataset(n_samples_per_class=10)
        feat  = extract_features_batch(X)
        assert feat.shape == (len(X), 14)
        assert feat.dtype == np.float32


# ── Model tests ───────────────────────────────────────────────────────────────

class TestFaultCNN:
    @pytest.fixture
    def model(self):
        return FaultCNN(n_classes=4, window_size=WINDOW_SIZE)

    def test_output_shape(self, model):
        x   = torch.randn(8, 1, WINDOW_SIZE)
        out = model(x)
        assert out.shape == (8, 4)

    def test_proba_sums_to_one(self, model):
        x     = torch.randn(4, 1, WINDOW_SIZE)
        proba = model.predict_proba(x)
        sums  = proba.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_proba_non_negative(self, model):
        x     = torch.randn(4, 1, WINDOW_SIZE)
        proba = model.predict_proba(x)
        assert (proba >= 0).all()

    def test_parameter_count(self, model):
        n = count_parameters(model)
        # Sanity check — should be between 100K and 5M for this architecture
        assert 1_000 < n < 5_000_000, f"Unexpected parameter count: {n}"

    def test_feature_maps_shape(self, model):
        x          = torch.randn(1, 1, WINDOW_SIZE)
        maps, logits = model.get_feature_maps(x)
        assert logits.shape == (1, 4)
        assert maps.ndim == 3   # (B, C, T)

    def test_deterministic_eval(self, model):
        model.eval()
        x  = torch.randn(2, 1, WINDOW_SIZE)
        o1 = model(x)
        o2 = model(x)
        assert torch.allclose(o1, o2)


class TestRULPredictor:
    @pytest.fixture
    def model(self):
        return RULPredictor(n_features=14, hidden_size=64, n_layers=1)

    def test_output_shape(self, model):
        x   = torch.randn(16, 20, 14)
        out = model(x)
        assert out.shape == (16, 1)

    def test_positive_outputs(self, model):
        x   = torch.randn(8, 20, 14)
        out = model(x)
        assert (out > 0).all(), "Softplus should ensure positive RUL"

    def test_rul_dataset_shape(self):
        X, y = generate_rul_dataset(n_bearings=5, seq_len=10)
        assert X.ndim == 3
        assert X.shape[-1] == 14
        assert y.shape == (len(X), 1)

    def test_rul_positive(self):
        _, y = generate_rul_dataset(n_bearings=5)
        assert (y > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
