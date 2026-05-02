"""
Digital signal processing feature extraction for bearing vibration signals.

Implements:
  - FFT power spectrum
  - Envelope (Hilbert) spectrum
  - Continuous Wavelet Transform scalogram
  - Statistical time-domain features
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from typing import Dict


SAMPLE_RATE = 12_000   # Hz


# ── FFT / Frequency domain ───────────────────────────────────────────────────

def compute_fft(signal: np.ndarray, fs: int = SAMPLE_RATE) -> Dict[str, np.ndarray]:
    """
    Compute the one-sided power spectrum of a vibration window.

    Returns dict with 'freqs' (Hz) and 'power' (dB re 1 (m/s²)²/Hz).
    """
    n      = len(signal)
    fft    = rfft(signal * np.hanning(n))          # Hann window to reduce leakage
    power  = (2.0 / n) * np.abs(fft) ** 2         # one-sided PSD
    freqs  = rfftfreq(n, d=1.0 / fs)
    power_db = 10 * np.log10(power + 1e-12)        # avoid log(0)
    return {"freqs": freqs, "power": power, "power_db": power_db}


def compute_envelope_spectrum(signal: np.ndarray, fs: int = SAMPLE_RATE,
                               band_low: float = 2_000, band_high: float = 5_000
                               ) -> Dict[str, np.ndarray]:
    """
    Compute the envelope (Hilbert) spectrum — key for bearing fault diagnosis.

    Steps:
      1. Bandpass filter around resonance frequency
      2. Compute analytic signal via Hilbert transform
      3. Take absolute value → envelope
      4. FFT of envelope → fault characteristic frequencies appear
    """
    # Bandpass filter
    nyq = fs / 2
    b, a = butter(4, [band_low / nyq, band_high / nyq], btype="band")
    filtered = filtfilt(b, a, signal)

    # Hilbert envelope
    analytic  = hilbert(filtered)
    envelope  = np.abs(analytic)
    envelope -= envelope.mean()   # remove DC

    # FFT of envelope
    fft_env  = rfft(envelope * np.hanning(len(envelope)))
    env_power = (2.0 / len(envelope)) * np.abs(fft_env) ** 2
    env_freqs = rfftfreq(len(envelope), d=1.0 / fs)

    return {"freqs": env_freqs, "power": env_power, "envelope": envelope}


def compute_cwt_scalogram(
    signal: np.ndarray,
    fs: int = SAMPLE_RATE,
    n_scales: int = 64,
    max_freq: float = 4_000,
) -> np.ndarray:
    """
    Compute a Continuous Wavelet Transform scalogram using PyWavelets.

    Returns a (n_scales, n_time) float32 array suitable for 2D-CNN input.
    Falls back gracefully if pywt is not installed.
    """
    try:
        import pywt
    except ImportError:
        # Fallback: simple spectrogram via STFT
        from scipy.signal import spectrogram
        _, _, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=64)
        Sxx_db = 10 * np.log10(Sxx[:n_scales] + 1e-12)
        return Sxx_db.astype(np.float32)

    widths = np.geomspace(fs / max_freq, fs / 10, num=n_scales)
    coef, _ = pywt.cwt(signal, widths, "morl", sampling_period=1.0 / fs)
    scalogram = np.abs(coef).astype(np.float32)  # (n_scales, n_time)
    return scalogram


# ── Time-domain statistical features ────────────────────────────────────────

def compute_time_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract 14 statistical features from a raw vibration window.

    These are used by classical ML baselines (SVM, Random Forest) and
    can supplement the CNN as an auxiliary feature vector.
    """
    s    = signal.astype(np.float64)
    rms  = np.sqrt(np.mean(s ** 2))
    peak = np.max(np.abs(s))

    features = {
        "mean"          : float(np.mean(s)),
        "std"           : float(np.std(s)),
        "rms"           : float(rms),
        "peak"          : float(peak),
        "crest_factor"  : float(peak / (rms + 1e-8)),
        "kurtosis"      : float(_kurtosis(s)),
        "skewness"      : float(_skewness(s)),
        "shape_factor"  : float(rms / (np.mean(np.abs(s)) + 1e-8)),
        "impulse_factor": float(peak / (np.mean(np.abs(s)) + 1e-8)),
        "clearance_factor": float(peak / (np.mean(np.sqrt(np.abs(s))) + 1e-8) ** 2),
        "peak_to_peak"  : float(np.max(s) - np.min(s)),
        "variance"      : float(np.var(s)),
        "energy"        : float(np.sum(s ** 2)),
        "zero_crossings": float(np.sum(np.diff(np.sign(s)) != 0)),
    }
    return features


def _kurtosis(x: np.ndarray) -> float:
    mu, std = x.mean(), x.std()
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4))


def _skewness(x: np.ndarray) -> float:
    mu, std = x.mean(), x.std()
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


# ── Batch feature extraction ─────────────────────────────────────────────────

def extract_features_batch(
    X: np.ndarray,
    fs: int = SAMPLE_RATE,
    include_cwt: bool = False,
) -> np.ndarray:
    """
    Extract time-domain statistical features for every window in X.

    Parameters
    ----------
    X : np.ndarray  shape (N, 1, W) or (N, W)

    Returns
    -------
    features : np.ndarray  shape (N, n_features)
    """
    if X.ndim == 3:
        X = X[:, 0, :]   # squeeze channel dim

    rows = []
    for window in X:
        tf = compute_time_features(window)
        rows.append(list(tf.values()))

    return np.array(rows, dtype=np.float32)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.loader import generate_synthetic_signal

    sig = generate_synthetic_signal(fault_type=1)
    print("Signal:", sig.shape, sig.dtype)

    fft_res = compute_fft(sig)
    print("FFT freqs:", fft_res["freqs"].shape)

    env_res = compute_envelope_spectrum(sig)
    print("Envelope freqs:", env_res["freqs"].shape)

    tf = compute_time_features(sig[:2048])
    print("Time features:", tf)
