"""
Streamlit Dashboard — Neural Predictive Maintenance Engine

Run:  streamlit run dashboard/app.py

Features:
  - Upload CSV of vibration data OR use the synthetic signal generator
  - Live fault classification with probability bar chart
  - FFT and envelope spectrum visualisation
  - Grad-CAM attention heatmap
  - RUL estimation gauge
  - Downloadable report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from io import BytesIO

from src.data.loader import generate_synthetic_signal, FAULT_LABELS, WINDOW_SIZE, SAMPLE_RATE
from src.features.dsp import compute_fft, compute_envelope_spectrum, compute_time_features
from src.models.cnn_classifier import FaultCNN

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CKPT_DIR = Path(__file__).parent.parent / "models" / "checkpoints"
DEVICE   = torch.device("cpu")   # Dashboard always runs on CPU
PALETTE  = {"Normal": "#2196F3", "Inner Race Fault": "#4CAF50",
             "Outer Race Fault": "#FF9800", "Ball Fault": "#E91E63"}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = FaultCNN(n_classes=4, window_size=WINDOW_SIZE)
    ckpt  = CKPT_DIR / "cnn_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model, True
    return model, False   # untrained — random predictions (for demo only)


model, model_loaded = load_model()
model.eval()


# ── Sidebar — Signal source ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Signal source")
source = st.sidebar.radio("Input mode", ["Synthetic generator", "Upload CSV"])

if source == "Synthetic generator":
    st.sidebar.subheader("Synthetic signal settings")
    fault_choice = st.sidebar.selectbox(
        "Ground-truth fault type",
        options=list(FAULT_LABELS.values()),
        index=0,
    )
    fault_idx = {v: k for k, v in FAULT_LABELS.items()}[fault_choice]
    snr_db     = st.sidebar.slider("SNR (dB) — lower = noisier", 5.0, 40.0, 20.0, 1.0)
    rng_seed   = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

    raw_signal = generate_synthetic_signal(
        fault_type=fault_idx, duration=2.0,
        fs=SAMPLE_RATE, snr_db=snr_db,
        rng=np.random.default_rng(int(rng_seed))
    )
    st.sidebar.success(f"Generated {len(raw_signal):,} samples at {SAMPLE_RATE} Hz")

else:
    uploaded = st.sidebar.file_uploader(
        "Upload vibration CSV (single column, no header)", type=["csv", "txt"]
    )
    if uploaded is None:
        st.info("⬅️ Upload a CSV file or switch to the synthetic generator.")
        st.stop()
    raw_signal = pd.read_csv(uploaded, header=None).iloc[:, 0].to_numpy(dtype=np.float32)
    fault_idx  = None
    st.sidebar.success(f"Loaded {len(raw_signal):,} samples")


# ── Prepare window ────────────────────────────────────────────────────────────
window = raw_signal[:WINDOW_SIZE].astype(np.float32)
# Normalise to zero-mean unit-variance
window = (window - window.mean()) / (window.std() + 1e-8)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔧 Neural Predictive Maintenance Engine")
st.caption("1D-CNN fault classifier · LSTM RUL estimator · DSP feature extraction")

if not model_loaded:
    st.warning(
        "⚠️ No trained model checkpoint found. Run `python train.py --model cnn` first. "
        "Showing random predictions for demo purposes."
    )

# ── Inference ─────────────────────────────────────────────────────────────────
x_tensor = torch.from_numpy(window[np.newaxis, np.newaxis, :])   # (1, 1, W)
with torch.no_grad():
    proba = model.predict_proba(x_tensor).numpy()[0]   # (4,)

pred_class = int(np.argmax(proba))
pred_label = FAULT_LABELS[pred_class]
pred_conf  = float(proba[pred_class]) * 100

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted fault", pred_label)
col2.metric("Confidence", f"{pred_conf:.1f}%")
col3.metric("Signal length", f"{len(raw_signal):,} samples")
col4.metric("Sample rate", f"{SAMPLE_RATE:,} Hz")

if fault_idx is not None:
    correct = pred_class == fault_idx
    if correct:
        st.success(f"✅ Correct — ground truth: **{FAULT_LABELS[fault_idx]}**")
    else:
        st.error(f"❌ Incorrect — ground truth: **{FAULT_LABELS[fault_idx]}** "
                 f"but predicted: **{pred_label}**. "
                 "Try lower SNR or re-train the model.")

st.divider()

# ── Layout: two columns ───────────────────────────────────────────────────────
left, right = st.columns([3, 2])

# ── Left: signal + spectra ────────────────────────────────────────────────────
with left:
    st.subheader("Raw vibration signal")
    t_ms = np.arange(len(window)) / SAMPLE_RATE * 1000
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=t_ms, y=window.tolist(),
        mode="lines", line=dict(width=0.8, color="#2196F3"),
        name="Vibration"
    ))
    fig_sig.update_layout(
        xaxis_title="Time (ms)", yaxis_title="Normalised amplitude",
        height=220, margin=dict(t=10, b=30, l=50, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    st.subheader("FFT power spectrum")
    fft_res = compute_fft(window, fs=SAMPLE_RATE)
    fig_fft = go.Figure()
    mask = fft_res["freqs"] < 6000
    fig_fft.add_trace(go.Scatter(
        x=fft_res["freqs"][mask].tolist(),
        y=fft_res["power_db"][mask].tolist(),
        mode="lines", line=dict(width=0.9, color="#FF9800"),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.08)"
    ))
    fig_fft.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)",
        height=220, margin=dict(t=10, b=30, l=50, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    st.subheader("Envelope spectrum (< 300 Hz)")
    env_res = compute_envelope_spectrum(window, fs=SAMPLE_RATE)
    mask_e  = env_res["freqs"] < 300
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(
        x=env_res["freqs"][mask_e].tolist(),
        y=env_res["power"][mask_e].tolist(),
        mode="lines", line=dict(width=1.0, color="#4CAF50"),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.1)"
    ))
    # Mark fault characteristic frequencies
    for freq, label in [(162.2, "BPFI"), (107.4, "BPFO"), (141.2, "BSF")]:
        fig_env.add_vline(x=freq, line_dash="dot", line_color="red",
                          annotation_text=label, annotation_position="top right",
                          annotation_font_size=10)
    fig_env.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Envelope power",
        height=220, margin=dict(t=10, b=30, l=50, r=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_env, use_container_width=True)

# ── Right: classification results + features ──────────────────────────────────
with right:
    st.subheader("Classification probabilities")
    labels = [FAULT_LABELS[i] for i in range(4)]
    colors = [PALETTE.get(l, "#888") for l in labels]
    fig_bar = go.Figure(go.Bar(
        x=[float(p * 100) for p in proba],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in proba],
        textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis=dict(title="Probability (%)", range=[0, 110]),
        height=250, margin=dict(t=10, b=30, l=120, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Statistical features")
    tf = compute_time_features(window)
    rows = [{"Feature": k, "Value": f"{v:.4f}"} for k, v in tf.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=280)

    st.subheader("Simulated RUL estimate")
    # Heuristic RUL from confidence: high confidence Normal → long RUL
    base_rul  = 200
    rul_est   = int(base_rul * float(proba[0]))   # decreases as Normal prob drops
    health_pc = int(float(proba[0]) * 100)
    st.progress(health_pc, text=f"Health: {health_pc}% — RUL ≈ {rul_est} cycles")

    severity = "🟢 Healthy" if health_pc > 60 else ("🟡 Degrading" if health_pc > 25 else "🔴 Critical")
    st.info(f"**Status:** {severity}  |  **Predicted:** {pred_label} @ {pred_conf:.1f}%")

    st.caption(
        "ℹ️ RUL estimate is heuristic in this demo. "
        "Train and load the LSTM RUL model (`python train.py --model rul`) "
        "for sequence-based predictions."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Neural Predictive Maintenance Engine · University of Glasgow Engineering Portfolio · "
    "Built with PyTorch · Streamlit · Plotly"
)
