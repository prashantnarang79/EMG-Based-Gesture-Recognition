from pathlib import Path
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score

from models.TinyMyo import TinyMyo


ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    ROOT / "data" / "UCI_EMG"
    / "EMG_data_for_gestures-master"
    / "h5_5sec" / "test.h5"
)

CONFIG = ROOT / "model_assets" / "config.json"

CHECKPOINT = (
    ROOT / "model_assets" / "UCI_finetune_5sec.safetensors"
)

RESULTS_DIR = ROOT / "results"
CONFIDENCE_DIR = RESULTS_DIR / "confidence"

GESTURES = [
    "Hand at rest",
    "Fist",
    "Wrist flexion",
    "Wrist extension",
    "Radial deviation",
    "Ulnar deviation",
]

st.set_page_config(
    page_title="EMG Gesture Recognition",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")

    model = TinyMyo(**CONFIG)

    state = load_file(str(CHECKPOINT), device="cpu")
    state = {
        key.removeprefix("model."): value
        for key, value in state.items()
    }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@st.cache_data
def load_test_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Test data missing: {DATA_FILE}")

    with h5py.File(DATA_FILE, "r") as f:
        X = np.asarray(f["data"][:], dtype=np.float32)
        y = np.asarray(f["label"][:], dtype=np.int64)

    if X.shape[1:] != (8, 1000):
        raise ValueError(f"Unexpected data shape: {X.shape}")

    return X, y


@st.cache_data
def load_baseline_results():
    true_path = RESULTS_DIR / "y_true.npy"
    pred_path = RESULTS_DIR / "y_pred.npy"

    if not true_path.exists() or not pred_path.exists():
        return None, None

    return np.load(true_path), np.load(pred_path)


@st.cache_data
def load_warning_threshold():
    path = CONFIDENCE_DIR / "confidence_summary.csv"

    if not path.exists():
        return 0.825987

    df = pd.read_csv(path)
    return float(df.iloc[0]["threshold"])


def predict_one(model, signal):
    """Return the top gesture, softmax scores, and inference time."""
    tensor = torch.from_numpy(signal[None, :, :])

    with torch.inference_mode():
        start = time.perf_counter()
        logits = model(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000

        probabilities = torch.softmax(logits, dim=1)[0].numpy()

    predicted_class = int(np.argmax(probabilities))
    score = float(probabilities[predicted_class])

    return predicted_class, probabilities, score, elapsed_ms


def plot_emg(signal):
    time_axis = np.arange(signal.shape[1]) / 200.0

    fig, axes = plt.subplots(
        8, 1, figsize=(11, 9), sharex=True
    )

    for channel, ax in enumerate(axes):
        ax.plot(time_axis, signal[channel], linewidth=0.8)
        ax.set_ylabel(f"Ch {channel + 1}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Eight-channel preprocessed EMG window")
    fig.tight_layout()

    return fig


def main():
    st.title("EMG-Based Hand Gesture Recognition")
    st.caption(
        "Recorded-signal demonstration | Six gestures | "
        "Eight EMG channels | CPU inference"
    )

    try:
        model = load_model()
        X, y = load_test_data()
        y_true_saved, y_pred_saved = load_baseline_results()
        warning_threshold = load_warning_threshold()
    except Exception as exc:
        st.error(f"Could not load the project: {exc}")
        st.stop()

    st.info(
        "This demonstration uses recorded UCI EMG windows. "
        "It does not collect live sensor data. Each selection is "
        "classified by the model, and the best-guess gesture is "
        "always displayed."
    )

    # Overall baseline metrics, computed from saved test predictions.
    if y_true_saved is not None and y_pred_saved is not None:
        accuracy = accuracy_score(y_true_saved, y_pred_saved)
        macro_f1 = f1_score(
            y_true_saved, y_pred_saved, average="macro"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline test accuracy", f"{accuracy * 100:.2f}%")
        m2.metric("Baseline macro F1", f"{macro_f1 * 100:.2f}%")
        m3.metric("Held-out test windows", f"{len(y):,}")

    st.divider()

    st.sidebar.header("Recorded EMG browser")
    sample_index = st.sidebar.slider(
        "Select test window",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
    )

    st.sidebar.write(f"Input shape: {X[sample_index].shape}")
    st.sidebar.write("Sampling rate: 200 Hz")
    st.sidebar.write("Window duration: 5 seconds")
    st.sidebar.write(
        f"Warning threshold: {warning_threshold:.3f}"
    )

    signal = X[sample_index]
    true_label = int(y[sample_index])

    predicted_class, probabilities, score, elapsed_ms = predict_one(
        model, signal
    )

    prediction_correct = predicted_class == true_label
    low_score = score < warning_threshold

    st.subheader(f"Test window #{sample_index}")

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("#### Predicted gesture")

        st.markdown(f"## {GESTURES[predicted_class]}")

        st.metric("Maximum softmax score", f"{score * 100:.2f}%")

        if low_score:
            st.warning(
                "Low-confidence warning: the gesture prediction is "
                "still shown, but the model score is below the "
                "threshold selected using validation data."
            )
        else:
            st.success(
                "The model score is at or above the "
                "validation-selected threshold."
            )

        st.markdown("**Dataset reference label**")
        st.write(GESTURES[true_label])

        if prediction_correct:
            st.success("Prediction matches the reference label.")
        else:
            st.error("Prediction differs from the reference label.")

        st.caption(
            f"Single-window forward-pass time: {elapsed_ms:.2f} ms. "
            "Timing varies by machine and system load."
        )

    with right:
        st.markdown("#### Score for each gesture")

        scores_df = pd.DataFrame({
            "Gesture": GESTURES,
            "Model score": probabilities,
        }).set_index("Gesture")

        st.bar_chart(scores_df)

        st.caption(
            "Scores are produced by the model's softmax output and "
            "should not be interpreted as perfectly calibrated "
            "probabilities."
        )

    st.divider()

    st.subheader("Recorded EMG signals")
    st.write(
        "Each trace represents one of the eight channels after "
        "the filtering and normalization steps used in preprocessing."
    )

    fig = plot_emg(signal)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    with st.expander("Technical details"):
        st.write(f"Window index: {sample_index}")
        st.write(f"True class index: {true_label}")
        st.write(f"Predicted class index: {predicted_class}")
        st.write(f"Input tensor shape: {signal.shape}")
        st.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        st.write(
            "Evaluation protocol: subject-wise train, validation and "
            "test splits created during dataset preprocessing."
        )


if __name__ == "__main__":
    main()