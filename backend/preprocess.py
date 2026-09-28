"""
Audio preprocessing module for the Audio Deepfake Detection pipeline.

Returns a list of WindowData objects from a single audio file, where each
window represents a 30-second contiguous chunk of the original file.

Steps performed:
  1. Load audio with librosa (16 kHz mono)
  2. DO NOT trim silence.
  3. Divide into 30-second windows.
  4. For each window:
     a. Detect silence based on short-time RMS energy.
     b. If silent, flag as is_silent=True.
     c. If not silent, zero-pad to 30 seconds if it's a short final window.
     d. Compute mel spectrogram -> log dB -> normalize.
     e. Generate ResNet tensor (1, N_MELS, T)
     f. Generate LSTM tensor (1, T, N_MELS)
     g. Render spectrogram to a base64 PNG.
"""

import base64
import io
from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Union

import librosa
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    POWER,
    SAMPLE_RATE,
    SILENCE_THRESHOLD_DB,
    WINDOW_DURATION_SEC,
    WINDOW_NUM_SAMPLES,
)


@dataclass
class WindowData:
    """Container for preprocessing outputs of a single 30-second window."""
    start_time_sec: float
    end_time_sec: float
    actual_duration_sec: float
    is_silent: bool
    resnet_tensor: Optional[torch.Tensor] = None
    lstm_tensor: Optional[torch.Tensor] = None
    spectrogram_base64: Optional[str] = None


def _is_silent(y: np.ndarray) -> bool:
    """
    Energy-based silence detector.
    Returns True if the maximum energy in the window is below the silence threshold.
    """
    if len(y) == 0:
        return True
    # Calculate short-time RMS energy
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    # Convert to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    # Check if ANY frame has meaningful energy
    # Note: amplitude_to_db with ref=np.max sets peak to 0 dB, so silence is negative.
    # To detect silence across the whole clip based on an absolute threshold, we can
    # instead compare peak amplitude to a fixed tiny value.
    # Alternatively, librosa.effects.trim does: y, index = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    # We can use a similar logic without trimming:
    # Just look at the max amplitude.
    peak_amplitude = np.max(np.abs(y))
    # threshold for silence
    threshold = 10 ** (-SILENCE_THRESHOLD_DB / 20.0)
    return float(peak_amplitude) < threshold


def _pad_to_window(y: np.ndarray) -> np.ndarray:
    """Zero-pad short audio to match exactly WINDOW_NUM_SAMPLES."""
    if len(y) < WINDOW_NUM_SAMPLES:
        y = np.pad(y, (0, WINDOW_NUM_SAMPLES - len(y)), mode='constant')
    return y


def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
    )


def _to_log_scale(mel_spec: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(mel_spec, ref=np.max)


def _normalize(mel_db: np.ndarray) -> np.ndarray:
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std <= 0:
        std = 1.0
    return (mel_db - mean) / std


def _spectrogram_to_base64(mel_db: np.ndarray, start_time: float, end_time: float) -> str:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    img = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel Spectrogram ({start_time:.1f}s - {end_time:.1f}s)", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _process_window(y_window: np.ndarray, start_time: float, end_time: float, actual_duration: float) -> WindowData:
    if _is_silent(y_window):
        return WindowData(
            start_time_sec=start_time,
            end_time_sec=end_time,
            actual_duration_sec=actual_duration,
            is_silent=True,
        )

    # Pad if short
    y_padded = _pad_to_window(y_window)

    mel_spec = _compute_mel_spectrogram(y_padded)
    mel_db = _to_log_scale(mel_spec)
    mel_norm = _normalize(mel_db)

    # ResNet tensor: (1, N_MELS, T)
    resnet_tensor = torch.from_numpy(mel_norm).float().unsqueeze(0)

    # LSTM tensor: (1, T, N_MELS)
    lstm_seq = mel_norm.T
    lstm_tensor = torch.from_numpy(lstm_seq).float().unsqueeze(0)

    spec_b64 = _spectrogram_to_base64(mel_db, start_time, end_time)

    return WindowData(
        start_time_sec=start_time,
        end_time_sec=end_time,
        actual_duration_sec=actual_duration,
        is_silent=False,
        resnet_tensor=resnet_tensor,
        lstm_tensor=lstm_tensor,
        spectrogram_base64=spec_b64,
    )


def _waveform_to_windows(y: np.ndarray) -> List[WindowData]:
    """Chunk the waveform into 30-second windows and process each."""
    total_samples = len(y)
    if total_samples == 0:
        return [WindowData(0.0, 0.0, 0.0, is_silent=True)]

    windows = []
    for i in range(0, total_samples, WINDOW_NUM_SAMPLES):
        y_chunk = y[i:i + WINDOW_NUM_SAMPLES]
        start_time = i / SAMPLE_RATE
        actual_duration = len(y_chunk) / SAMPLE_RATE
        end_time = start_time + actual_duration
        
        window_data = _process_window(y_chunk, start_time, end_time, actual_duration)
        windows.append(window_data)

    return windows


def preprocess_audio(file: Union[BinaryIO, bytes, str]) -> List[WindowData]:
    """
    Preprocess an audio source and return a list of WindowData.
    """
    try:
        if isinstance(file, bytes):
            file = io.BytesIO(file)
        elif isinstance(file, str):
            y, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)
            return _waveform_to_windows(y)
    
        raw = file.read() if hasattr(file, "read") else file
        if isinstance(raw, bytes):
            file = io.BytesIO(raw)
        y, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)
        return _waveform_to_windows(y)
    except Exception as e:
        raise ValueError(f"Could not decode audio file. ({str(e)})")
