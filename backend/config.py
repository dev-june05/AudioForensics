"""
Centralized configuration for the Audio Deepfake Detection backend.

All model paths, feature dimensions, ensemble weights, and preprocessing
constants live here so that every module imports from one place.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# -----------------------------------------------------------------------------
# Model file paths
# -----------------------------------------------------------------------------

RESNET_MODEL_FILENAME = "resnet_audio_model.pth"
LSTM_MODEL_FILENAME = "lstm_audio_model.pth"

RESNET_MODEL_PATH = MODELS_DIR / RESNET_MODEL_FILENAME
LSTM_MODEL_PATH = MODELS_DIR / LSTM_MODEL_FILENAME

# Legacy single-model path (backward compat)
LEGACY_MODEL_PATH = MODELS_DIR / "audio_model.pth"

# -----------------------------------------------------------------------------
# Audio preprocessing constants (shared with training/dataset.py)
# -----------------------------------------------------------------------------

# Target sample rate in Hz
SAMPLE_RATE = 16_000

# We process audio in fixed 30-second contiguous windows.
WINDOW_DURATION_SEC = 30.0
WINDOW_NUM_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_SEC)

# Mel spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
POWER = 2.0

# Silence detection threshold (dB relative to peak)
SILENCE_THRESHOLD_DB = 20.0

# -----------------------------------------------------------------------------
# LSTM feature dimensions
# -----------------------------------------------------------------------------

# The LSTM receives the mel-spectrogram at its natural time resolution.
# Time steps vary with audio length: T = 1 + floor(num_samples / HOP_LENGTH)
LSTM_INPUT_DIM = N_MELS        # 128 features per time step
LSTM_HIDDEN_DIM = 128          # hidden state size
LSTM_NUM_LAYERS = 2            # stacked LSTM layers
LSTM_NUM_CLASSES = 2           # Real / AI Generated

# ResNet classes
RESNET_NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# Ensemble weights  (must sum to 1.0)
# Adjust these to trust one model more than the other.
# -----------------------------------------------------------------------------
import json

def get_ensemble_weights():
    """Dynamically load ensemble weights from JSON, fallback to 0.5/0.5."""
    weights_file = MODELS_DIR / "ensemble_weights.json"
    if weights_file.exists():
        try:
            with open(weights_file, "r") as f:
                data = json.load(f)
            return data.get("resnet_weight", 0.5), data.get("lstm_weight", 0.5)
        except Exception:
            pass
    return 0.5, 0.5

RESNET_WEIGHT, LSTM_WEIGHT = get_ensemble_weights()

# -----------------------------------------------------------------------------
# Confidence threshold
# Predictions below this threshold are marked as "Uncertain".
# -----------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.70

# -----------------------------------------------------------------------------
# Class labels  (index 0 → Real, index 1 → AI Generated)
# -----------------------------------------------------------------------------

CLASS_LABELS = ("Real", "AI Generated")
