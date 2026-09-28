"""
Dual-model loader and ensemble inference for audio deepfake detection.

Manages two models:
  1. ResNet-18  — classifies mel-spectrogram images (1, N_MELS, T)
  2. LSTM       — classifies temporal mel sequences  (1, T, N_MELS)

At startup both .pth files are loaded into memory. The predict() function
runs both models, applies softmax, and computes a weighted ensemble.

Class index convention:
  0 → Real
  1 → Spoofed
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import (
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    LSTM_HIDDEN_DIM,
    LSTM_INPUT_DIM,
    LSTM_NUM_CLASSES,
    LSTM_NUM_LAYERS,
    RESNET_NUM_CLASSES,
    get_ensemble_weights,
)
from preprocess import WindowData

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Singleton model instances
# ---------------------------------------------------------------------------

_RESNET_MODEL: Optional[nn.Module] = None
_LSTM_MODEL: Optional[nn.Module] = None


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WindowPrediction:
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    is_silent: bool
    status: str  # "SILENT", "ANALYZED"
    ai_probability: float
    prediction: str  # "Real", "Spoofed", "Uncertain", "N/A"
    resnet_prob: float = 0.0
    lstm_prob: float = 0.0
    spectrogram_base64: Optional[str] = None


@dataclass
class FilePrediction:
    overall_status: str  # "NO_AUDIO", "MIXED / SUSPICIOUS", "Real", "Spoofed"
    overall_ai_probability: float
    total_duration_sec: float
    windows: List[WindowPrediction]

# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def _build_resnet() -> nn.Module:
    """
    ResNet-18: 1-channel input, 2-class output.
    Accepts variable spatial input via AdaptiveAvgPool2d(1,1).
    No pretrained weights at inference — architecture must match training.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, RESNET_NUM_CLASSES)
    return model


class _AudioLSTM(nn.Module):
    """
    Bidirectional LSTM with attention pooling, matching training/model.py AudioLSTM.
    """

    def __init__(
        self,
        input_dim: int = LSTM_INPUT_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = LSTM_NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2

        # Attention mechanism (must match training/model.py)
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        # Attention scores
        attn_scores = self.attention(lstm_out).squeeze(-1)

        # Mask padding if lengths provided
        if lengths is not None:
            max_len = x.size(1)
            positions = torch.arange(max_len, device=x.device).unsqueeze(0)
            padding_mask = positions >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(padding_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            lstm_out,
        ).squeeze(1)

        return self.classifier(context)


# ---------------------------------------------------------------------------
# State-dict cleaning (handles "model." prefix from training wrapper)
# ---------------------------------------------------------------------------

def _clean_state_dict(raw: dict) -> dict:
    """Strip optional 'model.' prefix from keys."""
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    cleaned = {}
    for k, v in raw.items():
        cleaned[k[len("model."):] if k.startswith("model.") else k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Load helpers (called once at startup)
# ---------------------------------------------------------------------------

def load_resnet(path: str) -> nn.Module:
    """Load and cache the ResNet model from a .pth checkpoint."""
    global _RESNET_MODEL
    if _RESNET_MODEL is not None:
        return _RESNET_MODEL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ResNet weights not found at {path}")

    checkpoint = torch.load(p, map_location=DEVICE)
    state_dict = _clean_state_dict(checkpoint)

    model = _build_resnet()
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    _RESNET_MODEL = model
    return _RESNET_MODEL


def load_lstm(path: str) -> nn.Module:
    """Load and cache the LSTM model from a .pth checkpoint."""
    global _LSTM_MODEL
    if _LSTM_MODEL is not None:
        return _LSTM_MODEL

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LSTM weights not found at {path}")

    checkpoint = torch.load(p, map_location=DEVICE)
    state_dict = _clean_state_dict(checkpoint)

    model = _AudioLSTM()
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    _LSTM_MODEL = model
    return _LSTM_MODEL


def models_loaded() -> dict:
    """Return which models are currently loaded."""
    return {
        "resnet": _RESNET_MODEL is not None,
        "lstm": _LSTM_MODEL is not None,
    }


# ---------------------------------------------------------------------------
# Inference: dual-model ensemble
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict(windows: List[WindowData]) -> FilePrediction:
    """
    Run both models on all non-silent windows and compute a duration-weighted ensemble.
    """
    if _RESNET_MODEL is None and _LSTM_MODEL is None:
        raise RuntimeError("No models loaded. Call load_resnet() and/or load_lstm() at startup.")

    window_predictions: List[WindowPrediction] = []
    
    total_analyzed_duration = 0.0
    weighted_ai_prob_sum = 0.0
    total_file_duration = 0.0

    for w in windows:
        total_file_duration += w.actual_duration_sec
        
        if w.is_silent:
            window_predictions.append(WindowPrediction(
                start_time_sec=w.start_time_sec,
                end_time_sec=w.end_time_sec,
                duration_sec=w.actual_duration_sec,
                is_silent=True,
                status="SILENT",
                ai_probability=0.0,
                prediction="N/A",
            ))
            continue

        resnet_probs = None
        lstm_probs = None

        if _RESNET_MODEL is not None and w.resnet_tensor is not None:
            batch = w.resnet_tensor.unsqueeze(0).to(DEVICE)
            logits = _RESNET_MODEL(batch)
            resnet_probs = torch.softmax(logits, dim=1).squeeze(0)

        if _LSTM_MODEL is not None and w.lstm_tensor is not None:
            batch = w.lstm_tensor.unsqueeze(0).to(DEVICE)
            seq_len = batch.size(1)
            lengths = torch.tensor([seq_len], device=DEVICE)
            logits = _LSTM_MODEL(batch, lengths=lengths)
            lstm_probs = torch.softmax(logits, dim=1).squeeze(0)

        # Ensemble
        if resnet_probs is not None and lstm_probs is not None:
            resnet_w, lstm_w = get_ensemble_weights()
            ensemble_probs = resnet_w * resnet_probs + lstm_w * lstm_probs
        elif resnet_probs is not None:
            ensemble_probs = resnet_probs
        else:
            ensemble_probs = lstm_probs
            
        ai_prob = float(ensemble_probs[1].item())
        resnet_ai_prob = float(resnet_probs[1].item()) if resnet_probs is not None else 0.0
        lstm_ai_prob = float(lstm_probs[1].item()) if lstm_probs is not None else 0.0
        
        # Classification
        if ai_prob >= CONFIDENCE_THRESHOLD:
            pred_label = "Spoofed"
        elif ai_prob <= (1 - CONFIDENCE_THRESHOLD):
            pred_label = "Real"
        else:
            pred_label = "Uncertain"

        window_predictions.append(WindowPrediction(
            start_time_sec=w.start_time_sec,
            end_time_sec=w.end_time_sec,
            duration_sec=w.actual_duration_sec,
            is_silent=False,
            status="ANALYZED",
            ai_probability=ai_prob,
            prediction=pred_label,
            resnet_prob=resnet_ai_prob,
            lstm_prob=lstm_ai_prob,
            spectrogram_base64=w.spectrogram_base64,
        ))

        total_analyzed_duration += w.actual_duration_sec
        weighted_ai_prob_sum += (ai_prob * w.actual_duration_sec)

    if total_analyzed_duration == 0:
        overall_status = "NO_AUDIO"
        overall_ai_prob = 0.0
    else:
        overall_ai_prob = weighted_ai_prob_sum / total_analyzed_duration
        if overall_ai_prob >= CONFIDENCE_THRESHOLD:
            overall_status = "Spoofed"
        elif overall_ai_prob <= (1 - CONFIDENCE_THRESHOLD):
            overall_status = "Real"
        else:
            overall_status = "MIXED / SUSPICIOUS"

    return FilePrediction(
        overall_status=overall_status,
        overall_ai_probability=overall_ai_prob,
        total_duration_sec=total_file_duration,
        windows=window_predictions,
    )
