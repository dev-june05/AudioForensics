"""
Model definitions for audio deepfake detection (training).

Contains two architectures:
  1. AudioResNet  — ResNet-18 with ImageNet transfer learning, adapted for
                    1-channel mel-spectrograms of *any* spatial size.
  2. AudioLSTM    — Bidirectional LSTM with attention-weighted pooling for
                    variable-length temporal mel-spectrogram sequences.

Both output 2 logits: index 0 → Real, index 1 → AI Generated.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# 1. ResNet-18 for spectrogram images (variable spatial size)
# ---------------------------------------------------------------------------

class AudioResNet(nn.Module):
    """
    ResNet-18 adapted for single-channel mel-spectrogram input and binary
    classification (Real vs AI Generated).

    Key design decisions:
      - conv1: 1 input channel instead of 3 (ImageNet weights averaged)
      - fc:    2 output classes instead of 1000
      - Accepts ANY spatial input size thanks to AdaptiveAvgPool2d(1,1)
        — no need to resize spectrograms to 224×224
      - ImageNet pretrained weights provide strong low-level feature
        extractors even for spectrogram data
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()

        # Load with or without ImageNet pretrained weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        # Adapt conv1: average the 3-channel pretrained weights into 1 channel
        if pretrained:
            original_weight = base.conv1.weight.data          # (64, 3, 7, 7)
            new_weight = original_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)

        base.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        if pretrained:
            base.conv1.weight.data = new_weight

        # Replace classifier head with 2 classes
        num_features = base.fc.in_features  # 512
        base.fc = nn.Linear(num_features, num_classes)

        # NOTE: base.avgpool is AdaptiveAvgPool2d(1, 1) — this allows
        # the model to accept ANY spatial input size, not just 224×224.
        # A mel spectrogram of shape (1, 128, T) for any T will work.

        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, N_MELS, T) — variable T is fine.
        Returns:
            (batch, num_classes) logits.
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# 2. Bidirectional LSTM with Attention Pooling
# ---------------------------------------------------------------------------

class AudioLSTM(nn.Module):
    """
    Bidirectional LSTM with learned attention-weighted pooling for
    classifying variable-length temporal mel-spectrogram sequences.

    Input shape:  (batch, seq_len, input_dim)   e.g. (B, T, 128)
    Output shape: (batch, num_classes)           e.g. (B, 2)

    Architecture:
      - Multi-layer bidirectional LSTM
      - Dropout between LSTM layers
      - Learned attention mechanism that scores every timestep
      - Attention-weighted sum of all LSTM outputs (not just the last)
      - FC classifier head

    Why attention pooling?
      If only part of the audio is fake (e.g., last 10 seconds of a
      30-second clip), the attention mechanism can learn to focus on
      the suspicious timesteps. Using only the last LSTM output would
      miss earlier anomalies, and mean-pooling would dilute them.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional → output is 2 * hidden_dim
        lstm_output_dim = hidden_dim * 2

        # Attention: scores each timestep with a scalar
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classifier head (same structure as before)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (batch, seq_len, input_dim) — variable seq_len is fine.
            lengths: (batch,) actual (unpadded) lengths for each sample.
                     Used to mask padding positions in attention.
                     If None, all positions are treated as valid.
        Returns:
            (batch, num_classes) logits.
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)

        # Compute attention scores for every timestep
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Mask padding positions (set to -inf before softmax)
        if lengths is not None:
            max_len = x.size(1)
            # Create mask: True where position >= actual length (i.e., padding)
            positions = torch.arange(max_len, device=x.device).unsqueeze(0)  # (1, seq_len)
            padding_mask = positions >= lengths.unsqueeze(1)  # (batch, seq_len)
            attn_scores = attn_scores.masked_fill(padding_mask, float("-inf"))

        # Softmax → attention weights
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum of all LSTM outputs
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out,                    # (batch, seq_len, hidden_dim * 2)
        ).squeeze(1)  # (batch, hidden_dim * 2)

        # Classify
        logits = self.classifier(context)  # (batch, num_classes)
        return logits
