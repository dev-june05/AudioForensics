"""
Model definitions for audio deepfake detection (training).

Contains four architectures:
  1. AudioResNet   — ResNet-18 with ImageNet transfer learning, adapted for
                     1-channel mel-spectrograms of *any* spatial size.
  2. AudioLSTM     — Bidirectional LSTM with attention-weighted pooling for
                     variable-length temporal mel-spectrogram sequences.
  3. AudioWav2Vec2 — Fine-tuned wav2vec 2.0 (facebook/wav2vec2-base) operating
                     directly on raw 16 kHz waveforms.
  4. AudioHuBERT   — Fine-tuned HuBERT (facebook/hubert-base-ls960) operating
                     directly on raw 16 kHz waveforms.

All output 2 logits: index 0 → Real, index 1 → AI Generated.
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


# ---------------------------------------------------------------------------
# 3. Wav2Vec 2.0 for raw waveform classification
# ---------------------------------------------------------------------------

class AudioWav2Vec2(nn.Module):
    """
    Fine-tuned wav2vec 2.0 for binary audio deepfake classification.

    Operates directly on raw 16 kHz waveforms — no mel-spectrogram needed.
    The model learns acoustic representations from the waveform that capture
    vocoder artifacts, phase discontinuities, and unnatural prosody that
    hand-crafted features (like mel-spectrograms) may miss.

    Architecture:
      - CNN Feature Extractor (FROZEN): Converts raw waveform to latent
        speech representations (~50 Hz frame rate, i.e. one frame per 20ms).
        Frozen because these low-level features generalize well.
      - Transformer Encoder (FINE-TUNED): 12 transformer layers that learn
        contextual representations. Fine-tuned to detect deepfake patterns.
      - Attention Pooling + Classifier Head (TRAINED FROM SCRATCH):
        Pools the variable-length transformer outputs into a fixed vector
        and classifies as Real vs Spoofed.

    Why attention pooling?
      A 30-second clip produces ~1500 transformer frames. If only part of
      the audio is fake, mean pooling dilutes the signal. Attention pooling
      lets the model focus on the suspicious frames.

    Input:  (batch, num_samples) — raw waveform, e.g. (B, 480000) for 30s
    Output: (batch, 2)           — classification logits
    """

    PRETRAINED_NAME = "facebook/wav2vec2-base"

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model

        name = model_name or self.PRETRAINED_NAME
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(name)

        # Freeze the CNN feature extractor — these low-level features
        # are already excellent and don't need fine-tuning.
        self.wav2vec2.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec2.config.hidden_size  # 768 for base

        # Attention pooling (same design philosophy as AudioLSTM)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_samples) — raw 16 kHz waveform.
        Returns:
            (batch, num_classes) logits.
        """
        # Normalize waveform (zero mean, unit variance per sample)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)

        # Extract contextual representations
        outputs = self.wav2vec2(x)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Attention-weighted pooling
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)         # (batch, seq_len)
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            hidden_states,              # (batch, seq_len, hidden_size)
        ).squeeze(1)                    # (batch, hidden_size)

        return self.classifier(context)


# ---------------------------------------------------------------------------
# 4. HuBERT for raw waveform classification
# ---------------------------------------------------------------------------

class AudioHuBERT(nn.Module):
    """
    Fine-tuned HuBERT for binary audio deepfake classification.

    HuBERT (Hidden-Unit BERT) learns speech representations via an offline
    clustering step followed by a BERT-like masked prediction objective.
    This gives it particularly strong phoneme-level representations, making
    it excellent at detecting partially spoofed audio where only certain
    phonemes or words have been replaced.

    Architecture is identical to AudioWav2Vec2 (frozen CNN + fine-tuned
    transformer + attention pooling + classifier), but uses the HuBERT
    backbone instead of wav2vec 2.0.

    Input:  (batch, num_samples) — raw waveform, e.g. (B, 480000) for 30s
    Output: (batch, 2)           — classification logits
    """

    PRETRAINED_NAME = "facebook/hubert-base-ls960"

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import HubertModel

        name = model_name or self.PRETRAINED_NAME
        self.hubert = HubertModel.from_pretrained(name)

        # Freeze the CNN feature extractor
        self.hubert.feature_extractor._freeze_parameters()

        hidden_size = self.hubert.config.hidden_size  # 768 for base

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_samples) — raw 16 kHz waveform.
        Returns:
            (batch, num_classes) logits.
        """
        # Normalize waveform
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)

        # Extract contextual representations
        outputs = self.hubert(x)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Attention-weighted pooling
        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            hidden_states,
        ).squeeze(1)

        return self.classifier(context)
