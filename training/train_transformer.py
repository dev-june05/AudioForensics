"""
Fine-tuning pipeline for wav2vec 2.0 and HuBERT audio deepfake detectors.

These self-supervised speech models operate on raw 16 kHz waveforms and learn
representations that capture vocoder artifacts, phase discontinuities, and
unnatural prosody — complementing the spectrogram-based ResNet and LSTM.

Key differences from train.py (ResNet/LSTM):
  - Raw waveform input (no mel-spectrogram)
  - Differential learning rates (lower for pretrained backbone, higher for head)
  - Mixed precision (fp16) to handle the large transformer model efficiently
  - Gradient accumulation to simulate larger effective batch sizes
  - Smaller default batch size (8 vs 32) due to model memory footprint

Usage:
  # Fine-tune wav2vec2:
  python train_transformer.py --model wav2vec2 \
      --audio-dirs ~/partialspoof/extracted_audio/database/train/con_wav \
      --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt \
      --epochs 15 --batch-size 8 --num-workers 16

  # Fine-tune HuBERT:
  python train_transformer.py --model hubert \
      --audio-dirs ~/partialspoof/extracted_audio/database/train/con_wav \
      --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt \
      --epochs 15 --batch-size 8 --num-workers 16
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PartialSpoofDataset
from model import AudioWav2Vec2, AudioHuBERT

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = TRAINING_DIR / "outputs"

MODEL_FILENAMES = {
    "wav2vec2": "wav2vec2_audio_model.pth",
    "hubert": "hubert_audio_model.pth",
}

CLASS_NAMES = ["Real", "Spoofed"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def run_epoch_train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    grad_accum_steps: int = 4,
) -> Dict[str, float]:
    """Training loop with mixed precision and gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps  # Scale loss for accumulation

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Step optimizer every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    n = total if total > 0 else 1
    return {"loss": running_loss / n, "accuracy": correct / n}


def run_epoch_val(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validation loop with mixed precision."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    n = total if total > 0 else 1
    return {"loss": running_loss / n, "accuracy": correct / n}


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    """Collect predictions for classification report."""
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Collecting predictions", leave=False):
            inputs = inputs.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs[:, 1].cpu().numpy().tolist())

    return y_true, y_pred, y_probs

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend(); plt.grid(True); plt.savefig(path); plt.close()

def plot_accuracy_curve(train_acc, val_acc, path):
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve")
    plt.legend(); plt.grid(True); plt.savefig(path); plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix"); plt.savefig(path); plt.close()

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(architecture: str) -> torch.nn.Module:
    if architecture == "wav2vec2":
        return AudioWav2Vec2(num_classes=2)
    elif architecture == "hubert":
        return AudioHuBERT(num_classes=2)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'wav2vec2' or 'hubert'.")

# ---------------------------------------------------------------------------
# Optimizer with differential learning rates
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: torch.nn.Module,
    architecture: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with differential learning rates:
      - Pretrained backbone (transformer layers): lower LR to preserve learned features
      - New layers (attention pooling + classifier head): higher LR to learn from scratch
    """
    backbone_name = "wav2vec2" if architecture == "wav2vec2" else "hubert"

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(backbone_name):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ]

    return AdamW(param_groups)


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def train(
    architecture: str,
    audio_dirs: List[str],
    segment_labels: str,
    models_dir: str | None = None,
    batch_size: int = 8,
    num_epochs: int = 15,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    val_split: float = 0.2,
    num_workers: int = 0,
    patience: int = 5,
    grad_accum_steps: int = 4,
    augment: bool = True,
    verbose: bool = False,
) -> None:
    logger = setup_logging(verbose)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Fine-tuning %s Model", architecture.upper())
    logger.info("Device: %s", device)

    models_dir_path = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    models_dir_path.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir_path / MODEL_FILENAMES[architecture]

    # ---- Split dataset on file IDs (same logic as train.py) ----
    if str(segment_labels).endswith('.npy'):
        labels_dict = np.load(segment_labels, allow_pickle=True).item()
    else:
        labels_dict = {}
        with open(segment_labels, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels_dict[parts[1]] = 1

    all_files = list(labels_dict.keys())
    random.shuffle(all_files)

    split_idx = int(len(all_files) * (1.0 - val_split))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # ---- Build datasets in "raw" mode (returns raw waveform) ----
    train_dataset = PartialSpoofDataset(
        audio_dirs=audio_dirs,
        segment_labels_path=segment_labels,
        mode="raw",
        augment=augment,
        file_list=train_files,
    )
    val_dataset = PartialSpoofDataset(
        audio_dirs=audio_dirs,
        segment_labels_path=segment_labels,
        mode="raw",
        augment=False,
        file_list=val_files,
    )

    logger.info("Total files: %d (split %d train, %d val)", len(all_files), len(train_files), len(val_files))
    logger.info("Total train windows: %d", len(train_dataset))
    logger.info("Total val windows: %d", len(val_dataset))
    logger.info("Batch size: %d | Gradient accumulation: %d | Effective batch size: %d",
                batch_size, grad_accum_steps, batch_size * grad_accum_steps)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # ---- Build model ----
    logger.info("Loading pretrained %s model...", architecture.upper())
    model = _build_model(architecture).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info("Total params: %.1fM | Trainable: %.1fM | Frozen: %.1fM",
                total_params / 1e6, trainable_params / 1e6, frozen_params / 1e6)

    # ---- Optimizer with differential LR ----
    optimizer = _build_optimizer(model, architecture, backbone_lr, head_lr)
    criterion = CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # Mixed precision scaler
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    logger.info("Backbone LR: %.1e | Head LR: %.1e", backbone_lr, head_lr)
    logger.info("Mixed precision (fp16): %s", "Enabled" if use_amp else "Disabled")

    # ---- Training loop ----
    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        train_metrics = run_epoch_train(
            model, train_loader, optimizer, criterion, device, scaler, grad_accum_steps
        )
        val_metrics = run_epoch_val(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])

        logger.info(
            "Epoch %03d/%03d | Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f | LR: %.2e",
            epoch + 1, num_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            current_lr,
        )

        # Early stopping & checkpointing
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> Best model saved to %s (val_acc=%.4f)", best_model_path, best_val_acc)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # ---- Save training curves ----
    output_dir = DEFAULT_OUTPUTS_DIR / architecture
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(train_losses, val_losses, output_dir / "loss_curve.png")
    plot_accuracy_curve(train_accs, val_accs, output_dir / "accuracy_curve.png")

    # ---- Final evaluation on validation set ----
    logger.info("Loading best model for validation set report...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)

    y_true, y_pred, y_probs = collect_predictions(model, val_loader, device)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    logger.info("Classification Report:\n%s", report)

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, output_dir / "confusion_matrix.png")
    logger.info("Training complete. Outputs saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune wav2vec 2.0 or HuBERT for audio deepfake detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=["wav2vec2", "hubert"])
    parser.add_argument("--audio-dirs", nargs='+', required=True)
    parser.add_argument("--segment-labels", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Learning rate for pretrained transformer layers (default: 1e-5)")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="Learning rate for classification head (default: 1e-3)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4, effective batch = batch-size * grad-accum)")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train(
        architecture=args.model,
        audio_dirs=[Path(d) for d in args.audio_dirs],
        segment_labels=args.segment_labels,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        patience=args.patience,
        grad_accum_steps=args.grad_accum,
        augment=not args.no_augment,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
