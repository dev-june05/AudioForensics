"""
Training pipeline for the Audio Deepfake Detection models.

Trains either a ResNet-18 (spectrogram image classifier) or a bidirectional
LSTM (temporal sequence classifier) depending on the --model flag.

Improvements over the original pipeline:
  - 30-Second Fixed Windows (no OOM, fast batching)
  - Raw Waveform support (Wav2Vec2 / HuBERT)
  - Zero Data Leakage via file-ID splitting
  - Cosine annealing learning rate scheduler
  - Early stopping with configurable patience
  - Per-architecture output directories (no overwriting between models)

Usage:
  python train.py --model resnet --audio-dirs ... --segment-labels ...
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import PartialSpoofDataset
from model import AudioLSTM, AudioResNet

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = TRAINING_DIR / "outputs"

MODEL_FILENAMES = {
    "resnet": "resnet_audio_model.pth",
    "lstm": "lstm_audio_model.pth",
    "wav2vec2": "wav2vec2_audio_model.pth",
    "hubert": "hubert_audio_model.pth",
}

CLASS_NAMES = ["Real", "Spoofed"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_train_logging(verbose: bool = False) -> logging.Logger:
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
    architecture: str,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
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
    architecture: str,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
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
    architecture: str,
) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs[:, 1].cpu().numpy().tolist())

    return y_true, y_pred, y_probs

# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses: List[float], val_losses: List[float], path: Path) -> None:
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_accuracy_curve(train_acc: List[float], val_acc: List[float], path: Path) -> None:
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()

def generate_classification_report(y_true: List[int], y_pred: List[int], y_probs: List[float], classes: List[str], path: Path, logger: logging.Logger) -> None:
    report = classification_report(y_true, y_pred, target_names=classes)
    logger.info("Classification Report:\n%s", report)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def _build_model(architecture: str) -> torch.nn.Module:
    if architecture == "resnet":
        return AudioResNet(num_classes=2, pretrained=True)
    elif architecture == "lstm":
        return AudioLSTM(num_classes=2)
    elif architecture == "wav2vec2":
        from model import AudioWav2Vec2
        return AudioWav2Vec2(num_classes=2)
    elif architecture == "hubert":
        from model import AudioHuBERT
        return AudioHuBERT(num_classes=2)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def train(
    architecture: str,
    audio_dirs: List[str],
    segment_labels: str,
    models_dir: str | None = None,
    batch_size: int = 32,
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    num_workers: int = 0,
    patience: int = 5,
    augment: bool = True,
    verbose: bool = False,
) -> None:
    logger = setup_train_logging(verbose)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training %s Model", architecture.upper())
    logger.info("Device: %s", device)

    models_dir_path = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    models_dir_path.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir_path / MODEL_FILENAMES.get(architecture, f"{architecture}_model.pth")

    # Split dataset cleanly on file IDs
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

    train_dataset = PartialSpoofDataset(
        audio_dirs=audio_dirs,
        segment_labels_path=segment_labels,
        mode=architecture,
        augment=augment,
        file_list=train_files
    )
    val_dataset = PartialSpoofDataset(
        audio_dirs=audio_dirs,
        segment_labels_path=segment_labels,
        mode=architecture,
        augment=False,
        file_list=val_files
    )

    logger.info("Total train files: %d (split %d train, %d val)", len(all_files), len(train_files), len(val_files))
    logger.info("Total train windows: %d", len(train_dataset))
    logger.info("Total val windows: %d", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = _build_model(architecture).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch_train(model, train_loader, optimizer, criterion, device, architecture)
        val_metrics = run_epoch_val(model, val_loader, criterion, device, architecture)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_loss = train_metrics["loss"]
        train_acc = train_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info(
            "Epoch %03d/%03d | Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f | LR: %.2e",
            epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, current_lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> Best model saved to %s (val_acc=%.4f)", best_model_path, val_acc)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)

    outputs_dir = DEFAULT_OUTPUTS_DIR / architecture
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    plot_loss_curve(train_losses, val_losses, outputs_dir / "loss_curve.png")
    plot_accuracy_curve(train_accuracies, val_accuracies, outputs_dir / "accuracy_curve.png")

    logger.info("Loading best model for validation set report...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    y_true, y_pred, y_probs = collect_predictions(model, val_loader, device, architecture)
    
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, outputs_dir / "confusion_matrix.png")
    generate_classification_report(y_true, y_pred, y_probs, CLASS_NAMES, outputs_dir / "classification_report.txt", logger)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detection Models")
    parser.add_argument("--model", type=str, choices=["resnet", "lstm", "wav2vec2", "hubert"], required=True)
    parser.add_argument("--audio-dirs", nargs='+', required=True, help="Directories containing audio files")
    parser.add_argument("--segment-labels", type=str, required=True, help="Path to train_seglab_0.64.npy")
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        architecture=args.model,
        audio_dirs=args.audio_dirs,
        segment_labels=args.segment_labels,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        patience=args.patience,
        augment=not args.no_augment,
        verbose=args.verbose,
    )
