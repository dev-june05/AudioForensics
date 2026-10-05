"""
Evaluate trained models on the PartialSpoof dev/eval sets.

Produces proper baseline metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Equal Error Rate (EER) — the gold standard for spoofing detection
  - Confusion Matrix (saved as PNG)
  - DET Curve (saved as PNG)
  - Full classification report (saved as TXT)

Usage:
  # Evaluate ResNet on the dev set:
  python evaluate.py --model resnet \
      --audio-dirs ~/partialspoof/extracted_audio/database/dev/con_wav \
      --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt

  # Evaluate ALL models at once for comparison:
  python evaluate.py --model all \
      --audio-dirs ~/partialspoof/extracted_audio/database/dev/con_wav \
      --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt

  # Evaluate on the eval set:
  python evaluate.py --model resnet \
      --audio-dirs ~/partialspoof/extracted_audio/database/eval/con_wav \
      --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt
"""

import argparse
import json
import logging
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
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PartialSpoofDataset
from model import AudioLSTM, AudioResNet, AudioWav2Vec2, AudioHuBERT

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = TRAINING_DIR / "eval_outputs"

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
# EER Calculation
# ---------------------------------------------------------------------------

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    EER is the point where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    Lower EER = better model. State-of-the-art deepfake detectors aim for EER < 1%.
    
    Returns:
        (eer, threshold): The EER value and the threshold at which it occurs.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the point where FPR and FNR are closest (EER)
    abs_diff = np.abs(fpr - fnr)
    eer_idx = np.argmin(abs_diff)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(architecture: str, model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model from disk."""
    if architecture == "resnet":
        model = AudioResNet(num_classes=2, pretrained=False)
    elif architecture == "lstm":
        model = AudioLSTM(num_classes=2)
    elif architecture == "wav2vec2":
        model = AudioWav2Vec2(num_classes=2)
    elif architecture == "hubert":
        model = AudioHuBERT(num_classes=2)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference and collect predictions.
    
    Returns:
        y_true: Ground truth labels
        y_pred: Predicted labels (argmax)
        y_scores: Predicted probability of class 1 (Spoofed)
    """
    y_true_list = []
    y_pred_list = []
    y_scores_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=True):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            y_true_list.extend(labels.cpu().numpy().tolist())
            y_pred_list.extend(preds.cpu().numpy().tolist())
            y_scores_list.extend(probs[:, 1].cpu().numpy().tolist())
    
    return (
        np.array(y_true_list),
        np.array(y_pred_list),
        np.array(y_scores_list),
    )

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes, path):
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_det_curve(y_true, y_scores, eer, path):
    """
    Save a Detection Error Tradeoff (DET) curve.
    DET curves are the standard visualization for speaker/spoofing detection.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr * 100, fnr * 100, linewidth=2, label=f"DET Curve (EER={eer*100:.2f}%)")
    ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8, alpha=0.5, label="EER Line")
    ax.scatter([eer * 100], [eer * 100], color='red', s=100, zorder=5, label=f"EER Point")
    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title("Detection Error Tradeoff (DET) Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(fpr * 100) * 1.1])
    ax.set_ylim([0, max(fnr * 100) * 1.1])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison_bar_chart(results: Dict[str, Dict], path: Path):
    """
    Create a side-by-side bar chart comparing metrics across models.
    """
    model_names = list(results.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "EER"]
    
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    
    for i, (name, res) in enumerate(results.items()):
        values = [
            res["accuracy"] * 100,
            res["precision"] * 100,
            res["recall"] * 100,
            res["f1"] * 100,
            res["eer"] * 100,  # Note: lower is better for EER
        ]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name.upper(),
                      color=colors[i % len(colors)], alpha=0.85)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Model Comparison (Dev Set Evaluation)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add note about EER
    ax.annotate("↓ Lower EER = Better", xy=(4, 5), fontsize=9, color='gray', ha='center')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Single Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_single_model(
    architecture: str,
    audio_dirs: List[str],
    segment_labels: str,
    models_dir: Path,
    output_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    logger: logging.Logger = None,
) -> Dict:
    """Evaluate a single model and save all results."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check model file exists
    model_filename = MODEL_FILENAMES.get(architecture)
    if model_filename is None:
        logger.error(f"Unknown architecture: {architecture}")
        return None
        
    model_path = models_dir / model_filename
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path} — skipping {architecture}")
        return None
    
    # Create output directory
    model_output_dir = output_dir / architecture
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Evaluating %s", architecture.upper())
    logger.info("=" * 70)
    logger.info("Model: %s", model_path)
    logger.info("Device: %s", device)
    
    # Load dataset
    logger.info("Loading evaluation dataset...")
    dataset_mode = "raw" if architecture in ["wav2vec2", "hubert"] else architecture
    dataset = PartialSpoofDataset(
        audio_dirs=[Path(d) for d in audio_dirs],
        segment_labels_path=segment_labels,
        mode=dataset_mode,
        augment=False,
    )
    logger.info("Total evaluation windows: %d", len(dataset))
    
    if len(dataset) == 0:
        logger.error("No evaluation windows found! Check your audio-dirs and segment-labels paths.")
        return None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Load model
    logger.info("Loading trained model...")
    model = load_model(architecture, model_path, device)
    
    # Run inference
    logger.info("Running inference...")
    y_true, y_pred, y_scores = run_inference(
        model, loader, device, desc=f"Eval {architecture.upper()}"
    )
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    eer, eer_threshold = compute_eer(y_true, y_scores)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    logger.info("Classification Report:\n%s", report)
    logger.info("Equal Error Rate (EER): %.4f (%.2f%%)", eer, eer * 100)
    logger.info("EER Threshold: %.4f", eer_threshold)
    logger.info("Overall Accuracy: %.4f (%.2f%%)", accuracy, accuracy * 100)
    
    # Save classification report
    report_path = model_output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {architecture.upper()}\n")
        f.write(f"Model File: {model_path}\n")
        f.write(f"Evaluation Windows: {len(dataset)}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(report)
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Equal Error Rate (EER): {eer:.4f} ({eer * 100:.2f}%)\n")
        f.write(f"EER Threshold: {eer_threshold:.4f}\n")
    
    # Save confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        model_output_dir / "confusion_matrix.png"
    )
    logger.info("Saved confusion matrix to %s", model_output_dir / "confusion_matrix.png")
    
    # Save DET curve
    plot_det_curve(
        y_true, y_scores, eer,
        model_output_dir / "det_curve.png"
    )
    logger.info("Saved DET curve to %s", model_output_dir / "det_curve.png")
    
    # Save metrics as JSON for easy programmatic access
    metrics = {
        "architecture": architecture,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "total_windows": len(dataset),
        "class_distribution": {
            "real": int((y_true == 0).sum()),
            "spoofed": int((y_true == 1).sum()),
        },
    }
    
    metrics_path = model_output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved metrics to %s", metrics_path)
    
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on dev/eval sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["resnet", "lstm", "wav2vec2", "hubert", "all"],
        help="Which model to evaluate. Use 'all' to evaluate all available models.",
    )
    parser.add_argument(
        "--audio-dirs", nargs='+', required=True,
        help="Directories containing audio files for the split being evaluated.",
    )
    parser.add_argument(
        "--segment-labels", type=str, required=True,
        help="Path to the protocol .txt file for the split being evaluated.",
    )
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    
    # Determine which models to evaluate
    if args.model == "all":
        architectures = ["resnet", "lstm", "wav2vec2", "hubert"]
    else:
        architectures = [args.model]
    
    all_results = {}
    
    for arch in architectures:
        result = evaluate_single_model(
            architecture=arch,
            audio_dirs=args.audio_dirs,
            segment_labels=args.segment_labels,
            models_dir=models_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
        )
        if result is not None:
            all_results[arch] = result
    
    # If we evaluated multiple models, generate a comparison
    if len(all_results) > 1:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 70)
        
        header = f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'EER':>10}"
        logger.info(header)
        logger.info("-" * len(header))
        
        for name, res in all_results.items():
            logger.info(
                f"{name.upper():<12} {res['accuracy']*100:>9.2f}% {res['precision']*100:>9.2f}% "
                f"{res['recall']*100:>9.2f}% {res['f1']*100:>9.2f}% {res['eer']*100:>9.2f}%"
            )
        
        # Find best model
        best_eer_model = min(all_results, key=lambda k: all_results[k]['eer'])
        best_acc_model = max(all_results, key=lambda k: all_results[k]['accuracy'])
        
        logger.info("")
        logger.info("Best by EER:      %s (%.2f%%)", best_eer_model.upper(), all_results[best_eer_model]['eer'] * 100)
        logger.info("Best by Accuracy: %s (%.2f%%)", best_acc_model.upper(), all_results[best_acc_model]['accuracy'] * 100)
        
        # Generate comparison bar chart
        comparison_path = output_dir / "model_comparison.png"
        plot_comparison_bar_chart(all_results, comparison_path)
        logger.info("Saved comparison chart to %s", comparison_path)
        
        # Save combined results JSON
        combined_path = output_dir / "comparison_results.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        logger.info("Saved combined results to %s", combined_path)
    
    elif len(all_results) == 1:
        name, res = list(all_results.items())[0]
        logger.info("")
        logger.info("=" * 70)
        logger.info("FINAL RESULTS: %s", name.upper())
        logger.info("  Accuracy:  %.2f%%", res['accuracy'] * 100)
        logger.info("  Precision: %.2f%%", res['precision'] * 100)
        logger.info("  Recall:    %.2f%%", res['recall'] * 100)
        logger.info("  F1-Score:  %.2f%%", res['f1'] * 100)
        logger.info("  EER:       %.2f%%", res['eer'] * 100)
        logger.info("=" * 70)
    
    else:
        logger.error("No models were successfully evaluated.")


if __name__ == "__main__":
    main()
