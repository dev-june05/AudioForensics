"""
Learned Ensemble Weighting

Extracts logits from trained ResNet and LSTM models on the validation set,
and optimizes a learned blending weight to minimize cross-entropy loss.
Saves the normalized weights to ensemble_weights.json.
"""

import argparse
import json
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from dataset import AudioDataset, collate_variable_length
from model import AudioLSTM, AudioResNet

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    return logging.getLogger(__name__)

class EnsembleWeighter(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with 0s (which softmaxes to 0.5, 0.5)
        self.weights = nn.Parameter(torch.zeros(2))
        
    def forward(self, resnet_logits, lstm_logits):
        w = torch.softmax(self.weights, dim=0)
        return w[0] * resnet_logits + w[1] * lstm_logits

def _clean_state_dict(raw: dict) -> dict:
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    # Do not strip "model." because AudioResNet actually uses self.model
    return raw

def main():
    parser = argparse.ArgumentParser(description="Learn optimal ensemble weights.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_root = Path(args.dataset)
    models_dir = Path(args.models_dir)
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        train_dir = dataset_root

    resnet_path = models_dir / "resnet_audio_model.pth"
    lstm_path = models_dir / "lstm_audio_model.pth"
    
    if not resnet_path.exists() or not lstm_path.exists():
        logger.error("Both resnet and lstm models must exist in models/ to train ensemble.")
        return

    # Reproducibility to match train.py split exactly
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("Loading validation datasets...")
    # Get all labels
    try:
        dummy_dataset = AudioDataset(train_dir, mode="resnet", augment=False)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
        
    all_labels = dummy_dataset.get_labels()
    
    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=seed)
    _, val_indices = next(splitter.split(range(len(all_labels)), all_labels))
    val_indices = val_indices.tolist()

    # Create Subsets for validation
    resnet_ds = Subset(AudioDataset(train_dir, mode="resnet", augment=False), val_indices)
    lstm_ds = Subset(AudioDataset(train_dir, mode="lstm", augment=False), val_indices)

    resnet_loader = DataLoader(
        resnet_ds, batch_size=args.batch_size, shuffle=False, 
        collate_fn=partial(collate_variable_length, mode="resnet")
    )
    lstm_loader = DataLoader(
        lstm_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate_variable_length, mode="lstm")
    )

    logger.info("Loading models...")
    resnet = AudioResNet(num_classes=2, pretrained=False)
    resnet.load_state_dict(_clean_state_dict(torch.load(resnet_path, map_location=device)))
    resnet.to(device).eval()

    lstm = AudioLSTM(input_dim=128, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3)
    lstm.load_state_dict(_clean_state_dict(torch.load(lstm_path, map_location=device)))
    lstm.to(device).eval()

    logger.info("Extracting logits from validation set...")
    resnet_logits_list = []
    lstm_logits_list = []
    labels_list = []

    with torch.no_grad():
        for (r_batch, r_labels), (l_batch_data) in zip(resnet_loader, lstm_loader):
            l_batch, l_labels, lengths = l_batch_data
            if not torch.equal(r_labels, l_labels):
                logger.error("Label mismatch between resnet and lstm dataloaders!")
                return

            r_batch = r_batch.to(device)
            r_out = resnet(r_batch)
            resnet_logits_list.append(r_out)

            l_batch = l_batch.to(device)
            lengths = lengths.to(device)
            l_out = lstm(l_batch, lengths=lengths)
            lstm_logits_list.append(l_out)

            labels_list.append(r_labels.to(device))

    resnet_logits_all = torch.cat(resnet_logits_list, dim=0)
    lstm_logits_all = torch.cat(lstm_logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    logger.info("Training ensemble weights...")
    weighter = EnsembleWeighter().to(device)
    optimizer = Adam(weighter.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_weights = None

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = weighter(resnet_logits_all, lstm_logits_all)
        loss = criterion(out, labels_all)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = torch.softmax(weighter.weights, dim=0).detach().cpu().numpy()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            w = torch.softmax(weighter.weights, dim=0).detach().cpu().numpy()
            logger.info("Epoch %03d/%03d | Loss: %.4f | ResNet W: %.4f | LSTM W: %.4f", 
                        epoch + 1, args.epochs, loss.item(), w[0], w[1])

    logger.info("Final Best Loss: %.4f", best_loss)
    weights_dict = {
        "resnet_weight": float(best_weights[0]),
        "lstm_weight": float(best_weights[1])
    }
    
    weights_path = Path(args.models_dir) / "ensemble_weights.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=4)
        
    logger.info("Saved ensemble weights to %s:\n%s", weights_path, json.dumps(weights_dict, indent=4))

if __name__ == "__main__":
    main()
