# Training — AudioForensics Models

Training pipeline for the ResNet-18 and BiLSTM models used in the audio deepfake detection system. Both models are trained on mel-spectrogram features extracted from the **PartialSpoof** dataset using fixed **30-second contiguous windows**.

---

## Dataset: PartialSpoof

The training pipeline uses the **PartialSpoof** dataset (ASVspoof-derived) with segment-level spoof labels at 0.64-second resolution.

### What you need:

1. **Audio files** — `.wav` or `.flac` files from the PartialSpoof corpus
2. **Segment labels** — A `.npy` file (e.g., `train_seglab_0.64.npy`) mapping each file ID to an array of per-segment labels (`'0'` = spoofed, `'1'` = bonafide)

### How windowing works:

Each audio file is divided into **30-second contiguous windows**. A window is labeled as **Spoofed** (label 1) if **any** 0.64-second segment within it is spoofed. Otherwise, it is labeled **Real** (label 0).

```
Audio file (e.g., 90 seconds)
├── Window 1 (0–30s)  → segments all bonafide → label 0 (Real)
├── Window 2 (30–60s) → segments [25] is spoofed → label 1 (Spoofed)
└── Window 3 (60–90s) → segments all bonafide → label 0 (Real)
```

---

## Preprocessing

Both models share the same initial preprocessing:

1. **Load** audio with librosa at 16,000 Hz mono.
2. **Window** into fixed 30-second chunks (480,000 samples per window).
3. **Pad** the last window if shorter than 30 seconds.
4. **Compute** mel spectrogram with 128 mel bands, 1024 FFT window, 512 hop length.
5. **Convert** to log-decibel scale.
6. **Normalize** to zero mean and unit variance (per spectrogram).

The two models then diverge:

| Model   | Tensor shape            | Notes                                      |
|---------|------------------------|--------------------------------------------|
| ResNet  | `(1, 128, T)`          | 2D spectrogram image, AdaptiveAvgPool handles variable T |
| LSTM    | `(T, 128)`             | Temporal sequence, attention pooling over all timesteps |

---

## File Structure

| File                 | Purpose                                                     |
|----------------------|-------------------------------------------------------------|
| `train.py`           | CLI training script with `--model resnet\|lstm` flag         |
| `train_ensemble.py`  | Learn optimal ensemble blending weights                     |
| `model.py`           | `AudioResNet` and `AudioLSTM` class definitions             |
| `dataset.py`         | `PartialSpoofDataset` — 30s windowed dataset with segment labels |
| `download_dataset.py`| Dataset download helper                                     |

---

## How Training Works

1. The segment labels `.npy` file is loaded and parsed into file IDs.
2. File IDs are split into train/val sets (**no data leakage** — split by file, not by window).
3. Each file is divided into 30-second windows with per-window binary labels.
4. The model trains with **Adam** optimizer, **CosineAnnealingLR** scheduler, and **CrossEntropyLoss**.
5. **Early stopping** halts training if validation accuracy doesn't improve for `--patience` epochs.
6. The best model (by validation accuracy) is saved to `models/`.
7. Evaluation plots (loss curve, accuracy curve, confusion matrix) are saved to `training/outputs/<architecture>/`.

---

## How to Run Training

### Train ResNet-18

```bash
cd training
python train.py --model resnet \
  --audio-dirs /path/to/audio/dir \
  --segment-labels /path/to/train_seglab_0.64.npy \
  --epochs 30 --batch-size 32 --lr 1e-4
```

Saves weights to: `models/resnet_audio_model.pth`

### Train LSTM

```bash
cd training
python train.py --model lstm \
  --audio-dirs /path/to/audio/dir \
  --segment-labels /path/to/train_seglab_0.64.npy \
  --epochs 30 --batch-size 32 --lr 1e-4
```

Saves weights to: `models/lstm_audio_model.pth`

### Learn Ensemble Weights

After both models are trained:

```bash
python train_ensemble.py \
  --audio-dirs /path/to/audio/dir \
  --segment-labels /path/to/train_seglab_0.64.npy
```

Saves weights to: `models/ensemble_weights.json`

### Full CLI options

```
python train.py --help

  --model {resnet,lstm}        Architecture to train (required)
  --audio-dirs DIR [DIR ...]   Directories containing audio files (required)
  --segment-labels PATH        Path to segment labels .npy file (required)
  --models-dir PATH            Output directory for .pth files (default: ../models)
  --batch-size N               Batch size (default: 32)
  --epochs N                   Number of epochs (default: 30)
  --lr FLOAT                   Learning rate (default: 1e-4)
  --val-split FLOAT            Validation fraction (default: 0.2)
  --num-workers N              DataLoader workers (default: 0)
  --patience N                 Early stopping patience (default: 5)
  --no-augment                 Disable data augmentation
  --verbose                    Enable debug logging
```

---

## Where Models Are Saved

```
models/
├── resnet_audio_model.pth    # ResNet-18 state_dict
├── lstm_audio_model.pth      # LSTM state_dict
└── ensemble_weights.json     # Learned blend weights (optional)
```

These are standard PyTorch `state_dict` files loaded by `backend/model_loader.py` at server startup.

---

## Difference Between ResNet and LSTM Training

| Aspect        | ResNet-18                           | LSTM                                  |
|---------------|-------------------------------------|---------------------------------------|
| Input         | 2D spectrogram image `(1, 128, T)`  | 1D temporal sequence `(T, 128)`       |
| Architecture  | Deep CNN with residual connections  | 2-layer BiLSTM + attention + FC head  |
| What it learns| Spatial texture patterns in spectrograms | Temporal dynamics in mel frames   |
| Parameters    | ~11M                                | ~400K                                 |
| Strength      | Catches frequency-domain artifacts  | Catches temporal inconsistencies      |
| Pooling       | AdaptiveAvgPool2d(1,1)              | Learned attention over all timesteps  |

Together, they complement each other in the ensemble for more robust detection.

---

## Training Environment

The models are trained on an **NVIDIA DGX B200** server. See `requirements_b200.txt` for server-specific dependencies.

---

## Tips

- **Use a GPU.** Training is significantly faster with CUDA. The scripts auto-detect GPU availability.
- **Zero data leakage.** The train/val split is done by file ID, ensuring no windows from the same file appear in both sets.
- **Consistent preprocessing.** The `dataset.py` preprocessing matches `backend/preprocess.py` for consistent train↔inference features.
- **Augmentation.** Use `--no-augment` for faster debugging runs. Enable augmentation (default) for production training.
