# AudioForensics — AI-Generated Audio Detection

**Using Spectral Feature Analysis and Deep Learning**

A full-stack web application that detects AI-generated and partially spoofed audio using a **ResNet-18 + BiLSTM ensemble** with **30-second windowed analysis** and **attention-based temporal pooling**. Upload a speech clip, and the system classifies it as **Real** or **Spoofed** with confidence scores from two independent deep learning models.

---

## Problem Statement

The rapid advancement of AI voice synthesis (text-to-speech, voice cloning, deepfake audio) has made it increasingly difficult to distinguish real human speech from machine-generated audio. Modern spoofing attacks can inject fake segments into otherwise genuine recordings, making **partial spoofing** a critical challenge. This project addresses the problem by combining two complementary deep learning approaches trained on the PartialSpoof dataset.

## Motivation

- AI-generated speech is being misused for fraud, misinformation, and identity theft.
- Partial spoofing (where only segments of audio are fake) requires models that can analyze temporal context across longer windows.
- Single-model classifiers can be brittle — an ensemble of architecturally different models provides more robust detection.
- Mel-spectrogram analysis captures both spatial (frequency domain patterns) and temporal (time-series dynamics) anomalies left by audio generation algorithms.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│  Upload audio → Preview → Send to API → Display ensemble results │
│  Show spectrogram image, ResNet card, LSTM card, Ensemble card   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │  HTTP POST /predict (multipart file)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                           │
│                                                                  │
│  1. Validate uploaded audio file                                 │
│  2. Preprocess (preprocess.py):                                  │
│     • Resample to 16 kHz mono                                    │
│     • Divide into 30-second contiguous windows                   │
│     • Compute mel spectrogram (128 mel bands)                    │
│     • Generate:                                                  │
│       - ResNet tensor (1, 128, T) per window                     │
│       - LSTM tensor (1, T, 128) per window                       │
│       - Base64 spectrogram PNG                                   │
│                                                                  │
│  3. Inference (model_loader.py):                                 │
│     • ResNet-18 → softmax probabilities                          │
│     • LSTM (BiLSTM) → softmax probabilities                      │
│     • Weighted ensemble (configurable/learned weights)           │
│                                                                  │
│  4. Return JSON response with all predictions                    │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User** selects an audio file (WAV, MP3, FLAC, etc.) in the React frontend.
2. **Frontend** sends the file via `POST /predict` to the FastAPI backend.
3. **Backend** preprocesses the audio into fixed 30-second windows and produces tensors for both models.
4. **ResNet-18** receives a mel-spectrogram image `(1, 128, T)` and classifies spatial patterns.
5. **LSTM** receives a temporal mel sequence `(T, 128)` with attention pooling and classifies time-series dynamics.
6. **Ensemble** merges both probability vectors via weighted averaging.
7. **Response** JSON is sent back containing individual predictions, confidence scores, and the spectrogram image.
8. **Frontend** displays the spectrogram and three prediction cards (ResNet, LSTM, Ensemble).

---

## Features

- **Dual-model ensemble** — ResNet-18 (CNN) + Bidirectional LSTM (RNN) for robust detection.
- **30-second windowed analysis** — audio is divided into fixed 30s contiguous windows for consistent processing.
- **PartialSpoof support** — detects partially spoofed audio where only segments within a file are fake.
- **Attention-based LSTM** — learned attention mechanism focuses on the most informative timesteps.
- **Transfer learning** — ResNet-18 initialized with ImageNet pretrained weights for better feature extraction.
- **Data augmentation** — noise injection, time stretch, pitch shift during training.
- **Confidence thresholds** — predictions below 70% confidence are marked as Uncertain.
- **Learned ensemble weights** — `train_ensemble.py` optimizes the ResNet/LSTM blend using validation data.
- **Zero data leakage** — train/val split is done by file ID, not by window.
- **Individual model predictions** — See what each model thinks independently.
- **Spectrogram visualization** — The mel spectrogram is displayed on the dashboard.
- **Audio preview** — Listen to the uploaded clip directly in the browser.
- **Modern dark-themed UI** — Glassmorphism, GSAP animations, responsive layout.

---

## Tech Stack

| Layer             | Technology                                         |
|-------------------|---------------------------------------------------|
| **Frontend**      | React 18, Tailwind CSS 3, GSAP, Lucide Icons, Vite |
| **Backend**       | FastAPI, Uvicorn, Python 3.10+                     |
| **Deep Learning** | PyTorch, torchvision, librosa                      |
| **Training**      | NVIDIA DGX B200 (multi-GPU)                        |
| **Dataset**       | PartialSpoof (ASVspoof-derived, segment-level labels) |
| **Visualization** | matplotlib (server-side spectrogram rendering)     |

---

## Folder Structure

```
AudioForensics/
├── backend/                  # FastAPI backend
│   ├── config.py             # Centralized constants (paths, dims, weights)
│   ├── main.py               # FastAPI app, endpoints, lifecycle
│   ├── model_loader.py       # Dual model loading + ensemble inference
│   ├── preprocess.py         # Audio → ResNet tensor + LSTM tensor + spectrogram
│   ├── utils.py              # Logging, validation, helpers
│   └── requirements.txt      # Python dependencies
│
├── docs/                     # Detailed architectural documentation
│   ├── Internship_Project_Report.md
│   ├── LSTM.md
│   ├── Mathematical Calculation.md
│   └── Resnet.md
│
├── frontend/                 # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx           # Main layout and API call
│   │   ├── components/
│   │   │   ├── UploadCard.jsx    # File upload + audio preview
│   │   │   └── ResultCard.jsx    # Spectrogram + 3 prediction cards
│   │   ├── App.css           # Additional styles
│   │   ├── index.css         # Tailwind base + body background
│   │   └── main.jsx          # React entry point
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.mjs
│   └── vite.config.mjs
│
├── training/                 # Model training pipeline
│   ├── train.py              # CLI training script (--model resnet|lstm)
│   ├── train_ensemble.py     # Learn optimal ensemble weights
│   ├── model.py              # AudioResNet + AudioLSTM definitions
│   ├── dataset.py            # PartialSpoofDataset (30s windows, segment labels)
│   └── download_dataset.py   # Dataset download helper
│
├── models/                   # Trained weight files
│   ├── resnet_audio_model.pth
│   ├── lstm_audio_model.pth
│   └── ensemble_weights.json # Learned blend weights (generated by train_ensemble.py)
│
├── requirements_b200.txt     # DGX B200 server dependencies
└── README.md                 # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm
- (Optional) NVIDIA GPU with CUDA for faster training

### 1. Clone the repository

```bash
git clone https://github.com/dev-june05/AudioForensics
cd AudioForensics
```

### 2. Backend setup

```bash
cd backend
python -m venv venv

# Activate the virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

### 4. Train models

Training uses the **PartialSpoof** dataset with segment-level labels. You need:
- Audio directories containing `.wav` or `.flac` files
- A segment labels `.npy` file (e.g., `train_seglab_0.64.npy`)

```bash
cd training

# Train ResNet-18
python train.py --model resnet \
  --audio-dirs /path/to/audio/dir1 /path/to/audio/dir2 \
  --segment-labels /path/to/train_seglab_0.64.npy \
  --epochs 30 --batch-size 32

# Train LSTM
python train.py --model lstm \
  --audio-dirs /path/to/audio/dir1 \
  --segment-labels /path/to/train_seglab_0.64.npy \
  --epochs 30 --batch-size 32

# Learn optimal ensemble weights (after both models are trained)
python train_ensemble.py \
  --audio-dirs /path/to/audio/dir1 \
  --segment-labels /path/to/train_seglab_0.64.npy
```

Model weights will be saved to `models/`.

### 5. Run the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Model Details

### ResNet-18 (Spectrogram Image Classifier)

| Property       | Value                                                 |
|----------------|-------------------------------------------------------|
| Architecture   | ResNet-18 (modified: 1-channel input, 2-class output) |
| Input          | Mel spectrogram `(1, 128, T)` from 30s window         |
| Features       | 128 mel bands, 1024 FFT, 512 hop length               |
| Preprocessing  | Log-dB, z-score normalization, AdaptiveAvgPool2d(1,1)  |
| Transfer Learning | ImageNet pretrained weights (3-channel averaged to 1-channel) |
| Activations    | ReLU (in every residual block and after conv1)        |
| Output         | 2 logits → softmax → [P(Real), P(Spoofed)]            |

### LSTM (Temporal Sequence Classifier)

| Property              | Value                                                    |
|-----------------------|----------------------------------------------------------|
| Architecture          | 2-layer Bidirectional LSTM + FC classifier               |
| Input                 | Mel-spectrogram time series `(T, 128)` from 30s window   |
| Features              | Same mel spectrogram (transposed), with attention-weighted pooling |
| Pooling               | Learned attention mechanism over all timesteps           |
| Hidden dim            | 128 per direction (256 total)                            |
| Gate activations      | Sigmoid (input/forget/output gates), Tanh (cell state)   |
| Classifier activation | ReLU (in FC head between Linear layers)                  |
| Dropout               | 0.3                                                      |
| Output                | 2 logits → softmax → [P(Real), P(Spoofed)]               |

### Loss Function

Both models are trained with **CrossEntropyLoss** (`torch.nn.CrossEntropyLoss`).

```
L = -log( exp(logits[y]) / Σ exp(logits[j]) )
```

Where `y` is the true class index (0 = Real, 1 = Spoofed).

### Ensemble Strategy

Both models produce softmax probability vectors `[P(Real), P(Spoofed)]`. The ensemble computes a weighted average:

```
ensemble_probs = w_resnet × resnet_probs + w_lstm × lstm_probs
```

Weights can be learned automatically via `train_ensemble.py` or set manually in `backend/config.py`. If `models/ensemble_weights.json` exists, the backend loads it dynamically; otherwise it falls back to 50/50.

---

## API Documentation

### `GET /health`

Health check. Returns model loading status.

**Response:**
```json
{
  "status": "ok",
  "models": {
    "resnet": true,
    "lstm": true
  }
}
```

### `POST /predict`

Upload an audio file and receive an ensemble prediction.

**Request:** Multipart form data with a `file` field containing the audio file.

**Response:**
```json
{
  "resnet_prediction": "Real",
  "resnet_confidence": 0.91,
  "lstm_prediction": "Spoofed",
  "lstm_confidence": 0.87,
  "ensemble_prediction": "Spoofed",
  "ensemble_confidence": 0.89,
  "audio_duration_sec": 12.5,
  "confidence_threshold": 0.70,
  "spectrogram_image": "data:image/png;base64,..."
}
```

**Error responses:**
- `400` — Invalid file (empty, wrong type, too large)
- `422` — Audio processing error
- `503` — No models loaded
- `500` — Internal server error

---

## Screenshots

> Screenshots will be added after the UI is finalized. The dashboard shows:
> - Audio upload card with preview player
> - Mel spectrogram visualization
> - Three prediction cards: ResNet-18, LSTM, and Ensemble
> - Confidence bars with gradient colors

---

## Future Improvements

- Add a third model (e.g., wav2vec2 / HuBERT) for a stronger ensemble.
- Provide Grad-CAM visualizations on the spectrogram.
- Deploy to cloud with Docker containerization.
- Add user authentication and prediction history.
- Support real-time microphone input.
- Add cross-dataset evaluation (ASVspoof 2019, FakeAVCeleb).
- Add adversarial robustness testing.
