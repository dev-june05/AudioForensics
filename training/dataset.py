"""
PyTorch Dataset for PartialSpoof audio classification.

Provides 30-second contiguous windows.
Each window is given a binary label:
- 1 (Fake) if ANY 0.64s segment within it is spoofed.
- 0 (Real) otherwise.
"""

import math
import os
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
from config import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    POWER,
    SAMPLE_RATE,
    SILENCE_THRESHOLD_DB,
    WINDOW_DURATION_SEC,
    WINDOW_NUM_SAMPLES,
)


def _pad_to_window(y: np.ndarray) -> np.ndarray:
    if len(y) < WINDOW_NUM_SAMPLES:
        y = np.pad(y, (0, WINDOW_NUM_SAMPLES - len(y)), mode='constant')
    elif len(y) > WINDOW_NUM_SAMPLES:
        y = y[:WINDOW_NUM_SAMPLES]
    return y


def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=POWER,
    )


def _to_log_scale(mel_spec: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(mel_spec, ref=np.max)


def _normalize(mel_db: np.ndarray) -> np.ndarray:
    mean = float(np.mean(mel_db))
    std = float(np.std(mel_db))
    if std <= 0:
        std = 1.0
    return (mel_db - mean) / std


class AudioAugmentor:
    """Waveform augmentation."""
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        if random.random() < self.p:
            y = self._add_gaussian_noise(y)
        if random.random() < self.p:
            y = self._time_stretch(y, sr)
        if random.random() < self.p:
            y = self._pitch_shift(y, sr)
        return y

    @staticmethod
    def _add_gaussian_noise(y: np.ndarray, snr_db_range: Tuple[float, float] = (10, 30)) -> np.ndarray:
        snr_db = random.uniform(*snr_db_range)
        signal_power = np.mean(y ** 2)
        if signal_power <= 0:
            return y
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return (y + noise).astype(y.dtype)

    @staticmethod
    def _time_stretch(y: np.ndarray, sr: int, rate_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        rate = random.uniform(*rate_range)
        return librosa.effects.time_stretch(y, rate=rate)

    @staticmethod
    def _pitch_shift(y: np.ndarray, sr: int, semitone_range: Tuple[float, float] = (-2, 2)) -> np.ndarray:
        n_steps = random.uniform(*semitone_range)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


class PartialSpoofDataset(Dataset):
    """
    Dataset for PartialSpoof.
    Splits files into fixed 30s windows.
    Returns (tensor, label) for a single window.
    """
    VALID_MODES = {"resnet", "lstm", "raw"}

    def __init__(
        self,
        audio_dirs: List[str | Path],
        segment_labels_path: str | Path,
        mode: str = "resnet",
        augment: bool = False,
        augment_prob: float = 0.5,
        resolution_sec: float = 0.64,
        file_list: Optional[List[str]] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.audio_dirs = [Path(d) for d in audio_dirs]
        self.mode = mode
        self.augment = augment
        self.augmentor = AudioAugmentor(p=augment_prob) if augment else None
        self.resolution_sec = resolution_sec

        # Load segment labels (dict mapping file_id to array of '0' and '1')
        if str(segment_labels_path).endswith('.npy'):
            raw_labels = np.load(segment_labels_path, allow_pickle=True).item()
        else:
            # Parse PartialSpoof .txt protocol file directly
            raw_labels = {}
            with open(segment_labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        file_id = parts[1]
                        
                        # Find the label key ('bonafide' or 'spoof')
                        key_idx = -1
                        for i, p in enumerate(parts):
                            if p.lower() in ('bonafide', 'spoof'):
                                key_idx = i
                                break
                                
                        if key_idx != -1:
                            if key_idx + 1 < len(parts):
                                segments_str = parts[key_idx + 1:]
                                # Handle both space-separated ['0', '1', '1'] and concatenated ['0110']
                                if len(segments_str) == 1 and len(segments_str[0]) > 1:
                                    segments = list(segments_str[0])
                                else:
                                    segments = segments_str
                                raw_labels[file_id] = segments
                            else:
                                # Utterance-level label only
                                overall_label = '0' if parts[key_idx].lower() == 'spoof' else '1' # In original data 0=spoof, 1=bonafide
                                raw_labels[file_id] = overall_label
            
            if not raw_labels:
                raise ValueError(f"Could not parse any segment labels from {segment_labels_path}")

        # Filter files if file_list is provided
        if file_list is not None:
            allowed_ids = {Path(f).stem for f in file_list}
            self.labels_dict = {k: v for k, v in raw_labels.items() if k in allowed_ids}
        else:
            self.labels_dict = raw_labels

        self.windows = []
        import os

        for file_id, segments_or_label in self.labels_dict.items():
            audio_path = None
            for d in self.audio_dirs:
                p_wav = d / f"{file_id}.wav"
                p_flac = d / f"{file_id}.flac"
                if p_wav.exists():
                    audio_path = p_wav
                    break
                if p_flac.exists():
                    audio_path = p_flac
                    break
            
            if audio_path is None:
                continue
                
            if isinstance(segments_or_label, list):
                total_duration_sec = len(segments_or_label) * self.resolution_sec
                segments = segments_or_label
            else:
                # Calculate duration from file size (assuming 16kHz 16-bit mono WAV)
                file_size = os.path.getsize(audio_path)
                num_samples = max(0, (file_size - 44) // 2)
                total_duration_sec = num_samples / SAMPLE_RATE
                num_segments = max(1, int(np.ceil(total_duration_sec / self.resolution_sec)))
                segments = [segments_or_label] * num_segments

            total_samples = int(total_duration_sec * SAMPLE_RATE)
            
            # Divide into 30s windows
            for start_sample in range(0, total_samples, WINDOW_NUM_SAMPLES):
                end_sample = min(start_sample + WINDOW_NUM_SAMPLES, total_samples)
                
                # Determine window label
                start_sec = start_sample / SAMPLE_RATE
                end_sec = end_sample / SAMPLE_RATE
                
                start_seg_idx = int(start_sec / self.resolution_sec)
                end_seg_idx = int(math.ceil(end_sec / self.resolution_sec))
                
                window_segments = segments[start_seg_idx:end_seg_idx]
                
                # If ANY segment in this window is '0' (spoofed), the window is spoofed.
                # In our training label mapping: 0 = Real, 1 = Fake
                is_fake = any(seg == '0' for seg in window_segments)
                window_label = 1 if is_fake else 0
                
                self.windows.append((audio_path, start_sample, end_sample, window_label))

    def get_labels(self) -> List[int]:
        return [w[3] for w in self.windows]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, start_sample, end_sample, label = self.windows[idx]
        
        # Load just the needed chunk
        offset = start_sample / SAMPLE_RATE
        duration = (end_sample - start_sample) / SAMPLE_RATE
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, offset=offset, duration=duration, mono=True)

        if self.augment and self.augmentor is not None:
            y = self.augmentor(y, SAMPLE_RATE)

        y = _pad_to_window(y)

        if self.mode == "raw":
            return torch.from_numpy(y).float(), label

        mel_spec = _compute_mel_spectrogram(y)
        mel_db = _to_log_scale(mel_spec)
        mel_norm = _normalize(mel_db)

        if self.mode == "lstm":
            tensor = torch.from_numpy(mel_norm.T).float()
        else:
            tensor = torch.from_numpy(mel_norm).float().unsqueeze(0)

        return tensor, label
