#!/bin/bash

# Start Wav2Vec 2.0
nohup python train_transformer.py \
  --model wav2vec2 \
  --audio-dirs ~/partialspoof/extracted_audio/database/train/con_wav \
  --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt \
  --epochs 15 \
  --batch-size 8 \
  --grad-accum 4 \
  --num-workers 16 > wav2vec2_training.log 2>&1 &

echo "Wav2Vec2 started."

# Start HuBERT
nohup python train_transformer.py \
  --model hubert \
  --audio-dirs ~/partialspoof/extracted_audio/database/train/con_wav \
  --segment-labels ~/partialspoof/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.train.trl.txt \
  --epochs 15 \
  --batch-size 8 \
  --grad-accum 4 \
  --num-workers 16 > hubert_training.log 2>&1 &

echo "HuBERT started."
