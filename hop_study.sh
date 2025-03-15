#!/bin/bash

# This script trains LSTM_CTC models with different hop lengths
# Usage: ./train_lstm_hop_study.sh

echo "Starting hop length experiments for LSTM_CTC models (65 epochs each)..."

# Create experiments directory if it doesn't exist
EXPERIMENTS_DIR="experiments/lstm_hop_study/"
mkdir -p "$EXPERIMENTS_DIR"
echo "Saving all results to: $EXPERIMENTS_DIR"

# Train LSTM_CTC with different hop lengths
for HOP_LENGTH in 5 10 15; do
  echo "Training LSTM_CTC model with hop_length=$HOP_LENGTH..."

  python -m emg2qwerty.train \
    --multirun
    user="single_user" \
    model="lstm_ctc" \
    trainer.accelerator=gpu \
    trainer.devices=2 \
    trainer.max_epochs=65 \
    +transform.log_spectrogram.hop_length=$HOP_LENGTH \
    +name="lstm_ctc_hop${HOP_LENGTH}_65epochs" \
    hydra.run.dir="$EXPERIMENTS_DIR/hop${HOP_LENGTH}/${now:%Y-%m-%d}/${now:%H-%M-%S}"

  echo "Completed training LSTM_CTC with hop_length=$HOP_LENGTH"
done

echo "All LSTM_CTC hop length experiments completed!" 