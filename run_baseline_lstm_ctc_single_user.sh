#!/bin/bash

# Activate the conda environment if needed
# conda activate emg2qwerty

# Define the models to run
MODELS=("lstm_ctc" "lstm_ctc_small" "lstm_ctc_large")

# Run training for each model separately
echo "Training model: lstm_ctc"
python -m emg2qwerty.train \
  user="single_user" \
  model=lstm_ctc \
  trainer.accelerator=gpu trainer.devices=2 \
  --multirun

echo "Training model: lstm_ctc_small"
python -m emg2qwerty.train \
  user="single_user" \
  model=lstm_ctc_small \
  trainer.accelerator=gpu trainer.devices=2 \
  --multirun

echo "Training model: lstm_ctc_medium"
python -m emg2qwerty.train \
  user="single_user" \
  model=lstm_ctc_medium \
  trainer.accelerator=gpu trainer.devices=2 \
  --multirun

echo "Training model: lstm_ctc_large"
python -m emg2qwerty.train \
  user="single_user" \
  model=lstm_ctc_large \
  trainer.accelerator=gpu trainer.devices=2 \
  --multirun

echo "All training completed." 