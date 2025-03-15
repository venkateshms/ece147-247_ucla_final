#!/bin/bash

# Activate the conda environment if needed

# Run training for transformer_tiny_ctc model
echo "Starting training for transformer_tiny_ctc model..."
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_tiny_ctc \
  trainer.accelerator=gpu trainer.devices=1 \
  lr_scheduler=linear_warmup_cosine_annealing

# Run training for transformer_small_ctc model
echo "Starting training for transformer_small_ctc model..."
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_small_ctc \
  trainer.accelerator=gpu trainer.devices=1 \
  lr_scheduler=linear_warmup_cosine_annealing

# Run training for medium_transformer_ctc model
echo "Starting training for medium_transformer_ctc model..."
python -m emg2qwerty.train \
  user="single_user" \
  model=medium_transformer_ctc \
  trainer.accelerator=gpu trainer.devices=1 \
  lr_scheduler=linear_warmup_cosine_annealing

# Run training for transformer_large_ctc model
echo "Starting training for transformer_large_ctc model..."
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_large_ctc \
  trainer.accelerator=gpu trainer.devices=1 \
  lr_scheduler=linear_warmup_cosine_annealing

# Run training for transformer_xlarge_ctc model
echo "Starting training for transformer_xlarge_ctc model..."
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_xlarge_ctc \
  trainer.accelerator=gpu trainer.devices=1 \
  lr_scheduler=linear_warmup_cosine_annealing
