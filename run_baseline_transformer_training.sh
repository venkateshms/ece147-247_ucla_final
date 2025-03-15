#!/bin/bash

# Activate the conda environment if needed
# conda activate emg2qwerty

# Run the training script with 2 GPUs using the medium Transformer model
# Use the existing linear_warmup_cosine_annealing scheduler configuration
python -m emg2qwerty.train \
  user="single_user" \
  model=transformer_tiny_ctc \
  trainer.accelerator=gpu trainer.devices=2 \
  lr_scheduler=linear_warmup_cosine_annealing \
  --multirun

