#!/bin/bash

# Activate the conda environment if needed
# conda activate emg2qwerty

# Run the training script with 2 GPUs using the RNN model
# Use the existing linear_warmup_cosine_annealing scheduler configuration
python -m emg2qwerty.train \
  user="single_user" \
  model=rnn_ctc \
  trainer.accelerator=gpu trainer.devices=2 \
  lr_scheduler=linear_warmup_cosine_annealing \
  --multirun

# Note: The --multirun flag is uncommented to enable running multiple configurations
# This is useful when using multiple GPUs

# Wait for training to complete
echo "Training completed. Starting testing..."

# Run the testing script with the RNN model
python -m emg2qwerty.train \
  user="single_user" \
  model=rnn_ctc \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/single_user.ckpt" \
  train=False trainer.accelerator=gpu trainer.devices=2 \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun

echo "Testing completed." 