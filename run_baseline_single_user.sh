#!/bin/bash

# Run the training script with 1 GPUs
echo "Starting training with 2 GPUs..."
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 \
  --multirun

# Wait for training to complete
echo "Training completed. Starting testing..."

# Run the testing script
echo "Starting testing with simplified checkpoint path..."
python -m emg2qwerty.train \
  user="single_user" \
  checkpoint="$SIMPLE_CHECKPOINT" \
  train=False trainer.accelerator=gpu trainer.devices=2 \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun

echo "Testing completed."

# Optional: Remove the symbolic link after testing
# rm "$SIMPLE_CHECKPOINT" 
