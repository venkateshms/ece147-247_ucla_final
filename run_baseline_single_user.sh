#!/bin/bash

# Path to the original checkpoint
ORIGINAL_CHECKPOINT="/u/home/m/mven/project-spellman/emg2qwerty/logs/2025-03-10/20-13-28/job0_trainer.devices=2,user=single_user/checkpoints/epoch=109-step=6600.ckpt"

# Create a symbolic link with a simple name
SIMPLE_CHECKPOINT="/u/home/m/mven/project-spellman/emg2qwerty/checkpoint_simple.ckpt"
echo "Creating symbolic link to checkpoint..."
ln -sf "$ORIGINAL_CHECKPOINT" "$SIMPLE_CHECKPOINT"

# # Run the training script with 2 GPUs
# echo "Starting training with 2 GPUs..."
# python -m emg2qwerty.train \
#   user="single_user" \
#   trainer.accelerator=gpu trainer.devices=2 \
#   --multirun

# # Wait for training to complete
# echo "Training completed. Starting testing..."

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