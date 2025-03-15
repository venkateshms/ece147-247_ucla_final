#!/bin/bash

# Scaling experiments for transformers of different sizes
# This script runs training and evaluation for 5 transformer models of increasing size
# to establish scaling laws for transformer performance on the EMG2QWERTY task.

# Activate the conda environment if needed
# conda activate emg2qwerty

# Create output directory for results
mkdir -p scaling_results

# Function to run training and testing for a model
run_experiment() {
    local model_name=$1
    local config_path=$2
    
    echo "============================================================"
    echo "Starting experiment for $model_name"
    echo "============================================================"
    
    # Run training with the existing linear_warmup_cosine_annealing scheduler
    echo "Training $model_name..."
    python -m emg2qwerty.train \
      user="single_user" \
      model=$config_path \
      trainer.accelerator=gpu trainer.devices=2 \
      trainer.default_root_dir="scaling_results/$model_name" \
      lr_scheduler=linear_warmup_cosine_annealing \
      --multirun
    
    echo "$model_name training completed."
    
    # Run testing
    echo "Testing $model_name..."
    python -m emg2qwerty.train \
      user="single_user" \
      model=$config_path \
      checkpoint="scaling_results/$model_name/lightning_logs/version_0/checkpoints/last.ckpt" \
      train=False trainer.accelerator=gpu trainer.devices=2 \
      decoder=ctc_greedy \
      hydra.launcher.mem_gb=64 \
      --multirun
    
    echo "$model_name testing completed."
    echo ""
}

# Run experiments for all model sizes
echo "Starting scaling experiments..."

# Run tiny transformer experiment
run_experiment "transformer_tiny" "transformer_tiny_ctc"

# Run small transformer experiment
run_experiment "transformer_small" "transformer_small_ctc"

# Run medium transformer experiment
run_experiment "transformer_medium" "medium_transformer_ctc"

# Run large transformer experiment
run_experiment "transformer_large" "transformer_large_ctc"

# Run xlarge transformer experiment
run_experiment "transformer_xlarge" "transformer_xlarge_ctc"

echo "All scaling experiments completed!" 
