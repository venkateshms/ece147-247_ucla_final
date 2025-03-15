# Transformer Scaling Experiments for EMG2QWERTY

This directory contains code and configurations for running transformer scaling experiments on the EMG2QWERTY dataset. The goal is to establish scaling laws for transformer performance on this specific task.

## Overview

We train 5 different transformer models of increasing size by varying the number of layers and hidden dimension size. All models are trained on the same dataset with the same fixed hyperparameters (learning rate, optimizer, etc.) to isolate the effect of model size on performance.

## Model Sizes

| Model | d_model | Heads | Layers | FF Dim | Approx. Params |
|-------|---------|-------|--------|--------|---------------|
| Tiny  | 128     | 2     | 2      | 512    | 0.5M          |
| Small | 256     | 4     | 3      | 768    | 2M            |
| Medium| 384     | 6     | 4      | 1024   | 5M            |
| Large | 512     | 8     | 6      | 2048   | 15M           |
| XLarge| 768     | 12    | 8      | 3072   | 35M           |

## Running the Experiments

To run all the scaling experiments, use:

```bash
./run_scaling_experiments.sh
```

This will:
1. Train each model one after another
2. Run evaluation on each model after training
3. Store results in the `scaling_results/` directory

## Analyzing Results

After running the experiments, you can analyze the scaling behavior using:

```bash
python analyze_scaling_experiments.py
```

This script will:
1. Extract metrics from the experiment results
2. Plot scaling curves (error rate vs. model size)
3. Save the results in the `scaling_analysis/` directory

## Expected Outcome

The typical outcome of scaling experiments shows that:
1. Performance (error rate) follows a power law with respect to model size
2. There's often a "sweet spot" where increasing model size further provides diminishing returns

This information can help determine the optimal model size for deployment, balancing accuracy and computational efficiency.

## Configurations

All model configurations are in the `config/scaling_experiments/` directory. Each model shares the same hyperparameters except for the architectural choices that determine model size. 