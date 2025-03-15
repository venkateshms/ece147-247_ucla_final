#!/bin/bash

# Activate the conda environment if needed

python -m emg2qwerty.train \
  user="single_user" \
  model=medium_transformer_ctc \
  trainer.accelerator=gpu trainer.devices=2 \
  lr_scheduler=linear_warmup_cosine_annealing \
  --multirun
