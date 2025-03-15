#!/bin/bash
#!/bin/bash

# Activate the conda environment if needed
# conda activate emg2qwerty

# Run the training script with 2 GPUs using the RNN model
python -m emg2qwerty.train \
  user="single_user" \
  model=rnn_ctc \
  trainer.accelerator=gpu trainer.devices=2 \
  --multirun


echo "RNN Completed, starting Transformer" 


python -m emg2qwerty.train \
  user="single_user" \
  model=medium_transformer_ctc \
  trainer.accelerator=gpu trainer.devices=2 \
  --multirun


# Note: The --multirun flag is uncommented to enable running multiple configurations
# This is useful when using multiple GPUs

# Wait for training to complete
echo "Transformer Completed"
