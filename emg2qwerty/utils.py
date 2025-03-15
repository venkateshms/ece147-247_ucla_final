# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
import math
import torch


def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def cpus_per_task(gpus_per_node: int, tasks_per_node: int, num_workers: int) -> int:
    """Number of CPUs to request per task per node taking into account
    the number of GPUs and dataloading workers."""
    gpus_per_task = gpus_per_node // tasks_per_node
    if gpus_per_task <= 0:
        return num_workers + 1
    else:
        return (num_workers + 1) * gpus_per_task


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Args:
        d_model (int): The embedding dimension of the model.
        max_len (int): Maximum sequence length for the positional encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x
