# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

training_config = EasyDict(__name__="Training Configuration")

training_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
training_config.random_seed = 42

training_config.output_dir = "output"

training_config.input_mode = "post"  # "post" or "pre_post"
training_config.in_channels = 3

training_config.batch_size = 16
training_config.epochs = 20
training_config.lr = 1e-3
training_config.weight_decay = 1e-4
training_config.label_smoothing = 0.1
training_config.grad_clip = 1.0
training_config.two_phase = True
training_config.warmup_epochs = 5
training_config.unfreeze_lr = 1e-4
training_config.num_workers = 0
training_config.ce_weight = 1.0
training_config.dice_weight = 1.0
training_config.ignore_index = -1

training_config.classes = {
    "background": {"id": 0, "weight": 0.2},
    "no_damage": {"id": 1, "weight": 1.0},
    "minor_damage": {"id": 2, "weight": 2.0},
    "major_damage": {"id": 3, "weight": 3.0},
    "destroyed": {"id": 4, "weight": 3.0},
}

training_config.early_stopping_patience = 5
training_config.ckpt_basename = "best_model"
training_config.ckpt_format = ".pth"  # '.pth 'or '.pt'


def get_config():
    input_channels = {
        "post": 3,
        "pre_post": 6,
    }

    if training_config.input_mode not in input_channels:
        raise ValueError(
            f"input_mode must be one of {tuple(input_channels)}, "
            f"got {training_config.input_mode!r}"
        )

    training_config.in_channels = input_channels[training_config.input_mode]
    return training_config
