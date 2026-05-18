# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

###################
# training config #
###################

training_config = EasyDict(__name__="Training Configuration")

training_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
training_config.random_seed = 42

training_config.output_dir = "outputs"

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

training_config.append_train_history_step = 10
training_config.append_val_history_step = 5 # must be <= batch_size // 2

training_config.early_stopping_patience = 5
training_config.ckpt_basename = "best_model"
training_config.ckpt_format = "pth"  # or 'pt'

training_config.ce_weight = 1.0
training_config.dice_weight = 1.0
training_config.ignore_index = -1


def get_config():
    return training_config
