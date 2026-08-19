# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import time
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import time_formatter, get_class_weights
from src.callbacks import TrainingCallback, TrainCheckpoint, EarlyStopping
from src.modules import SegmentationLoss
from src.metrics import update_confusion_matrix, compute_training_metrics


def freeze_backbone(model: nn.Module):
    """
    Freeze common ResNet-style encoder layers for warmup training.
    """
    frozen_params = 0

    # name for resnet encoder layer
    for name in ("layer0", "layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, name, None)
        if layer is None:
            continue
        # freeze param by disabling grad
        for param in layer.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    print(f"[freeze] Backbone frozen ({frozen_params:,} parameters).")


def unfreeze_all(model: nn.Module, new_lr: float, optimizer):
    """
    Unfreeze every parameter and add them to the optimizer.
    """
    # check existing optimizer params to avoid duplicates
    existing_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    # gather new params that are not already in the optimizer
    new_params = []
    for param in model.parameters():
        param.requires_grad = True
        if id(param) not in existing_ids:
            new_params.append(param)

    # add new param group with the specified learning rate
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": new_lr})

    print(f"[unfreeze] All layers unfrozen. New LR for new params: {new_lr}")


def get_amp_dtype(device):
    """
    Choose an autocast dtype that works across CPU and different CUDA GPUs.
    """
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16

    return torch.bfloat16


class Trainer:
    """
    Trainer class.

    Expected dataset __getitem__: (image_tensor, mask_tensor)
        image: (C, H, W) float
        mask: (H, W) long integer class indices

    Config keys:
        device, output_dir, epochs, lr, weight_decay, label_smoothing,
        grad_clip, two_phase, warmup_epochs, unfreeze_lr,
        early_stopping_patience, ckpt_basename, ckpt_format,
        ce_weight, dice_weight, ignore_index

    Args:
        model:
            torch module model
        model_name:
            model alias name
        train_loader:
            training set torch dataloader
        val_loader:
            validation set torch dataloader
        num_classes:
            number of label class
        config:
            configuration dict
    """

    def __init__(
        self,
        model,
        model_name,
        train_loader,
        val_loader,
        num_classes,
        config,
    ):
        self.config = config
        self.device = config["device"]
        self.amp_dtype = get_amp_dtype(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_classes = num_classes

        print("Preparing training...")

        self.model = model.to(self.device)

        if config["two_phase"]:
            freeze_backbone(self.model)

        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = SegmentationLoss(
            num_classes=self.num_classes,
            ce_weight=config["ce_weight"],
            dice_weight=config["dice_weight"],
            label_smoothing=config["label_smoothing"],
            ignore_index=config["ignore_index"],
            class_weights=get_class_weights(config, self.num_classes),
        ).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for optimizer setup.")

        self.optimizer = AdamW(
            trainable_params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=config["lr"] * 1e-2
        )

        ckpt_filename = (
            config["ckpt_basename"].format(model_name) + config["ckpt_format"]
        )
        ckpt_path = model_dir / ckpt_filename

        self.callbacks = TrainingCallback(
            checkpoint=TrainCheckpoint(filepath=ckpt_path, mode="max"),  # max for mIoU
            early_stop=EarlyStopping(
                patience=config["early_stopping_patience"],
                mode="max",  # max for mIoU
            ),
        )

        history_filename = f"{model_name}_history.pkl"
        self.history_path = model_dir / history_filename

        self.history = []

    def fit(self):
        print("Starting training...\n")

        start_time = time.time()
        ignore_index = self.config["ignore_index"]

        for epoch in range(1, self.config["epochs"] + 1):
            epoch_start = time.time()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # two-phase: unfreeze backbone after warmup
            if self.config["two_phase"] and epoch == self.config["warmup_epochs"] + 1:
                # unfreeze all layers
                unfreeze_all(self.model, self.config["unfreeze_lr"], self.optimizer)

                # update scheduler with new T_max for remaining epochs
                remaining = self.config["epochs"] - epoch + 1

                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining,
                    eta_min=self.config["unfreeze_lr"] * 1e-2,
                )

            # train
            self.model.train()

            train_loss = train_ce = train_dice = 0.0
            train_confmat = torch.zeros(
                self.num_classes,
                self.num_classes,
                device=self.device,
                dtype=torch.long,
            )

            train_step = 0

            batch_iter = tqdm(self.train_loader, desc=f"epoch {epoch}")

            for imgs, masks in batch_iter:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device).long()

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits = self.model(imgs)
                    loss, ce_loss, dice_loss = self.criterion(logits, masks)

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )
                self.optimizer.step()

                preds = logits.argmax(dim=1)
                train_confmat = update_confusion_matrix(
                    train_confmat,
                    preds,
                    masks,
                    self.num_classes,
                    ignore_index,
                )

                # global loss
                train_loss += loss.item()
                train_ce += ce_loss.item()
                train_dice += dice_loss.item()

                train_step += 1
                batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

            self.scheduler.step()

            # eval
            self.model.eval()

            val_loss = val_ce = val_dice = 0.0
            val_confmat = torch.zeros(
                self.num_classes,
                self.num_classes,
                device=self.device,
                dtype=torch.long,
            )

            with torch.no_grad():
                for imgs, masks in tqdm(self.val_loader, desc="validation"):
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device).long()

                    with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        logits = self.model(imgs)
                        loss, ce_loss, dice_loss = self.criterion(logits, masks)

                    preds = logits.argmax(dim=1)
                    val_confmat = update_confusion_matrix(
                        val_confmat,
                        preds,
                        masks,
                        self.num_classes,
                        ignore_index,
                    )

                    # windowed loss
                    val_loss += loss.item()
                    val_ce += ce_loss.item()
                    val_dice += dice_loss.item()

            # avg train loss
            train_loss /= len(self.train_loader)
            train_ce /= len(self.train_loader)
            train_dice /= len(self.train_loader)

            # avg val loss
            val_loss /= len(self.val_loader)
            val_ce /= len(self.val_loader)
            val_dice /= len(self.val_loader)

            # IoU is the primary metric
            train_metrics = compute_training_metrics(train_confmat)
            val_metrics = compute_training_metrics(val_confmat)
            train_f1 = train_metrics["f1"]
            val_f1 = val_metrics["f1"]
            train_miou = train_metrics["miou"]
            val_miou = val_metrics["miou"]

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": {
                        "total": train_loss,
                        "ce": train_ce,
                        "dice": train_dice,
                    },
                    "val_loss": {
                        "total": val_loss,
                        "ce": val_ce,
                        "dice": val_dice,
                    },
                    "train_metrics": {
                        "f1": train_f1,
                        "miou": train_miou,
                    },
                    "val_metrics": {
                        "f1": val_f1,
                        "miou": val_miou,
                        "building_iou": val_metrics["building_iou"],
                        "damage_miou": val_metrics["damage_miou"],
                    },
                }
            )

            # logging
            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{self.config['epochs']} - {time_formatter(epoch_time)} | "
                f"train_loss={train_loss:.6f} (ce={train_ce:.4f}, dice={train_dice:.4f}) | "
                f"val_loss={val_loss:.6f} (ce={val_ce:.4f}, dice={val_dice:.4f}) | "
                f"train_f1={train_f1:.4f} | val_f1={val_f1:.4f} | "
                f"train_mIoU={train_miou:.4f} | val_mIoU={val_miou:.4f} | "
                f"val_building_IoU={val_metrics['building_iou']:.4f} | "
                f"val_damage_mIoU={val_metrics['damage_miou']:.4f}"
            )

            # callbacks
            model_dict = self.model.state_dict()

            optimizer_dict = {
                "adam": self.optimizer.state_dict(),
                "cosine_scheduler": self.scheduler.state_dict(),
            }

            metadata = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_miou": train_miou,
                "val_miou": val_miou,
                "val_building_iou": val_metrics["building_iou"],
                "val_damage_miou": val_metrics["damage_miou"],
            }

            is_stopping = self.callbacks.step(
                monitor_value=val_miou,  # using val mIoU for monitoring metrics
                model_dict=model_dict,
                metadata=metadata,
                optimizer_dict=optimizer_dict,
            )

            if is_stopping:
                break

            print("\n")

        end_time = time.time()
        print(f"elapsed time: {time_formatter(end_time - start_time)}")

        with open(self.history_path, "wb") as f:
            pickle.dump(self.history, f)

        print(f"training complete, history saved to {self.history_path}")
