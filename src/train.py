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
from sklearn.metrics import f1_score

from src.utils import time_formatter
from src.callbacks import TrainingCallback, TrainCheckpoint, EarlyStopping
from src.modules.loss import SegmentationLoss


def f1(labels, preds):
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return round(f1, 4)


def mean_iou(preds_flat, labels_flat, num_classes):
    """
    Compute mean IoU (intersection over union)
    over all classes (ignoring classes absent in both).
    """
    preds_t = torch.tensor(preds_flat)
    labels_t = torch.tensor(labels_flat)

    iou_list = []

    for c in range(num_classes):
        # mask for class c in preds and labels
        pred_c = preds_t == c
        true_c = labels_t == c

        # compute intersection and union
        inter = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()

        # skip if union is zero (class not present in preds and labels)
        if union == 0:
            continue

        # compute IoU for class c and append to list
        iou_list.append(inter / union)

    # return mean IoU over classes, or 0.0 if no classes present
    return round(sum(iou_list) / len(iou_list), 4) if iou_list else 0.0


def unfreeze_all(model: nn.Module, new_lr: float, optimizer):
    """Unfreeze every parameter and add them to the optimizer."""
    # check existing optimizer params to avoid duplicates
    existing_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    # gather new params that are not already in the optimizer
    new_params = [p for p in model.parameters() if id(p) not in existing_ids]
    # add new param group with the specified learning rate
    optimizer.add_param_group({"params": new_params, "lr": new_lr})
    print(f"[unfreeze] All layers unfrozen. New LR for backbone: {new_lr}")


class SegmentationTrainer:
    """
    Trainer for semantic segmentation models (UNet, DeepLabV3, etc.).

    Expected dataset __getitem__: (image_tensor, mask_tensor)
        image: (C, H, W)  float
        mask:  (H, W)     long  — integer class indices

    Config keys (same conventions as the classification Trainer):
        device, output_dir, epochs, lr, weight_decay, label_smoothing,
        grad_clip, two_phase, warmup_epochs, unfreeze_lr,
        append_train_history_step, append_val_history_step,
        early_stopping_patience, ckpt_basename, ckpt_format,
        num_classes,
        ce_weight        (default 1.0),
        dice_weight      (default 1.0),
        ignore_index     (default -1, set to e.g. 255 for VOC-style void),
    """

    def __init__(
        self,
        model,
        model_name,
        train_loader,
        val_loader,
        config,
    ):
        self.config = config
        self.device = config["device"]

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_classes = config["num_classes"]

        print("Preparing training...")

        self.model = model.to(self.device)

        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = SegmentationLoss(
            num_classes=self.num_classes,
            ce_weight=config.get("ce_weight", 1.0),
            dice_weight=config.get("dice_weight", 1.0),
            label_smoothing=config.get("label_smoothing", 0.0),
            ignore_index=config.get("ignore_index", -1),
        )

        self.optimizer = AdamW(
            self.model.parameters(),
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

        self.history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_dice_loss": [],
            "val_loss": [],
            "val_ce_loss": [],
            "val_dice_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    @staticmethod
    def _flatten(preds, labels, ignore_index=-1):
        """Flatten spatial dims and optionally drop ignore_index pixels."""
        p = preds.view(-1).cpu().tolist()
        l = labels.view(-1).cpu().tolist()

        if ignore_index >= 0:
            pairs = [(pp, ll) for pp, ll in zip(p, l) if ll != ignore_index]
            
            if pairs:
                p, l = zip(*pairs)
                return list(p), list(l)
            
            return [], []

        return p, l

    def fit(self):
        print("Starting training...\n")

        start_time = time.time()
        ignore_index = self.config.get("ignore_index", -1)

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

            global_train_loss = global_train_ce = global_train_dice = 0.0
            window_train_loss = window_train_ce = window_train_dice = 0.0

            window_train_preds, window_train_labels = [], []
            epoch_train_preds, epoch_train_labels = [], []

            train_step = 0

            batch_iter = tqdm(self.train_loader, desc=f"epoch {epoch}")

            for imgs, masks in batch_iter:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device).long()

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = self.model(imgs)
                    loss, ce_loss, dice_loss = self.criterion(logits, masks)

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )
                self.optimizer.step()

                preds = logits.argmax(dim=1)

                p_flat, l_flat = self._flatten(preds, masks, ignore_index)

                # append preds and labels
                window_train_preds.extend(p_flat)
                window_train_labels.extend(l_flat)
                epoch_train_preds.extend(p_flat)
                epoch_train_labels.extend(l_flat)

                # windowed loss
                window_train_loss += loss.item()
                window_train_ce += ce_loss.item()
                window_train_dice += dice_loss.item()

                # global loss
                global_train_loss += loss.item()
                global_train_ce += ce_loss.item()
                global_train_dice += dice_loss.item()

                train_step += 1
                batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

                if train_step % self.config["append_train_history_step"] == 0:
                    # avg loss
                    avg = window_train_loss / train_step

                    # append to history
                    self.history["train_loss"].append(round(avg, 6))
                    self.history["train_ce_loss"].append(
                        round(window_train_ce / train_step, 6)
                    )
                    self.history["train_dice_loss"].append(
                        round(window_train_dice / train_step, 6)
                    )
                    self.history["train_f1"].append(
                        f1(window_train_labels, window_train_preds)
                    )

                    # reset
                    window_train_loss = window_train_ce = window_train_dice = 0.0
                    window_train_preds, window_train_labels = [], []
                    train_step = 0

            self.scheduler.step()

            # eval
            self.model.eval()

            global_val_loss = global_val_ce = global_val_dice = 0.0
            window_val_loss = window_val_ce = window_val_dice = 0.0

            window_val_preds, window_val_labels = [], []
            epoch_val_preds, epoch_val_labels = [], []

            val_step = 0

            with torch.no_grad():
                for imgs, masks in tqdm(self.val_loader, desc="validation"):
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device).long()

                    with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        logits = self.model(imgs)
                        loss, ce_loss, dice_loss = self.criterion(logits, masks)

                    preds = logits.argmax(dim=1)

                    p_flat, l_flat = self._flatten(preds, masks, ignore_index)

                    # append preds and labels
                    window_val_preds.extend(p_flat)
                    window_val_labels.extend(l_flat)
                    epoch_val_preds.extend(p_flat)
                    epoch_val_labels.extend(l_flat)

                    # windowed loss
                    window_val_loss += loss.item()
                    window_val_ce += ce_loss.item()
                    window_val_dice += dice_loss.item()

                    # global loss
                    global_val_loss += loss.item()
                    global_val_ce += ce_loss.item()
                    global_val_dice += dice_loss.item()

                    val_step += 1

                    if val_step % self.config["append_val_history_step"] == 0:
                        # avg loss
                        avg = window_val_loss / val_step

                        # append to history
                        self.history["val_loss"].append(round(avg, 6))
                        self.history["val_ce_loss"].append(
                            round(window_val_ce / val_step, 6)
                        )
                        self.history["val_dice_loss"].append(
                            round(window_val_dice / val_step, 6)
                        )
                        self.history["val_f1"].append(
                            f1(window_val_labels, window_val_preds)
                        )

                        # reset
                        window_val_loss = window_val_ce = window_val_dice = 0.0
                        window_val_preds, window_val_labels = [], []
                        val_step = 0

            # global train loss
            global_train_loss /= len(self.train_loader)
            global_train_ce /= len(self.train_loader)
            global_train_dice /= len(self.train_loader)

            # global val loss
            global_val_loss /= len(self.val_loader)
            global_val_ce /= len(self.val_loader)
            global_val_dice /= len(self.val_loader)

            # compute epoch-level f1
            train_f1 = f1(epoch_train_labels, epoch_train_preds)
            val_f1 = f1(epoch_val_labels, epoch_val_preds)

            # IoU is the primary metrics
            train_miou = mean_iou(
                epoch_train_preds, epoch_train_labels, self.num_classes
            )
            val_miou = mean_iou(epoch_val_preds, epoch_val_labels, self.num_classes)

            # logging
            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{self.config['epochs']} - {time_formatter(epoch_time)} | "
                f"train_loss={global_train_loss:.6f} (ce={global_train_ce:.4f}, dice={global_train_dice:.4f}) | "
                f"val_loss={global_val_loss:.6f} (ce={global_val_ce:.4f}, dice={global_val_dice:.4f}) | "
                f"train_f1={train_f1:.4f} | val_f1={val_f1:.4f} | "
                f"train_mIoU={train_miou:.4f} | val_mIoU={val_miou:.4f}"
            )

            # callbacks
            model_dict = self.model.state_dict()

            optimizer_dict = {
                "adam": self.optimizer.state_dict(),
                "cosine_scheduler": self.scheduler.state_dict(),
            }

            metadata = {
                "epoch": epoch,
                "train_loss": global_train_loss,
                "val_loss": global_val_loss,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_miou": train_miou,
                "val_miou": val_miou,
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
