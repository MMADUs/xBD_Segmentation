# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss averaged over classes (macro).

    Args:
        num_classes: 
            number of classes in the segmentation task
        smooth: 
            smoothing factor to avoid division by zero
        ignore_index: 
            class index to ignore in the loss computation
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        ignore_index: int = -1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits:
                (B, C, H, W) raw model outputs
            targets:
                (B, H, W) integer class labels
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # one-hot encode targets -> (B, C, H, W)
        targets_oh = F.one_hot(targets.clamp(min=0), self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # optionally mask out ignore_index pixels
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
            probs = probs * mask
            targets_oh = targets_oh * mask

        # flatten spatial dims
        probs_flat = probs.view(probs.shape[0], self.num_classes, -1)  # (B, C, N)
        targets_flat = targets_oh.view(targets_oh.shape[0], self.num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_per_class.mean()


class SegmentationLoss(nn.Module):
    """
    The multiclass semantic segmentation loss consist of a combined Cross Entropy + Macro Dice loss

    Args:
        num_classes: 
            number of classes in the segmentation task
        ce_weight: 
            weight for the cross-entropy loss
        dice_weight: 
            weight for the Dice loss
        label_smoothing: 
            label smoothing factor
        ignore_index: 
            class index to ignore in the loss computation
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -1,
        class_weights: list[float] | None = None,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights)
        else:
            self.class_weights = None

        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )
        dice_loss = self.dice(logits, targets)

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss, ce_loss, dice_loss
