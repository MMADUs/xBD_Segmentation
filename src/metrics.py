# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch


def update_confusion_matrix(confmat, preds, labels, num_classes, ignore_index=-1):
    """
    Update a confusion matrix from segmentation predictions without storing pixels.
    """
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    if ignore_index >= 0:
        keep = labels != ignore_index
        preds = preds[keep]
        labels = labels[keep]

    valid = (
        (labels >= 0)
        & (labels < num_classes)
        & (preds >= 0)
        & (preds < num_classes)
    )

    if valid.any().item():
        indices = num_classes * labels[valid] + preds[valid]
        confmat += torch.bincount(
            indices,
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)

    return confmat


def compute_training_metrics(confmat, damage_class_ids=(2, 3, 4)):
    """
    Compute compact training metrics for xBD segmentation.
    """
    confmat = confmat.float()

    true_positive = confmat.diag()
    false_positive = confmat.sum(dim=0) - true_positive
    false_negative = confmat.sum(dim=1) - true_positive

    iou_denominator = true_positive + false_positive + false_negative
    f1_denominator = (2 * true_positive) + false_positive + false_negative

    iou = torch.zeros_like(true_positive)
    f1 = torch.zeros_like(true_positive)

    iou_valid = iou_denominator > 0
    f1_valid = f1_denominator > 0

    iou[iou_valid] = true_positive[iou_valid] / iou_denominator[iou_valid]
    f1[f1_valid] = (2 * true_positive[f1_valid]) / f1_denominator[f1_valid]

    present = iou_denominator > 0

    building_true = confmat[1:, :].sum()
    building_pred = confmat[:, 1:].sum()
    building_tp = confmat[1:, 1:].sum()
    building_denominator = building_true + building_pred - building_tp
    building_iou = (
        (building_tp / building_denominator).item()
        if building_denominator.item() > 0
        else 0.0
    )

    damage_miou = 0.0
    damage_ids = torch.as_tensor(
        damage_class_ids,
        device=confmat.device,
        dtype=torch.long,
    )
    damage_ids = damage_ids[damage_ids < confmat.shape[0]]

    if damage_ids.numel() > 0:
        damage_present = present[damage_ids]
        if damage_present.any().item():
            damage_miou = iou[damage_ids][damage_present].mean().item()

    return {
        "f1": f1[present].mean().item() if present.any().item() else 0.0,
        "miou": iou[present].mean().item() if present.any().item() else 0.0,
        "building_iou": building_iou,
        "damage_miou": damage_miou,
    }
