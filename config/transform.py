# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.Resize(512, 512),

    # Geometric — safe for nadir satellite view
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, border_mode=0, p=0.4),   # keep small, buildings shouldn't tilt much

    # Photometric — conservative, preserve damage color cues
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.2),           # tighter than before
        contrast_limit=(-0.15, 0.15),
        p=0.5,
    ),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),   # mild only — blur can erase damage texture
    A.GaussNoise(p=0.2),

    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])