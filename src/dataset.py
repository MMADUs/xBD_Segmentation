# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import json
import os

import cv2
import numpy as np
from shapely import wkt
from torch.utils.data import Dataset

DAMAGE_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

# patch grid: (row_start, row_end, col_start, col_end)
PATCH_COORDS = [
    (0, 512, 0, 512),  # top-left
    (0, 512, 512, 1024),  # top-right
    (512, 1024, 0, 512),  # bottom-left
    (512, 1024, 512, 1024),  # bottom-right
]


class XBDDataset(Dataset):
    """
    Torch dataset for xBD dataset in patches.
    """

    NUM_CLASSES = 5  # 0 = background, 1-4 = damage levels
    PATCH_SIZE = 512
    IMAGE_SIZE = 1024
    NUM_PATCHES = len(PATCH_COORDS)  # 4

    def __init__(self, root: str, patch_division=False, transform=None):
        self.root = root
        self.transform = transform
        self.patch_division = patch_division

        img_dir = os.path.join(root, "images")
        label_dir = os.path.join(root, "labels")

        valid_ext = (".png", ".jpg", ".jpeg")

        # (image_path, label_path) pairs
        self.samples: list[tuple[str, str]] = []

        for filename in sorted(os.listdir(img_dir)):
            # skip if not supported image format
            if not filename.lower().endswith(valid_ext):
                continue

            stem = os.path.splitext(filename)[0]
            img_path = os.path.join(img_dir, filename)
            label_path = os.path.join(label_dir, stem + ".json")

            # skip images without labels
            if not os.path.isfile(label_path):
                continue

            self.samples.append((img_path, label_path))

        # self._index maps flat idx → (sample_idx, patch_idx)
        if self.patch_division:
            # each sample expands into NUM_PATCHES entries
            self._index: list[tuple[int, int]] = [
                (s, p)
                for s in range(len(self.samples))
                for p in range(self.NUM_PATCHES)
            ]
        else:
            # each sample is a single entry, patch_idx unused
            self._index: list[tuple[int, int]] = [
                (s, 0) for s in range(len(self.samples))
            ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        sample_idx, patch_idx = self._index[idx]
        img_path, label_path = self.samples[sample_idx]

        image = self._load_image(img_path)
        mask = self._load_mask(label_path)

        if self.patch_division:
            row_0, row_1, col_0, col_1 = PATCH_COORDS[patch_idx]

            image = image[row_0:row_1, col_0:col_1]
            mask = mask[row_0:row_1, col_0:col_1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)

            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

    def _load_image(self, path: str) -> np.ndarray:
        """
        Load image as RGB uint8 (H, W, 3)
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: str) -> np.ndarray:
        """
        Build a multi-class segmentation mask from an xBD label JSON
        """
        with open(path, "r") as f:
            data = json.load(f)

        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)

        for feat in data["features"]["xy"]:
            subtype = feat["properties"].get("subtype", "")
            class_id = DAMAGE_MAP.get(subtype)

            # skip unknown subtypes
            if class_id is None:
                continue

            polygon = wkt.loads(feat["wkt"])

            if polygon.is_empty:
                continue

            coords = np.array(list(polygon.exterior.coords), dtype=np.int32)
            cv2.fillPoly(mask, [coords.reshape(-1, 1, 2)], class_id)

        return mask

    def get_num_images(self) -> int:
        """
        Number of raw images (before patching)
        """
        return len(self.samples)

    def get_class_names(self) -> dict[int, str]:
        """
        Maps class id -> damage label
        """
        return {
            0: "background", 
            **{v: k for k, v in DAMAGE_MAP.items()},
        }

    # def get_patch_coords(self) -> list[tuple[int, int, int, int]]:
    #     """Returns the (r0, r1, c0, c1) crop coordinates for each patch slot."""
    #     return PATCH_COORDS
