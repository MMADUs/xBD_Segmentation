# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import argparse

from torch.utils.data import DataLoader, Subset

from src.utils import PROJECT_ROOT, DeviceDataLoader, set_random_seed
from src.dataset import XBDDataset
from src.unet import UNet
from src.deeplab import DeepLabV3Plus
from src.train import Trainer
from config.training import get_config
from config.transform import train_transform, val_transform

N_TEST_SAMPLES = 500


class CustomSubset(Subset):
    def labels(self):
        return [self.dataset.samples[i][1] for i in self.indices]


def main():
    config = get_config()

    set_random_seed(config["random_seed"])

    parser = argparse.ArgumentParser(description="trainer script arg parser")

    parser.add_argument(
        "--test",
        action="store_true",
        help="run in test mode with a small subset of data",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["unet"],
        choices=["unet", "deeplab"],
        help="model selection",
    )

    args = parser.parse_args()

    train_path = PROJECT_ROOT / ".dataset" / "train"
    val_path = PROJECT_ROOT / ".dataset" / "hold"

    train_ds = XBDDataset(
        root=train_path, patch_division=True, transform=train_transform
    )
    val_ds = XBDDataset(root=val_path, patch_division=True, transform=val_transform)

    if args.test:
        train_ds = CustomSubset(train_ds, list(range(N_TEST_SAMPLES)))
        val_ds = CustomSubset(val_ds, list(range(N_TEST_SAMPLES // 4)))

    batch_size = config["batch_size"]
    val_test_batch_size = batch_size // 2
    num_workers = config.get("num_workers", 0)

    train_dl = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=val_test_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = config["device"]

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    selected_models = args.model

    if "unet" in selected_models:
        unet = UNet(num_classes=XBDDataset.NUM_CLASSES).to(device)
        unet_trainer = Trainer(
            unet, "UNet", train_dl, val_dl, XBDDataset.NUM_CLASSES, config
        )
        unet_trainer.fit()

    if "deeplab" in selected_models:
        deeplab = DeepLabV3Plus(num_classes=XBDDataset.NUM_CLASSES).to(device)
        deeplab_trainer = Trainer(
            deeplab, "DeepLabV3+", train_dl, val_dl, XBDDataset.NUM_CLASSES, config
        )
        deeplab_trainer.fit()


# python script.py --test --model unet deeplab
if __name__ == "__main__":
    main()
