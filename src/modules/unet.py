# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Decoder block that upsamples features and fuses with skip connections from the encoder.

    Args:
        in_channels: number of input channels from the previous layer
        skip_channels: number of channels from the skip connection
        out_channels: number of output channels for this block
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # concatenate upsampled features with skip connection features
        in_plus_skip = in_channels + skip_channels

        self.conv1 = nn.Conv2d(in_plus_skip, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up_sample(x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class UNet(nn.Module):
    """
    UNet Model.

    Args:
        num_classes: number of output classes for segmentation
    """

    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.decoder4 = DecoderBlock(
            in_channels=2048,
            skip_channels=1024,
            out_channels=512,
        )
        self.decoder3 = DecoderBlock(
            in_channels=512,
            skip_channels=512,
            out_channels=256,
        )
        self.decoder2 = DecoderBlock(
            in_channels=256,
            skip_channels=256,
            out_channels=128,
        )
        self.decoder1 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)

        out = self.final_up(d1)
        out = self.segmentation_head(out)

        return out


if __name__ == "__main__":
    model = UNet(num_classes=4)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Output shape:", y.shape)  # expect (2, 4, 512, 512)
