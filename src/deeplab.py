# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=256,
        rates=[1, 6, 12, 18],
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # 1x1 convolution block
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # dilated convolution blocks
        for r in rates:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=r,
                        dilation=r,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # image pooling block
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        in_project = out_channels * (
            len(rates) + 2
        )  # 1x1 conv + total rate + image pool = total rate + 2 (1x1 conv + image pool)

        self.project = nn.Sequential(
            nn.Conv2d(in_project, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        result = []

        for block in self.blocks:
            out = block(x)
            result.append(out)

        # pool and interpolate
        image_feat = self.image_pool(x)
        image_feat = F.interpolate(
            image_feat, size=(h, w), mode="bilinear", align_corners=True
        )
        result.append(image_feat)

        # concatenate and project
        result = torch.cat(result, dim=1)
        result = self.project(result)

        return result


class DeepLabDecoder(nn.Module):
    def __init__(
        self,
        low_channels,
        out_channels=48,
        final_channels=256,
    ):
        super().__init__()

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # concat low and high features for fuse
        low_high = out_channels + final_channels

        self.fuse = nn.Sequential(
            nn.Conv2d(low_high, final_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
            # idk why we have 2 conv here, maybe 1 works ?
            nn.Conv2d(
                final_channels, final_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_feat, high_feat):
        # project low-level features and upsample high-level features
        low = self.low_proj(low_feat)
        high = F.interpolate(
            high_feat, size=low.size()[2:], mode="bilinear", align_corners=True
        )

        # fuse low and high features
        fused = torch.cat([low, high], dim=1)
        fused = self.fuse(fused)

        return fused


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

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

        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self.decoder = DeepLabDecoder(
            low_channels=256, out_channels=48, final_channels=256
        )

        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        aspp_out = self.aspp(x4)

        decoder_out = self.decoder(x1, aspp_out)

        output = self.final_conv(decoder_out)
        output = F.interpolate(
            output, size=input_size, mode="bilinear", align_corners=False
        )

        return output


if __name__ == "__main__":
    model = DeepLabV3Plus(num_classes=4)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Output:", y.shape)
