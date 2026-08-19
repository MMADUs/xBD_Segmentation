# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Adapt an ImageNet pretrained RGB conv to 3-channel or 6-channel input.
    """
    if in_channels == conv.in_channels:
        return conv

    if in_channels != 6 or conv.in_channels != 3:
        raise ValueError(
            f"Only 3-channel or 6-channel input is supported, got {in_channels}"
        )

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = conv.weight / 2.0
        new_conv.weight[:, 3:] = conv.weight / 2.0

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv
