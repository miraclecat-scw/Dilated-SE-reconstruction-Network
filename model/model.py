"""
    @Author: Su changwei
    @Email: scw727@outlook.com
    @Description: Dilated SE reconstruction model for 2D TP fields.
    @File: model.py
"""

import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        weights = self.pool(x)
        weights = self.conv1(weights)
        weights = self.act(weights)
        weights = self.conv2(weights)
        weights = self.gate(weights)
        return x * weights


class DilatedSEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, se_reduction):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.se = SEBlock(out_channels, se_reduction)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.se(out)
        out = out + residual
        out = self.act(out)
        return out


class DilatedSEReconstructionNet(nn.Module):
    def __init__(self, in_channels, stem_channels, block_channels, dilations, se_reduction):
        super().__init__()

        if len(block_channels) != len(dilations):
            raise ValueError("block_channels and dilations must have the same length.")
        if len(block_channels) == 0:
            raise ValueError("block_channels must not be empty.")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
        )

        blocks = []
        current_channels = stem_channels
        for out_channels, dilation in zip(block_channels, dilations):
            blocks.append(
                DilatedSEResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    se_reduction=se_reduction,
                )
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(current_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
