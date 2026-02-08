from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


def pad_to_multiple(
    x: torch.Tensor,
    multiple_h: int,
    multiple_w: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Pads (H,W) so that H % multiple_h == 0 and W % multiple_w == 0.
    Pads only on bottom/right. Returns (x_padded, (left,right,top,bottom)).
    """
    if x.ndim != 4:
        raise ValueError("Expected x to have shape [B,C,H,W].")

    _, _, h, w = x.shape
    h = int(h)
    w = int(w)
    pad_h = int((multiple_h - (h % multiple_h)) % multiple_h)
    pad_w = int((multiple_w - (w % multiple_w)) % multiple_w)

    left = 0
    right = pad_w
    top = 0
    bottom = pad_h

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    pad = [int(left), int(right), int(top), int(bottom)]
    x_pad = F.pad(x, pad, mode="replicate")
    return x_pad, (left, right, top, bottom)


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return x
    _, _, h, w = x.shape
    new_h = h - top - bottom
    new_w = w - left - right
    return x[:, :, top : top + new_h, left : left + new_w]


class UNet2D(nn.Module):
    """
    U-Net for spectrogram masks.
    Input:  [B, 1, F, T]
    Output: [B, S, F, T] with sigmoid (0..1)
    """

    def __init__(self, stems: int = 4, base_channels: int = 64):
        super().__init__()
        if stems <= 0:
            raise ValueError("stems must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.stems = stems
        self.base_channels = base_channels

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.enc1 = ConvBlock(1, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c4, c5)

        self.up4 = nn.Conv2d(c5, c4, kernel_size=1)
        self.dec4 = ConvBlock(c5, c4)

        self.up3 = nn.Conv2d(c4, c3, kernel_size=1)
        self.dec3 = ConvBlock(c4, c3)

        self.up2 = nn.Conv2d(c3, c2, kernel_size=1)
        self.dec2 = ConvBlock(c3, c2)

        self.up1 = nn.Conv2d(c2, c1, kernel_size=1)
        self.dec1 = ConvBlock(c2, c1)

        self.out_conv = nn.Conv2d(c1, stems, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Input must have shape [B, 1, F, T]")
        if x.shape[1] != 1:
            raise ValueError("Channel dimension must be 1 (mono spectrogram)")

        x_pad, pad = pad_to_multiple(x, 16, 16)

        e1 = self.enc1(x_pad)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=False)
        u4 = self.up4(u4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = F.interpolate(d4, scale_factor=2.0, mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, scale_factor=2.0, mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, scale_factor=2.0, mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_act(self.out_conv(d1))
        out = unpad(out, pad)
        return out
