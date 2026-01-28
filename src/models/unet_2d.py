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


def pad_to_multiple(x, multiple_h, multiple_w):
    """
    Pads (H,W) so that H % multiple_h == 0 and W % multiple_w == 0.
    Pads only on bottom/right. Returns (x_padded, (left,right,top,bottom)).
    """
    if x.ndim != 4:
        raise ValueError("Expected x to have shape [B,C,H,W].")

    _, _, h, w = x.shape
    pad_h = (multiple_h - (h % multiple_h)) % multiple_h
    pad_w = (multiple_w - (w % multiple_w)) % multiple_w

    left = 0
    right = pad_w
    top = 0
    bottom = pad_h

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    x_pad = F.pad(x, (left, right, top, bottom), mode="replicate")
    return x_pad, (left, right, top, bottom)


def unpad(x, pad):
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

    def __init__(self, stems=4):
        super().__init__()
        if stems <= 0:
            raise ValueError("stems must be positive")
        self.stems = stems

        self.enc1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.5),
        )

        self.up4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec4 = ConvBlock(256 + 256, 128)

        self.up3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec3 = ConvBlock(128 + 128, 64)

        self.up2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec2 = ConvBlock(64 + 64, 32)

        self.up1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec1 = ConvBlock(32 + 32, 16)

        self.out_conv = nn.Conv2d(16, stems, kernel_size=1)
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

        u4 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.up4(u4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_act(self.out_conv(d1))
        out = unpad(out, pad)
        return out
