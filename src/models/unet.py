# src/models/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv   = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        # pad if needed
        if x.shape != skip.shape:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,
                 in_ch: int          = 1,
                 num_sources: int    = 4,
                 chans: int          = 32,
                 num_pool_layers: int= 4):
        super().__init__()

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        ch = chans
        # first layer
        self.down_blocks.append(ConvBlock(in_ch, ch))
        # further downsampling
        for _ in range(1, num_pool_layers):
            self.down_blocks.append(DownBlock(ch, ch*2))
            ch *= 2

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(ch, ch*2)
        ch *= 2  # now channel count matches bottleneck output

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for _ in range(num_pool_layers):
            # in_ch = two times the skip channels
            self.up_blocks.append(UpBlock(ch, ch//2))
            ch //= 2

        # ch now equals the number of channels output by the last UpBlock
        # --- Final conv ---
        self.final_conv = nn.Conv2d(ch, num_sources, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.final_conv(x)
