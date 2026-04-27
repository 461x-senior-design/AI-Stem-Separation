# src/stemmy/models/unet_2d.py
"""2D U-Net for stereo spectrogram mask-logit prediction.

Input:
  x: [B, C, F, T]  (stereo magnitude spectrogram, C=2)

Output:
  logits: [B, S, C, F, T] (unnormalized stem logits)

Notes:
- Spatial padding is applied so the encoder/decoder pooling hierarchy works for
  arbitrary (F, T) sizes. Padding is removed before returning.
- This model intentionally returns raw logits. Training/inference are responsible
  for applying softmax over stems when probability masks are required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stemmy.constants import TARGET_CHANNELS


class ConvBlock(nn.Module):
    """Two-layer convolutional block used throughout the U-Net.

    Each block applies:
      Conv2d -> BatchNorm2d -> LeakyReLU
      Conv2d -> BatchNorm2d -> LeakyReLU
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize the block.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.

        Raises:
            ValueError: If in_ch or out_ch is non-positive.
        """
        super().__init__()
        if in_ch <= 0:
            raise ValueError("in_ch must be positive.")
        if out_ch <= 0:
            raise ValueError("out_ch must be positive.")

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block forward.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, out_ch, H, W].
        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


def pad_to_multiple(
    x: torch.Tensor,
    multiple_h: int,
    multiple_w: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad spatial dimensions so H and W are divisible by given multiples.

    Padding is applied only on the bottom and right to avoid shifting the origin.

    Args:
        x: Tensor [B, C, H, W].
        multiple_h: Height multiple (> 0).
        multiple_w: Width multiple (> 0).

    Returns:
        Tuple of:
          - x_padded: padded tensor
          - pad: (left, right, top, bottom)

    Raises:
        ValueError: If x does not have 4 dimensions or multiples are invalid.
    """
    if x.ndim != 4:
        raise ValueError("Expected x to have shape [B, C, H, W].")
    if multiple_h <= 0:
        raise ValueError("multiple_h must be > 0.")
    if multiple_w <= 0:
        raise ValueError("multiple_w must be > 0.")

    _, _, h, w = x.shape
    h_i = int(h)
    w_i = int(w)

    pad_h = int((multiple_h - (h_i % multiple_h)) % multiple_h)
    pad_w = int((multiple_w - (w_i % multiple_w)) % multiple_w)

    left = 0
    right = pad_w
    top = 0
    bottom = pad_h

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    pad = [int(left), int(right), int(top), int(bottom)]
    x_pad = F.pad(x, pad, mode="replicate")
    return x_pad, (left, right, top, bottom)


def unpad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """Remove padding applied by pad_to_multiple.

    Args:
        x: Tensor [B, C, H, W].
        pad: (left, right, top, bottom).

    Returns:
        Unpadded tensor.

    Raises:
        ValueError: If x is not 4D or pad values are invalid.
    """
    if x.ndim != 4:
        raise ValueError("Expected x to have shape [B, C, H, W].")

    left, right, top, bottom = pad
    if any(int(v) < 0 for v in (left, right, top, bottom)):
        raise ValueError("pad values must be >= 0.")

    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return x

    _, _, h, w = x.shape
    h_i = int(h)
    w_i = int(w)

    new_h = h_i - int(top) - int(bottom)
    new_w = w_i - int(left) - int(right)

    if new_h <= 0 or new_w <= 0:
        raise ValueError(
            "Invalid unpad resulting shape: new_h=%d new_w=%d (from h=%d w=%d pad=%s)"
            % (int(new_h), int(new_w), int(h_i), int(w_i), str(pad))
        )

    return x[:, :, int(top) : int(top) + int(new_h), int(left) : int(left) + int(new_w)]


def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Upsample x spatially to match ref (H, W)."""
    if x.ndim != 4 or ref.ndim != 4:
        raise ValueError("Expected 4D tensors for upsampling.")
    _, _, h, w = ref.shape
    return F.interpolate(x, size=(int(h), int(w)), mode="bilinear", align_corners=False)


class UNet2D(nn.Module):
    """U-Net for stereo spectrogram mask-logit prediction."""

    def __init__(
        self,
        stems: int = 4,
        base_channels: int = 64,
        audio_channels: int = TARGET_CHANNELS,
    ) -> None:
        """Initialize the U-Net.

        Args:
            stems: Number of output stems.
            base_channels: Base channel width for the first encoder level.
            audio_channels: Number of input/output audio channels.

        Raises:
            ValueError: If stems, base_channels, or audio_channels is non-positive.
        """
        super().__init__()
        if stems <= 0:
            raise ValueError("stems must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if audio_channels <= 0:
            raise ValueError("audio_channels must be positive.")

        self.stems = int(stems)
        self.base_channels = int(base_channels)
        self.audio_channels = int(audio_channels)

        c1 = self.base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.enc1 = ConvBlock(self.audio_channels, c1)
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

        self.out_conv = nn.Conv2d(c1, self.stems * self.audio_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the U-Net forward.

        Args:
            x: Input tensor [B, C, F, T].

        Returns:
            Output logits [B, S, C, F, T] (unnormalized).

        Raises:
            ValueError: If x does not have expected shape.
        """
        if x.ndim != 4:
            raise ValueError("Expected input x to be 4D [B, C, F, T].")
        if int(x.shape[1]) != int(self.audio_channels):
            raise ValueError(
                "Expected channel dimension to be %d, got %d."
                % (int(self.audio_channels), int(x.shape[1]))
            )

        x_pad, pad = pad_to_multiple(x, multiple_h=16, multiple_w=16)

        e1 = self.enc1(x_pad)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = _upsample_to(u4, e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        u3 = _upsample_to(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = _upsample_to(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = _upsample_to(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        out = unpad(out, pad)

        batch, channels, freq, time = out.shape
        expected_channels = self.stems * self.audio_channels
        if int(channels) != int(expected_channels):
            raise RuntimeError(
                "Unexpected output channels: got %d, expected %d."
                % (int(channels), int(expected_channels))
            )

        out = out.contiguous().view(
            int(batch),
            int(self.stems),
            int(self.audio_channels),
            int(freq),
            int(time),
        )
        return out
