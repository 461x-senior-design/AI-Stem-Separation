"""
EncoderBlock: Downsampling block for U-Net architecture

This module implements the encoder (downsampling) portion of the U-Net.
Each encoder block extracts features and reduces spatial dimensions by half.

The encoder uses:
- Two convolutional layers with batch normalization and LeakyReLU activation
- MaxPool2d for downsampling (halves spatial dimensions)
- Skip connection output for decoder path

Architecture:
    Input -> Conv1 -> BN1 -> LeakyReLU -> Conv2 -> BN2 -> LeakyReLU -> (Skip) -> MaxPool -> Output

Usage:
    encoder = EncoderBlock(in_channels=2, out_channels=16)
    x = torch.randn(4, 2, 256, 256)  # Batch of stereo spectrograms
    encoded, skip = encoder(x)
    print(encoded.shape)  # (4, 16, 128, 128) - spatial dims halved
    print(skip.shape)     # (4, 16, 256, 256) - saved for decoder

Author: Q1 U-Net Implementation
"""

import torch
import torch.nn as nn
from typing import Tuple


class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net architecture.

    Performs feature extraction and downsampling:
    - Two convolution blocks (Conv -> BatchNorm -> LeakyReLU)
    - MaxPool2d for 2x spatial downsampling
    - Returns both downsampled output and skip connection

    Args:
        in_channels: Number of input channels (e.g., 2 for stereo)
        out_channels: Number of output feature channels
        kernel_size: Size of convolutional kernels. Default: 3
        use_pooling: If True, use MaxPool2d. If False, use stride=2 conv. Default: True

    Shape:
        Input: (batch, in_channels, H, W)
        Output encoded: (batch, out_channels, H/2, W/2)
        Output skip: (batch, out_channels, H, W)

    Example:
        >>> encoder = EncoderBlock(in_channels=2, out_channels=16)
        >>> x = torch.randn(1, 2, 256, 256)
        >>> encoded, skip = encoder(x)
        >>> print(encoded.shape, skip.shape)
        torch.Size([1, 16, 128, 128]) torch.Size([1, 16, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_pooling: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate padding to maintain spatial dimensions through convolutions
        padding = kernel_size // 2

        # First convolution block: in_channels -> out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=0.01
        )

        # Second convolution block: out_channels -> out_channels (refines features)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=0.01
        )

        # LeakyReLU activation prevents "dying ReLU" problem
        # Negative slope allows small gradients for negative inputs
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Downsampling: MaxPool vs Stride
        if use_pooling:
            # MaxPool2d: Takes maximum value in 2x2 windows
            # Preserves strongest activations
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # Alternative: Use stride=2 in convolution
            # More efficient, learns downsampling filter
            self.downsample = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W)

        Returns:
            Tuple of (encoded, skip):
                encoded: Downsampled output (batch, out_channels, H/2, W/2)
                skip: Pre-pooling features (batch, out_channels, H, W)
                      This is passed to corresponding decoder block
        """
        # First conv block: in_channels -> out_channels
        # Shape: (batch, in_channels, H, W) -> (batch, out_channels, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        # Second conv block: refine features
        # Shape: (batch, out_channels, H, W) -> (batch, out_channels, H, W)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Save for skip connection BEFORE downsampling
        # This preserves full spatial resolution for decoder
        skip = x

        # Downsample: halve spatial dimensions
        # Shape: (batch, out_channels, H, W) -> (batch, out_channels, H/2, W/2)
        encoded = self.downsample(skip)

        return encoded, skip


def test_encoder_block():
    """Test the EncoderBlock implementation."""
    print("=" * 60)
    print("Testing EncoderBlock")
    print("=" * 60)

    # Test 1: Standard encoder block
    print("\n1. Standard EncoderBlock (with skip connection)")
    encoder = EncoderBlock(in_channels=2, out_channels=16)

    x = torch.randn(4, 2, 256, 256)
    print(f"   Input shape: {x.shape}")

    encoded, skip = encoder(x)
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Skip shape: {skip.shape}")

    assert encoded.shape == (4, 16, 128, 128), f"Encoded shape incorrect: {encoded.shape}"
    assert skip.shape == (4, 16, 256, 256), f"Skip shape incorrect: {skip.shape}"
    print("   [PASS] Shapes correct!")

    # Test 2: Different input size
    print("\n2. Test with different input size")
    x2 = torch.randn(1, 2, 512, 512)
    encoded2, skip2 = encoder(x2)
    print(f"   Input: {x2.shape} -> Encoded: {encoded2.shape}, Skip: {skip2.shape}")
    assert encoded2.shape == (1, 16, 256, 256), "Shape scaling incorrect"
    print("   [PASS] Scales correctly!")

    # Test 3: Gradient flow
    print("\n3. Test gradient flow")
    x3 = torch.randn(2, 2, 128, 128, requires_grad=True)
    encoded3, skip3 = encoder(x3)
    loss = encoded3.sum() + skip3.sum()
    loss.backward()
    assert x3.grad is not None, "No gradient for input"
    print("   [PASS] Gradients flow correctly!")

    # Test 4: Eval mode with batch_size=1
    print("\n4. Test eval mode with batch_size=1")
    encoder.eval()
    x4 = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        encoded4, skip4 = encoder(x4)
    print(f"   Input: {x4.shape} -> Encoded: {encoded4.shape}")
    print("   [PASS] Eval mode works with batch_size=1!")

    # Test 5: Parameter count
    print("\n5. Parameter count")
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"   EncoderBlock(2->16) has {num_params:,} parameters")

    print("\n" + "=" * 60)
    print("All EncoderBlock tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_encoder_block()
