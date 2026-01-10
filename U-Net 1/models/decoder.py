"""
DecoderBlock: Upsampling block for U-Net architecture

This module implements the decoder (upsampling) portion of the U-Net.
Each decoder block upsamples spatial dimensions and integrates skip connections
from the corresponding encoder layer.

The decoder uses:
- ConvTranspose2d for 2x upsampling
- Concatenation with skip connection from encoder
- Two convolutional layers with batch normalization and ReLU
- Optional dropout for regularization

Architecture:
    Input -> ConvTranspose (upsample) -> Concat(skip) -> Conv1 -> BN1 -> ReLU
         -> Conv2 -> BN2 -> ReLU -> Dropout -> Output

Usage:
    decoder = DecoderBlock(in_channels=32, out_channels=16)
    x = torch.randn(4, 32, 128, 128)      # From previous decoder/bottleneck
    skip = torch.randn(4, 16, 256, 256)   # From encoder
    output = decoder(x, skip)
    print(output.shape)  # (4, 16, 256, 256) - spatial dims doubled

Author: Q1 U-Net Implementation
"""

import torch
import torch.nn as nn
from typing import Optional


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net architecture.

    Performs upsampling and skip connection integration:
    - ConvTranspose2d for 2x spatial upsampling
    - Concatenation with encoder skip connection
    - Two convolution blocks (Conv -> BatchNorm -> ReLU)
    - Optional dropout for regularization

    Args:
        in_channels: Number of input channels from previous layer
        out_channels: Number of output channels
        skip_channels: Number of channels in skip connection.
            If None, assumes skip_channels = out_channels. Default: None
        kernel_size: Size of convolutional kernels. Default: 3
        dropout_prob: Dropout probability (0 = no dropout). Default: 0.0

    Shape:
        Input x: (batch, in_channels, H, W)
        Input skip: (batch, skip_channels, H*2, W*2)
        Output: (batch, out_channels, H*2, W*2)

    Example:
        >>> decoder = DecoderBlock(in_channels=32, out_channels=16)
        >>> x = torch.randn(1, 32, 128, 128)
        >>> skip = torch.randn(1, 16, 256, 256)
        >>> output = decoder(x, skip)
        >>> print(output.shape)
        torch.Size([1, 16, 256, 256])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: Optional[int] = None,
        kernel_size: int = 3,
        dropout_prob: float = 0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # If skip_channels not specified, assume same as out_channels
        if skip_channels is None:
            skip_channels = out_channels
        self.skip_channels = skip_channels

        # Transposed convolution for upsampling
        # kernel_size=2, stride=2 doubles spatial dimensions exactly
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )

        # After concatenation: out_channels (from upsample) + skip_channels
        concat_channels = out_channels + skip_channels
        padding = kernel_size // 2

        # First convolution: processes concatenated features
        # Reduces channels from concat_channels -> out_channels
        self.conv1 = nn.Conv2d(
            in_channels=concat_channels,
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

        # Second convolution: refines features
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

        # ReLU activation (standard ReLU in decoder, not LeakyReLU)
        self.activation = nn.ReLU(inplace=True)

        # Dropout for regularization (prevents overfitting)
        if dropout_prob > 0:
            self.dropout = nn.Dropout2d(p=dropout_prob)
        else:
            self.dropout = nn.Identity()  # No-op when dropout_prob=0

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.

        Args:
            x: Input tensor from previous layer (batch, in_channels, H, W)
            skip: Skip connection from encoder (batch, skip_channels, H*2, W*2)

        Returns:
            Output tensor (batch, out_channels, H*2, W*2)

        Note:
            The skip connection must have spatial dimensions exactly 2x the input.
            This is ensured when encoder and decoder are properly paired.
        """
        # Upsample: (batch, in_channels, H, W) -> (batch, out_channels, H*2, W*2)
        x = self.upsample(x)

        # Concatenate with skip connection along channel dimension
        # x: (batch, out_channels, H*2, W*2)
        # skip: (batch, skip_channels, H*2, W*2)
        # result: (batch, out_channels + skip_channels, H*2, W*2)
        x = torch.cat([x, skip], dim=1)

        # First conv block: reduce channels
        # Shape: (batch, concat_channels, H*2, W*2) -> (batch, out_channels, H*2, W*2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        # Second conv block: refine features
        # Shape: (batch, out_channels, H*2, W*2) -> (batch, out_channels, H*2, W*2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        return x


def test_decoder_block():
    """Test the DecoderBlock implementation."""
    print("=" * 60)
    print("Testing DecoderBlock")
    print("=" * 60)

    # Test 1: Standard decoder block
    print("\n1. Standard DecoderBlock")
    decoder = DecoderBlock(in_channels=32, out_channels=16)

    x = torch.randn(4, 32, 128, 128)
    skip = torch.randn(4, 16, 256, 256)
    print(f"   Input shape: {x.shape}")
    print(f"   Skip shape: {skip.shape}")

    output = decoder(x, skip)
    print(f"   Output shape: {output.shape}")

    assert output.shape == (4, 16, 256, 256), f"Output shape incorrect: {output.shape}"
    print("   [PASS] Shape correct!")

    # Test 2: With dropout
    print("\n2. DecoderBlock with dropout")
    decoder_dropout = DecoderBlock(
        in_channels=32,
        out_channels=16,
        dropout_prob=0.5
    )
    output_dropout = decoder_dropout(x, skip)
    print(f"   Output shape with dropout: {output_dropout.shape}")
    assert output_dropout.shape == (4, 16, 256, 256), "Dropout shape incorrect"
    print("   [PASS] Dropout works!")

    # Test 3: Different skip channel count
    print("\n3. DecoderBlock with different skip channels")
    decoder_diff = DecoderBlock(
        in_channels=32,
        out_channels=16,
        skip_channels=8
    )
    x3 = torch.randn(4, 32, 128, 128)
    skip3 = torch.randn(4, 8, 256, 256)
    output3 = decoder_diff(x3, skip3)
    print(f"   Input: {x3.shape}, Skip: {skip3.shape} -> Output: {output3.shape}")
    assert output3.shape == (4, 16, 256, 256), "Different skip channels failed"
    print("   [PASS] Different skip channels work!")

    # Test 4: Gradient flow
    print("\n4. Test gradient flow")
    x4 = torch.randn(2, 32, 64, 64, requires_grad=True)
    skip4 = torch.randn(2, 16, 128, 128, requires_grad=True)
    output4 = decoder(x4, skip4)
    loss = output4.sum()
    loss.backward()
    assert x4.grad is not None, "No gradient for input"
    assert skip4.grad is not None, "No gradient for skip"
    print("   [PASS] Gradients flow through input and skip!")

    # Test 5: Eval mode with batch_size=1
    print("\n5. Test eval mode with batch_size=1")
    decoder.eval()
    x5 = torch.randn(1, 32, 128, 128)
    skip5 = torch.randn(1, 16, 256, 256)
    with torch.no_grad():
        output5 = decoder(x5, skip5)
    print(f"   Input: {x5.shape} -> Output: {output5.shape}")
    print("   [PASS] Eval mode works with batch_size=1!")

    # Test 6: Shape mismatch detection
    print("\n6. Test shape mismatch detection")
    try:
        wrong_skip = torch.randn(4, 16, 128, 128)  # Wrong spatial size
        decoder.train()
        output_wrong = decoder(x, wrong_skip)
        print("   [FAIL] Should have caught shape mismatch!")
    except RuntimeError as e:
        print(f"   [PASS] Correctly caught shape mismatch: {type(e).__name__}")

    # Test 7: Parameter count
    print("\n7. Parameter count")
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"   DecoderBlock(32->16) has {num_params:,} parameters")

    print("\n" + "=" * 60)
    print("All DecoderBlock tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_decoder_block()
