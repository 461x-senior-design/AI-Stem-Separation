"""
Complete U-Net Implementation for Audio Source Separation

This module implements a full U-Net architecture that combines:
- EncoderBlock (downsampling path with feature extraction)
- DecoderBlock (upsampling path with skip connection integration)
- Bottleneck layer (deepest representation)
- Skip connections (preserve spatial information)
- Output projection (final 1x1 convolution)

The U-Net architecture forms a symmetric encoder-decoder structure:

    Input (2, 256, 256)
        |
    Encoder1 -> skip1 (16, 256, 256)
        |
    Encoder2 -> skip2 (32, 128, 128)
        |
    Encoder3 -> skip3 (64, 64, 64)
        |
    Encoder4 -> skip4 (128, 32, 32)
        |
    Bottleneck (256, 16, 16)
        |
    Decoder4 + skip4 -> (128, 32, 32)
        |
    Decoder3 + skip3 -> (64, 64, 64)
        |
    Decoder2 + skip2 -> (32, 128, 128)
        |
    Decoder1 + skip1 -> (16, 256, 256)
        |
    Final Conv (2, 256, 256)
        |
    Output (same shape as input)

Usage:
    from models import UNet

    model = UNet(input_channels=2, output_channels=2, initial_filters=16)
    x = torch.randn(4, 2, 256, 256)  # Batch of stereo spectrograms
    output = model(x)                 # (4, 2, 256, 256) - same shape!

    # Check parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

Author: Q1 U-Net Implementation
"""

import torch
import torch.nn as nn
from typing import List

from .encoder import EncoderBlock
from .decoder import DecoderBlock


class UNet(nn.Module):
    """
    Complete U-Net architecture for audio source separation.

    Architecture consists of:
    1. Encoder path: 4 encoder blocks, each doubling channels and halving spatial dims
    2. Bottleneck: Two conv layers at deepest representation (no spatial change)
    3. Decoder path: 4 decoder blocks, each halving channels and doubling spatial dims
    4. Skip connections: Encoder outputs connected to corresponding decoder inputs
    5. Final layer: 1x1 convolution projecting to output channels

    Args:
        input_channels: Number of input channels (e.g., 2 for stereo). Default: 2
        output_channels: Number of output channels (e.g., 2 for separated stereo). Default: 2
        initial_filters: Number of filters in first encoder block. Default: 16
            Filter progression: 16 -> 32 -> 64 -> 128 -> 256 (bottleneck)
        depth: Number of encoder/decoder levels. Default: 4
            Note: Current implementation is optimized for depth=4
        use_dropout: Whether to use dropout in decoder blocks. Default: False
        dropout_prob: Dropout probability if use_dropout=True. Default: 0.5

    Shape:
        Input: (batch, input_channels, H, W)
            H and W should be divisible by 2^depth (16 for depth=4)
        Output: (batch, output_channels, H, W)
            Same spatial dimensions as input

    Example:
        >>> model = UNet(input_channels=2, output_channels=2)
        >>> x = torch.randn(4, 2, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([4, 2, 256, 256])
    """

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        initial_filters: int = 16,
        depth: int = 4,
        use_dropout: bool = False,
        dropout_prob: float = 0.5
    ):
        super().__init__()

        # Store configuration
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.initial_filters = initial_filters
        self.depth = depth

        f = initial_filters  # Shorthand for filter count

        # ==================================================================
        # ENCODER PATH (Downsampling)
        # ==================================================================
        # Each encoder block:
        # - Extracts features with two conv layers
        # - Doubles number of channels
        # - Halves spatial dimensions via MaxPool
        # - Returns (encoded, skip) tuple

        # Encoder 1: input_channels -> f (e.g., 2 -> 16)
        # Spatial: 256x256 -> 128x128
        self.enc1 = EncoderBlock(input_channels, f)

        # Encoder 2: f -> f*2 (e.g., 16 -> 32)
        # Spatial: 128x128 -> 64x64
        self.enc2 = EncoderBlock(f, f * 2)

        # Encoder 3: f*2 -> f*4 (e.g., 32 -> 64)
        # Spatial: 64x64 -> 32x32
        self.enc3 = EncoderBlock(f * 2, f * 4)

        # Encoder 4: f*4 -> f*8 (e.g., 64 -> 128)
        # Spatial: 32x32 -> 16x16
        self.enc4 = EncoderBlock(f * 4, f * 8)

        # ==================================================================
        # BOTTLENECK (Deepest representation)
        # ==================================================================
        # Processes the most compressed representation
        # No spatial downsampling - just feature refinement
        # Input: f*8 channels (128), Output: f*16 channels (256)

        bottleneck_in = f * 8    # 128 for initial_filters=16
        bottleneck_out = f * 16  # 256 for initial_filters=16

        self.bottleneck = nn.Sequential(
            # First conv block
            nn.Conv2d(
                in_channels=bottleneck_in,
                out_channels=bottleneck_out,
                kernel_size=3,
                stride=1,  # NO downsampling
                padding=1
            ),
            nn.BatchNorm2d(bottleneck_out, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),

            # Second conv block (refines features)
            nn.Conv2d(
                in_channels=bottleneck_out,
                out_channels=bottleneck_out,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(bottleneck_out, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        # ==================================================================
        # DECODER PATH (Upsampling + Skip Connections)
        # ==================================================================
        # Each decoder block:
        # - Upsamples spatial dimensions by 2x
        # - Concatenates with skip connection from corresponding encoder
        # - Refines features with two conv layers
        # - Halves number of channels

        # Decoder 4: bottleneck_out -> f*8, uses skip from enc4
        # Spatial: 16x16 -> 32x32
        self.dec4 = DecoderBlock(
            in_channels=bottleneck_out,  # 256
            out_channels=f * 8,          # 128
            skip_channels=f * 8,         # 128 from enc4
            dropout_prob=dropout_prob if use_dropout else 0.0
        )

        # Decoder 3: f*8 -> f*4, uses skip from enc3
        # Spatial: 32x32 -> 64x64
        self.dec3 = DecoderBlock(
            in_channels=f * 8,   # 128
            out_channels=f * 4,  # 64
            skip_channels=f * 4, # 64 from enc3
            dropout_prob=dropout_prob if use_dropout else 0.0
        )

        # Decoder 2: f*4 -> f*2, uses skip from enc2
        # Spatial: 64x64 -> 128x128
        self.dec2 = DecoderBlock(
            in_channels=f * 4,   # 64
            out_channels=f * 2,  # 32
            skip_channels=f * 2, # 32 from enc2
            dropout_prob=dropout_prob if use_dropout else 0.0
        )

        # Decoder 1: f*2 -> f, uses skip from enc1
        # Spatial: 128x128 -> 256x256
        self.dec1 = DecoderBlock(
            in_channels=f * 2,   # 32
            out_channels=f,      # 16
            skip_channels=f,     # 16 from enc1
            dropout_prob=dropout_prob if use_dropout else 0.0
        )

        # ==================================================================
        # FINAL OUTPUT LAYER
        # ==================================================================
        # 1x1 convolution to project from decoder channels to output channels
        # No activation - let training determine output range

        self.final = nn.Conv2d(
            in_channels=f,                # 16 from dec1
            out_channels=output_channels, # 2 for stereo output
            kernel_size=1                 # 1x1 conv for channel projection
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (batch, input_channels, H, W)
               Example: (4, 2, 256, 256) for batch of 4 stereo spectrograms

        Returns:
            Output tensor of shape (batch, output_channels, H, W)
            Same spatial dimensions as input

        Note:
            H and W must be divisible by 2^depth (16 for depth=4) to ensure
            proper downsampling/upsampling alignment.
        """
        # Store input shape for validation
        input_shape = x.shape

        # ==================================================================
        # ENCODER PATH: Downsample and save skip connections
        # ==================================================================
        # Skip connections are saved in a list for clean decoder wiring
        # Decoders use skips in REVERSE order (LIFO)

        skips: List[torch.Tensor] = []

        # Encoder Block 1
        # (batch, 2, 256, 256) -> (batch, 16, 128, 128), skip: (batch, 16, 256, 256)
        x, skip1 = self.enc1(x)
        skips.append(skip1)

        # Encoder Block 2
        # (batch, 16, 128, 128) -> (batch, 32, 64, 64), skip: (batch, 32, 128, 128)
        x, skip2 = self.enc2(x)
        skips.append(skip2)

        # Encoder Block 3
        # (batch, 32, 64, 64) -> (batch, 64, 32, 32), skip: (batch, 64, 64, 64)
        x, skip3 = self.enc3(x)
        skips.append(skip3)

        # Encoder Block 4
        # (batch, 64, 32, 32) -> (batch, 128, 16, 16), skip: (batch, 128, 32, 32)
        x, skip4 = self.enc4(x)
        skips.append(skip4)

        # At this point:
        # x shape: (batch, 128, 16, 16) - most compressed representation
        # skips: [skip1, skip2, skip3, skip4]

        # ==================================================================
        # BOTTLENECK: Process deepest representation
        # ==================================================================
        # (batch, 128, 16, 16) -> (batch, 256, 16, 16)
        # Spatial size unchanged - only channels increase

        x = self.bottleneck(x)

        # ==================================================================
        # DECODER PATH: Upsample and use skip connections
        # ==================================================================
        # Key insight: Decoders use skips in REVERSE order
        # - dec4 uses skip4 (last encoder skip = first decoder skip)
        # - dec1 uses skip1 (first encoder skip = last decoder skip)

        # Decoder Block 4 + skip4
        # (batch, 256, 16, 16) + (batch, 128, 32, 32) -> (batch, 128, 32, 32)
        x = self.dec4(x, skips[3])

        # Decoder Block 3 + skip3
        # (batch, 128, 32, 32) + (batch, 64, 64, 64) -> (batch, 64, 64, 64)
        x = self.dec3(x, skips[2])

        # Decoder Block 2 + skip2
        # (batch, 64, 64, 64) + (batch, 32, 128, 128) -> (batch, 32, 128, 128)
        x = self.dec2(x, skips[1])

        # Decoder Block 1 + skip1
        # (batch, 32, 128, 128) + (batch, 16, 256, 256) -> (batch, 16, 256, 256)
        x = self.dec1(x, skips[0])

        # ==================================================================
        # FINAL OUTPUT LAYER: Project to output channels
        # ==================================================================
        # (batch, 16, 256, 256) -> (batch, 2, 256, 256)

        x = self.final(x)

        # Verify output shape matches input spatial dimensions
        assert x.shape[2:] == input_shape[2:], \
            f"Output spatial dims {x.shape[2:]} don't match input {input_shape[2:]}"

        return x

    def count_parameters(self) -> int:
        """
        Count total trainable parameters in model.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """
        Estimate model size in megabytes.

        Returns:
            Approximate model size in MB (parameters + buffers)
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def test_unet():
    """Comprehensive test suite for U-Net implementation."""
    print("=" * 70)
    print("Testing U-Net Implementation")
    print("=" * 70)

    # Test 1: Standard forward pass
    print("\n1. Testing forward pass with standard input (256x256)")
    model = UNet(input_channels=2, output_channels=2, initial_filters=16)
    x = torch.randn(4, 2, 256, 256)
    output = model(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print("   [PASS] Forward pass successful")

    # Test 2: Different input sizes
    print("\n2. Testing with different input sizes")
    test_sizes = [(128, 128), (256, 256), (512, 512), (256, 512)]
    for h, w in test_sizes:
        x = torch.randn(2, 2, h, w)
        output = model(x)
        assert output.shape == x.shape, f"Size {(h, w)} failed"
        print(f"   {h}x{w}: [PASS]")

    # Test 3: Gradient flow
    print("\n3. Testing backward pass (gradient flow)")
    x = torch.randn(4, 2, 256, 256, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input"
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("   [PASS] All parameters receive gradients")

    # Test 4: No NaN or Inf
    print("\n4. Testing for NaN/Inf values")
    x = torch.randn(2, 2, 256, 256)
    output = model(x)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("   [PASS] No NaN or Inf values")

    # Test 5: Eval mode with batch_size=1
    print("\n5. Testing eval mode with batch_size=1")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 2, 256, 256)
        output = model(x)
    assert output.shape == x.shape
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print("   [PASS] Eval mode works with single sample")

    # Test 6: Model statistics
    print("\n6. Model statistics")
    num_params = model.count_parameters()
    model_size = model.get_model_size_mb()
    print(f"   Parameters: {num_params:,}")
    print(f"   Model size: {model_size:.2f} MB")

    # Test 7: Save and load
    print("\n7. Testing save/load")
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    try:
        torch.save(model.state_dict(), temp_path)
        model2 = UNet(input_channels=2, output_channels=2, initial_filters=16)
        model2.load_state_dict(torch.load(temp_path, weights_only=True))
        print(f"   Saved and loaded from {temp_path}")
        print("   [PASS] Save/load works")
    finally:
        os.unlink(temp_path)

    # Test 8: Mono input/output
    print("\n8. Testing mono input (1 channel)")
    model_mono = UNet(input_channels=1, output_channels=1, initial_filters=16)
    x = torch.randn(2, 1, 256, 256)
    output = model_mono(x)
    assert output.shape == x.shape
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print("   [PASS] Mono input works")

    print("\n" + "=" * 70)
    print("All U-Net tests passed!")
    print("=" * 70)

    # Print model architecture
    print("\nModel Architecture Summary:")
    print(f"  Input channels:    {model.input_channels}")
    print(f"  Output channels:   {model.output_channels}")
    print(f"  Initial filters:   {model.initial_filters}")
    print(f"  Depth:             {model.depth}")
    print(f"  Total parameters:  {model.count_parameters():,}")
    print(f"  Model size:        {model.get_model_size_mb():.2f} MB")


if __name__ == "__main__":
    test_unet()
