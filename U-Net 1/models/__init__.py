"""
U-Net Models for Audio Source Separation

This package contains the complete U-Net architecture for audio source separation:
- EncoderBlock: Downsampling path with feature extraction
- DecoderBlock: Upsampling path with skip connection integration
- UNet: Complete encoder-decoder architecture with skip connections

Quarter 1 Deliverable - Architecture only (untrained weights)

Usage:
    from models import UNet, EncoderBlock, DecoderBlock

    model = UNet(input_channels=2, output_channels=2)
    x = torch.randn(4, 2, 256, 256)  # Batch of stereo spectrograms
    output = model(x)  # Same shape as input
"""

from .encoder import EncoderBlock
from .decoder import DecoderBlock
from .unet import UNet

__all__ = ['EncoderBlock', 'DecoderBlock', 'UNet']
