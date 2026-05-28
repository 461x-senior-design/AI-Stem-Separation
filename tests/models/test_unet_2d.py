import pytest
import torch
import torch.nn as nn

from stemmy.models.unet_2d import ConvBlock, UNet2D, _upsample_to, pad_to_multiple, unpad


def test_conv_block_init():
    # Valid
    block = ConvBlock(2, 64)
    assert isinstance(block.conv1, nn.Conv2d)
    assert block.conv1.in_channels == 2
    assert block.conv1.out_channels == 64
    assert isinstance(block.bn1, nn.BatchNorm2d)
    assert block.bn1.num_features == 64
    assert isinstance(block.conv2, nn.Conv2d)
    assert block.conv2.in_channels == 64
    assert block.conv2.out_channels == 64
    assert isinstance(block.bn2, nn.BatchNorm2d)
    assert block.bn2.num_features == 64
    assert isinstance(block.act, nn.LeakyReLU)
    assert block.act.negative_slope == 0.1
    
    # Invalid in_ch
    with pytest.raises(ValueError, match="in_ch must be positive"):
        ConvBlock(0, 64)
    with pytest.raises(ValueError, match="in_ch must be positive"):
        ConvBlock(-1, 64)
        
    # Invalid out_ch
    with pytest.raises(ValueError, match="out_ch must be positive"):
        ConvBlock(2, 0)
    with pytest.raises(ValueError, match="out_ch must be positive"):
        ConvBlock(2, -1)

def test_conv_block_forward():
    in_ch = 2
    out_ch = 64
    block = ConvBlock(in_ch, out_ch)
    x = torch.randn(1, in_ch, 32, 32)
    out = block(x)
    assert out.shape == (1, out_ch, 32, 32)
    
    # Test with different spatial dimensions
    x2 = torch.randn(2, in_ch, 15, 20)
    out2 = block(x2)
    assert out2.shape == (2, out_ch, 15, 20)

def test_pad_to_multiple():
    x = torch.randn(1, 2, 10, 10)
    # Pad to 16
    x_pad, pad = pad_to_multiple(x, 16, 16)
    assert x_pad.shape == (1, 2, 16, 16)
    assert pad == (0, 6, 0, 6)
    
    # No pad
    x2 = torch.randn(1, 2, 16, 16)
    x_pad2, pad2 = pad_to_multiple(x2, 16, 16)
    assert x_pad2.shape == (1, 2, 16, 16)
    assert pad2 == (0, 0, 0, 0)
    
    # Errors
    with pytest.raises(ValueError, match="Expected x to have shape"):
        pad_to_multiple(torch.randn(10, 10), 16, 16)
    with pytest.raises(ValueError, match="multiple_h must be > 0"):
        pad_to_multiple(x, 0, 16)
    with pytest.raises(ValueError, match="multiple_w must be > 0"):
        pad_to_multiple(x, 16, -1)

def test_unpad():
    x = torch.randn(1, 2, 16, 16)
    pad = (0, 6, 0, 6)
    x_unpad = unpad(x, pad)
    assert x_unpad.shape == (1, 2, 10, 10)
    
    # No unpad
    x_unpad2 = unpad(x, (0, 0, 0, 0))
    assert x_unpad2 is x
    
    # Errors
    with pytest.raises(ValueError, match="Expected x to have shape"):
        unpad(torch.randn(10, 10), (0,0,0,0))
    with pytest.raises(ValueError, match="pad values must be >= 0"):
        unpad(x, (0, -1, 0, 0))
    with pytest.raises(ValueError, match="Invalid unpad resulting shape"):
        unpad(x, (0, 20, 0, 0))

def test_upsample_to():
    x = torch.randn(1, 64, 8, 8)
    ref = torch.randn(1, 32, 16, 16)
    out = _upsample_to(x, ref)
    assert out.shape == (1, 64, 16, 16)
    
    with pytest.raises(ValueError, match="Expected 4D tensors"):
        _upsample_to(x[0], ref)

def test_unet2d_init():
    model = UNet2D(stems=4)
    assert model.stems == 4
    
    with pytest.raises(ValueError, match="stems must be positive"):
        UNet2D(stems=0)
    with pytest.raises(ValueError, match="base_channels must be positive"):
        UNet2D(base_channels=-1)
    with pytest.raises(ValueError, match="audio_channels must be positive"):
        UNet2D(audio_channels=0)

def test_unet2d_forward():
    model = UNet2D(stems=2, base_channels=8, audio_channels=2)
    model.eval()
    
    # Needs padding (10x10 -> 16x16)
    x = torch.randn(1, 2, 10, 10)
    out = model(x)
    assert out.shape == (1, 2, 2, 10, 10) # [B, S, C, F, T]
    
    # Already multiple of 16
    x2 = torch.randn(1, 2, 16, 16)
    out2 = model(x2)
    assert out2.shape == (1, 2, 2, 16, 16)

def test_unet2d_forward_errors():
    model = UNet2D(audio_channels=2)
    with pytest.raises(ValueError, match="Expected input x to be 4D"):
        model(torch.randn(2, 16, 16))
    with pytest.raises(ValueError, match="Expected channel dimension to be 2"):
        model(torch.randn(1, 1, 16, 16))

def test_unet2d_forward_runtime_error():
    model = UNet2D(stems=2, audio_channels=2)
    model.eval()
    x = torch.randn(1, 2, 16, 16)
    # Manually corrupt stems to trigger RuntimeError in forward
    model.stems = 3
    with pytest.raises(RuntimeError, match="Unexpected output channels"):
        model(x)
