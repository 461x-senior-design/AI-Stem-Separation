"""
Shape Validation Tests for U-Net

Verifies that tensor shapes are correct throughout the U-Net:
- Input/output shapes match
- Encoder shapes halve spatial dimensions
- Decoder shapes double spatial dimensions
- Skip connections have matching shapes

Usage:
    python -m tests.test_shapes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import UNet, EncoderBlock, DecoderBlock


def test_shapes() -> bool:
    """
    Test all tensor shapes throughout U-Net.

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("Shape Validation Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Standard input size
    print("\n1. Standard input size (256x256)")
    try:
        model = UNet(input_channels=2, output_channels=2)
        x = torch.randn(4, 2, 256, 256)
        output = model(x)

        if output.shape == x.shape:
            print(f"   Input:  {x.shape}")
            print(f"   Output: {output.shape}")
            print("   [PASS]")
        else:
            print(f"   [FAIL] Expected {x.shape}, got {output.shape}")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 2: Different input sizes
    print("\n2. Various input sizes")
    test_sizes = [
        (128, 128),
        (256, 256),
        (512, 512),
        (256, 512),  # Non-square
        (512, 256),  # Non-square (other direction)
    ]

    model = UNet(input_channels=2, output_channels=2)
    for h, w in test_sizes:
        try:
            x = torch.randn(2, 2, h, w)
            output = model(x)
            if output.shape == x.shape:
                print(f"   {h}x{w}: [PASS]")
            else:
                print(f"   {h}x{w}: [FAIL] Output {output.shape}")
                all_passed = False
        except Exception as e:
            print(f"   {h}x{w}: [FAIL] {e}")
            all_passed = False

    # Test 3: Different batch sizes
    print("\n3. Various batch sizes")
    for batch_size in [1, 2, 4, 8, 16]:
        try:
            x = torch.randn(batch_size, 2, 256, 256)
            output = model(x)
            if output.shape[0] == batch_size:
                print(f"   Batch {batch_size}: [PASS]")
            else:
                print(f"   Batch {batch_size}: [FAIL]")
                all_passed = False
        except Exception as e:
            print(f"   Batch {batch_size}: [FAIL] {e}")
            all_passed = False

    # Test 4: Mono input/output
    print("\n4. Mono input (1 channel)")
    try:
        model_mono = UNet(input_channels=1, output_channels=1)
        x = torch.randn(2, 1, 256, 256)
        output = model_mono(x)
        if output.shape == x.shape:
            print(f"   Input:  {x.shape}")
            print(f"   Output: {output.shape}")
            print("   [PASS]")
        else:
            print(f"   [FAIL]")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 5: Encoder block shapes
    print("\n5. EncoderBlock shapes")
    try:
        encoder = EncoderBlock(in_channels=2, out_channels=16)
        x = torch.randn(1, 2, 256, 256)
        encoded, skip = encoder(x)

        expected_encoded = (1, 16, 128, 128)
        expected_skip = (1, 16, 256, 256)

        if encoded.shape == expected_encoded and skip.shape == expected_skip:
            print(f"   Input:   {x.shape}")
            print(f"   Encoded: {encoded.shape} (expected {expected_encoded})")
            print(f"   Skip:    {skip.shape} (expected {expected_skip})")
            print("   [PASS]")
        else:
            print(f"   [FAIL] Shape mismatch")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 6: Decoder block shapes
    print("\n6. DecoderBlock shapes")
    try:
        decoder = DecoderBlock(in_channels=32, out_channels=16)
        x = torch.randn(1, 32, 128, 128)
        skip = torch.randn(1, 16, 256, 256)
        output = decoder(x, skip)

        expected_output = (1, 16, 256, 256)
        if output.shape == expected_output:
            print(f"   Input:  {x.shape}")
            print(f"   Skip:   {skip.shape}")
            print(f"   Output: {output.shape} (expected {expected_output})")
            print("   [PASS]")
        else:
            print(f"   [FAIL] Shape mismatch")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All shape validation tests PASSED!")
    else:
        print("Some shape validation tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_shapes()
    sys.exit(0 if success else 1)
