#!/usr/bin/env python3
"""
U-Net Validation Runner

Main entry point for validating the Q1 U-Net implementation.
Runs all tests to verify the model is ready for Q2 training.

Usage:
    python run_validation.py           # Run all tests
    python run_validation.py --quick   # Quick validation only
    python run_validation.py --verbose # Extra output

Author: Q1 U-Net Implementation
"""

import sys
import argparse
import torch

from models import UNet
from tests.run_all_tests import run_all_tests


def quick_validation():
    """
    Quick validation - just test forward/backward pass.

    Returns:
        True if validation passes
    """
    print("=" * 60)
    print("Quick Validation")
    print("=" * 60)

    try:
        # Create model
        print("\n1. Creating U-Net model...")
        model = UNet(input_channels=2, output_channels=2, initial_filters=16)
        print(f"   Parameters: {model.count_parameters():,}")

        # Forward pass
        print("\n2. Testing forward pass...")
        x = torch.randn(2, 2, 256, 256)
        output = model(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape}"
        print(f"   Input:  {x.shape}")
        print(f"   Output: {output.shape}")
        print("   [PASS]")

        # Backward pass
        print("\n3. Testing backward pass...")
        loss = output.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient: {name}"
        print("   All gradients computed")
        print("   [PASS]")

        # No NaN/Inf
        print("\n4. Checking for NaN/Inf...")
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print("   No NaN or Inf values")
        print("   [PASS]")

        print("\n" + "=" * 60)
        print("Quick validation PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="U-Net Validation for Quarter 1"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick validation only'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Overfitting test iterations (default: 100)'
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  U-Net Quarter 1 Validation")
    print("  Audio Source Separation Architecture")
    print("=" * 70)

    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.quick:
        success = quick_validation()
    else:
        success = run_all_tests()

    if success:
        print("\n" + "-" * 70)
        print("  VALIDATION COMPLETE - Ready for Quarter 2!")
        print("-" * 70)
        print("\nNext steps:")
        print("  1. Acquire training data (MUSDB18 or similar)")
        print("  2. Set up training pipeline")
        print("  3. Train model on real audio")
        print()
    else:
        print("\n" + "-" * 70)
        print("  VALIDATION FAILED - Fix issues before Q2")
        print("-" * 70)
        print("\nReview the test output above for failures.")
        print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
