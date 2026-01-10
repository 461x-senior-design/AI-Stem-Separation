"""
Overfitting Tests for U-Net

Verifies that U-Net can learn by overfitting to a single example.
If the model can't overfit, it can't learn - this is a critical validation.

Usage:
    python -m tests.test_overfitting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from models import UNet


def test_overfitting(iterations: int = 100, verbose: bool = True) -> bool:
    """
    Test that U-Net can overfit to a single example.

    This is a critical validation - if the model can't memorize one example,
    it won't be able to learn from a dataset.

    Args:
        iterations: Number of training iterations. Default: 100
        verbose: Print progress. Default: True

    Returns:
        True if overfitting succeeds (loss reduces by >50%), False otherwise
    """
    if verbose:
        print("=" * 60)
        print("Overfitting Test")
        print("=" * 60)

    # Create model
    model = UNet(input_channels=2, output_channels=2, initial_filters=16)

    # Single training example (random but fixed)
    torch.manual_seed(42)  # Reproducible
    x = torch.randn(1, 2, 256, 256)
    target = torch.randn(1, 2, 256, 256)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    losses = []

    if verbose:
        print(f"\nTraining for {iterations} iterations on single example...")
        print(f"Input shape:  {x.shape}")
        print(f"Target shape: {target.shape}")
        print()

    # Training loop
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and i % 20 == 0:
            print(f"   Iteration {i:3d}: Loss = {loss.item():.6f}")

    # Analyze results
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss

    if verbose:
        print(f"\nResults:")
        print(f"   Initial loss: {initial_loss:.6f}")
        print(f"   Final loss:   {final_loss:.6f}")
        print(f"   Reduction:    {reduction * 100:.1f}%")

    # Success criteria: loss should reduce by at least 50%
    success = reduction >= 0.5

    if verbose:
        print()
        if success:
            print("   [PASS] Model successfully overfits!")
            print("   The model can learn from data.")
        else:
            print("   [FAIL] Model did not overfit sufficiently")
            print("   The model may have issues learning.")

        print("\n" + "=" * 60)
        if success:
            print("Overfitting test PASSED!")
        else:
            print("Overfitting test FAILED!")
        print("=" * 60)

    return success


def test_overfitting_with_synthetic_spectrogram() -> bool:
    """
    Test overfitting with a more realistic spectrogram-like input.

    Returns:
        True if test passes
    """
    print("=" * 60)
    print("Overfitting Test with Synthetic Spectrogram")
    print("=" * 60)

    model = UNet(input_channels=2, output_channels=2, initial_filters=16)

    # Create spectrogram-like input (non-negative, typical spectrogram range)
    torch.manual_seed(42)
    mixture = torch.rand(1, 2, 256, 256) * 10  # 0-10 range like log spectrograms
    target = torch.rand(1, 2, 256, 256) * 10

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    losses = []
    print(f"\nTraining for 100 iterations...")

    for i in range(100):
        optimizer.zero_grad()
        output = model(mixture)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 25 == 0:
            print(f"   Iteration {i:3d}: Loss = {loss.item():.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss

    print(f"\nResults:")
    print(f"   Initial loss: {initial_loss:.6f}")
    print(f"   Final loss:   {final_loss:.6f}")
    print(f"   Reduction:    {reduction * 100:.1f}%")

    success = reduction >= 0.5

    print()
    if success:
        print("   [PASS]")
    else:
        print("   [FAIL]")

    print("\n" + "=" * 60)
    return success


def test_learning_rate_sensitivity() -> bool:
    """
    Test that model learns with different learning rates.

    Returns:
        True if at least one learning rate works
    """
    print("=" * 60)
    print("Learning Rate Sensitivity Test")
    print("=" * 60)

    learning_rates = [1e-4, 1e-3, 1e-2]
    results = {}

    for lr in learning_rates:
        print(f"\nTesting lr={lr}...")

        model = UNet(input_channels=2, output_channels=2, initial_filters=16)
        torch.manual_seed(42)
        x = torch.randn(1, 2, 128, 128)
        target = torch.randn(1, 2, 128, 128)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        initial_loss = None
        final_loss = None

        for i in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

        reduction = (initial_loss - final_loss) / initial_loss
        results[lr] = reduction
        print(f"   Reduction: {reduction * 100:.1f}%")

    # At least one LR should work
    best_lr = max(results, key=results.get)
    best_reduction = results[best_lr]

    print(f"\nBest learning rate: {best_lr} ({best_reduction * 100:.1f}% reduction)")

    success = best_reduction >= 0.3  # More lenient for quick test

    print("\n" + "=" * 60)
    if success:
        print("Learning rate sensitivity test PASSED!")
    else:
        print("Learning rate sensitivity test FAILED!")
    print("=" * 60)

    return success


if __name__ == "__main__":
    # Run primary overfitting test
    success1 = test_overfitting(iterations=100, verbose=True)

    print("\n" * 2)

    # Run spectrogram test
    success2 = test_overfitting_with_synthetic_spectrogram()

    print("\n" * 2)

    # Run learning rate test
    success3 = test_learning_rate_sensitivity()

    # Overall result
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Basic overfitting:    {'PASS' if success1 else 'FAIL'}")
    print(f"Spectrogram test:     {'PASS' if success2 else 'FAIL'}")
    print(f"Learning rate test:   {'PASS' if success3 else 'FAIL'}")

    all_passed = success1 and success2 and success3
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED!"))

    sys.exit(0 if all_passed else 1)
