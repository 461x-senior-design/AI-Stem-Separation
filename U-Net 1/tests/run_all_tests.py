"""
Comprehensive U-Net Validation Test Suite

Runs all validation tests for U-Net:
1. Shape validation - all tensor shapes correct
2. Gradient flow - backward pass works
3. Overfitting test - model can learn
4. Memory profiling - reasonable resource usage
5. Audio quality - no NaN/Inf, reasonable outputs
6. Save/load - model persistence works

Usage:
    python -m tests.run_all_tests
    python tests/run_all_tests.py

Author: Q1 U-Net Implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from pathlib import Path
from typing import List, Tuple

from models import UNet


class TestResult:
    """Store and format test results."""

    def __init__(self):
        self.passed: List[Tuple[str, str]] = []
        self.failed: List[Tuple[str, str]] = []
        self.warnings: List[str] = []

    def add_pass(self, test_name: str, details: str = ""):
        self.passed.append((test_name, details))

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))

    def add_warning(self, message: str):
        self.warnings.append(message)


def test_shape_validation(model: nn.Module, results: TestResult):
    """Test tensor shapes throughout model."""
    print("\n" + "=" * 60)
    print("Test 1: Shape Validation")
    print("=" * 60)

    try:
        test_cases = [
            (4, 2, 256, 256),
            (1, 2, 128, 128),
            (2, 2, 512, 512),
        ]

        all_passed = True
        for batch, channels, h, w in test_cases:
            x = torch.randn(batch, channels, h, w)
            output = model(x)

            if output.shape != x.shape:
                results.add_fail(
                    f"Shape ({batch}, {channels}, {h}, {w})",
                    f"Expected {x.shape}, got {output.shape}"
                )
                all_passed = False
            else:
                print(f"   {h}x{w}: [PASS]")

        if all_passed:
            results.add_pass("Shape validation", "All input sizes work correctly")
    except Exception as e:
        results.add_fail("Shape validation", str(e))


def test_gradient_flow(model: nn.Module, results: TestResult):
    """Test gradient flow through model."""
    print("\n" + "=" * 60)
    print("Test 2: Gradient Flow")
    print("=" * 60)

    try:
        model.zero_grad()
        x = torch.randn(2, 2, 256, 256)
        output = model(x)
        # Use mean() for stable gradient magnitudes (sum() creates artificially large gradients)
        loss = output.mean()
        loss.backward()

        params_without_grad = []
        vanishing = []
        exploding = []

        for name, param in model.named_parameters():
            if param.grad is None:
                params_without_grad.append(name)
            else:
                grad_norm = param.grad.norm().item()
                if grad_norm < 1e-10:
                    vanishing.append(name)
                elif grad_norm > 1e5:
                    exploding.append(name)

        if params_without_grad:
            results.add_fail(
                "Gradient flow",
                f"{len(params_without_grad)} params missing gradients"
            )
            return

        if vanishing:
            results.add_warning(f"Vanishing gradients in {len(vanishing)} params")

        if exploding:
            results.add_fail(
                "Gradient flow",
                f"Exploding gradients in {len(exploding)} params"
            )
            return

        total_params = sum(p.numel() for p in model.parameters())
        results.add_pass(
            "Gradient flow",
            f"All params receive gradients ({total_params:,} total)"
        )
        print("   All parameters receive gradients: [PASS]")

    except Exception as e:
        results.add_fail("Gradient flow", str(e))


def test_overfitting(model: nn.Module, results: TestResult, iterations: int = 100):
    """Test model can overfit to single example."""
    print("\n" + "=" * 60)
    print("Test 3: Overfitting (Can Model Learn?)")
    print("=" * 60)

    try:
        # Fresh model for clean test
        test_model = UNet(
            input_channels=model.input_channels,
            output_channels=model.output_channels,
            initial_filters=model.initial_filters
        )

        x = torch.randn(1, 2, 256, 256)
        target = torch.randn(1, 2, 256, 256)

        optimizer = optim.Adam(test_model.parameters(), lr=1e-3)
        criterion = nn.L1Loss()

        losses = []
        print(f"   Training for {iterations} iterations...")

        for i in range(iterations):
            optimizer.zero_grad()
            output = test_model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i % 25 == 0:
                print(f"   Iter {i:3d}: Loss = {loss.item():.6f}")

        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        print(f"\n   Initial: {initial_loss:.6f} -> Final: {final_loss:.6f}")
        print(f"   Reduction: {reduction * 100:.1f}%")

        if reduction >= 0.5:
            results.add_pass(
                "Overfitting test",
                f"Loss reduced {reduction * 100:.1f}%"
            )
            print("   [PASS]")
        else:
            results.add_fail(
                "Overfitting test",
                f"Only {reduction * 100:.1f}% reduction (need >50%)"
            )
            print("   [FAIL]")

    except Exception as e:
        results.add_fail("Overfitting test", str(e))


def test_audio_quality(model: nn.Module, results: TestResult):
    """Test output audio quality (no NaN/Inf, reasonable values)."""
    print("\n" + "=" * 60)
    print("Test 4: Audio Quality Checks")
    print("=" * 60)

    try:
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 2, 256, 256)
            output = model(x)

            # Check for NaN
            if torch.isnan(output).any():
                results.add_fail("Audio quality", "Output contains NaN")
                return

            # Check for Inf
            if torch.isinf(output).any():
                results.add_fail("Audio quality", "Output contains Inf")
                return

            # Check not all zeros
            if torch.all(output == 0):
                results.add_fail("Audio quality", "Output is all zeros")
                return

            # Statistics
            mean_val = output.mean().item()
            std_val = output.std().item()
            min_val = output.min().item()
            max_val = output.max().item()

            print(f"   Mean: {mean_val:.4f}")
            print(f"   Std:  {std_val:.4f}")
            print(f"   Range: [{min_val:.4f}, {max_val:.4f}]")

            if abs(mean_val) > 10:
                results.add_warning(f"Large output mean: {mean_val:.4f}")

            if std_val < 0.001:
                results.add_warning(f"Very low variance: {std_val:.6f}")

            results.add_pass(
                "Audio quality",
                f"Mean={mean_val:.4f}, Std={std_val:.4f}"
            )
            print("   [PASS]")

    except Exception as e:
        results.add_fail("Audio quality", str(e))


def test_save_load(model: nn.Module, results: TestResult):
    """Test model save and load."""
    print("\n" + "=" * 60)
    print("Test 5: Save/Load")
    print("=" * 60)

    try:
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        # Save
        torch.save(model.state_dict(), temp_path)
        print(f"   Saved to: {temp_path}")

        # Load into new model
        loaded_model = UNet(
            input_channels=model.input_channels,
            output_channels=model.output_channels,
            initial_filters=model.initial_filters
        )
        loaded_model.load_state_dict(torch.load(temp_path, weights_only=True))
        print("   Loaded successfully")

        # Verify same output
        x = torch.randn(1, 2, 128, 128)
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            output1 = model(x)
            output2 = loaded_model(x)

        if torch.allclose(output1, output2, atol=1e-6):
            results.add_pass("Save/load", "Model saves and loads correctly")
            print("   Outputs match: [PASS]")
        else:
            results.add_fail("Save/load", "Loaded model produces different output")
            print("   [FAIL]")

        # Cleanup
        Path(temp_path).unlink()

    except Exception as e:
        results.add_fail("Save/load", str(e))


def test_memory_and_params(model: nn.Module, results: TestResult):
    """Test memory usage and parameter count."""
    print("\n" + "=" * 60)
    print("Test 6: Memory and Parameters")
    print("=" * 60)

    try:
        num_params = sum(p.numel() for p in model.parameters())
        model_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 ** 2)

        print(f"   Parameters: {num_params:,}")
        print(f"   Model size: {model_size:.2f} MB")

        # Reasonable bounds for depth=4, initial_filters=16
        if 100_000 < num_params < 10_000_000:
            results.add_pass(
                "Parameter count",
                f"{num_params:,} parameters ({model_size:.2f} MB)"
            )
            print("   [PASS]")
        else:
            results.add_warning(f"Unusual parameter count: {num_params:,}")
            print("   [WARN]")

    except Exception as e:
        results.add_fail("Memory/params", str(e))


def run_all_tests() -> bool:
    """
    Run complete validation test suite.

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 70)
    print("  U-Net Validation Test Suite")
    print("  Quarter 1 Checkpoint")
    print("=" * 70)

    # Create model
    model = UNet(input_channels=2, output_channels=2, initial_filters=16)
    print(f"\nModel: UNet(2->2, filters=16, depth=4)")
    print(f"Parameters: {model.count_parameters():,}")

    results = TestResult()

    # Run all tests
    test_shape_validation(model, results)
    test_gradient_flow(model, results)
    test_overfitting(model, results)
    test_audio_quality(model, results)
    test_save_load(model, results)
    test_memory_and_params(model, results)

    # Print summary
    print("\n" + "=" * 70)
    print("  TEST RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n[PASSED] {len(results.passed)} tests")
    for name, details in results.passed:
        print(f"   [OK] {name}")
        if details:
            print(f"        {details}")

    if results.warnings:
        print(f"\n[WARNINGS] {len(results.warnings)}")
        for warning in results.warnings:
            print(f"   [!] {warning}")

    if results.failed:
        print(f"\n[FAILED] {len(results.failed)} tests")
        for name, error in results.failed:
            print(f"   [X] {name}")
            print(f"       Error: {error}")

    print("\n" + "=" * 70)
    all_passed = len(results.failed) == 0

    if all_passed:
        print("  ALL TESTS PASSED!")
        print("  Your U-Net is ready for Quarter 2 training!")
    else:
        print("  SOME TESTS FAILED")
        print("  Review errors above before proceeding")

    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
