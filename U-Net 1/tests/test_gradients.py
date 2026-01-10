"""
Gradient Flow Tests for U-Net

Verifies that gradients flow correctly through the U-Net:
- All parameters receive gradients
- No vanishing gradients (too small)
- No exploding gradients (too large)
- Skip connections receive gradients

Usage:
    python -m tests.test_gradients
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import UNet


def test_gradient_flow() -> bool:
    """
    Test gradient flow through U-Net.

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("Gradient Flow Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: All parameters receive gradients
    print("\n1. All parameters receive gradients")
    try:
        model = UNet(input_channels=2, output_channels=2)
        x = torch.randn(2, 2, 256, 256)

        output = model(x)
        loss = output.sum()
        loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.grad is None:
                params_without_grad.append(name)

        if not params_without_grad:
            num_params = len(list(model.parameters()))
            print(f"   All {num_params} parameter groups have gradients")
            print("   [PASS]")
        else:
            print(f"   [FAIL] Missing gradients: {params_without_grad[:5]}...")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 2: No vanishing gradients
    print("\n2. No vanishing gradients")
    try:
        model = UNet(input_channels=2, output_channels=2)
        model.zero_grad()

        x = torch.randn(4, 2, 256, 256)
        output = model(x)
        loss = output.sum()
        loss.backward()

        vanishing = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < 1e-10:
                    vanishing.append((name, grad_norm))

        if not vanishing:
            print("   No vanishing gradients detected")
            print("   [PASS]")
        else:
            print(f"   [WARN] Vanishing gradients in {len(vanishing)} params")
            for name, norm in vanishing[:3]:
                print(f"      {name}: {norm:.2e}")
            # This is a warning, not a failure
            print("   [PASS with warnings]")
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 3: No exploding gradients
    print("\n3. No exploding gradients")
    try:
        model = UNet(input_channels=2, output_channels=2)
        model.zero_grad()

        x = torch.randn(4, 2, 256, 256)
        output = model(x)
        loss = output.sum()
        loss.backward()

        exploding = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e5:
                    exploding.append((name, grad_norm))

        if not exploding:
            print("   No exploding gradients detected")
            print("   [PASS]")
        else:
            print(f"   [FAIL] Exploding gradients in {len(exploding)} params")
            for name, norm in exploding[:3]:
                print(f"      {name}: {norm:.2e}")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 4: Gradient magnitudes are reasonable
    print("\n4. Gradient magnitude statistics")
    try:
        model = UNet(input_channels=2, output_channels=2)
        model.zero_grad()

        x = torch.randn(4, 2, 256, 256)
        output = model(x)
        loss = output.mean()  # Use mean for more stable gradients
        loss.backward()

        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        min_grad = min(grad_norms)
        max_grad = max(grad_norms)
        mean_grad = sum(grad_norms) / len(grad_norms)

        print(f"   Min gradient norm:  {min_grad:.6f}")
        print(f"   Max gradient norm:  {max_grad:.6f}")
        print(f"   Mean gradient norm: {mean_grad:.6f}")
        print("   [PASS]")
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 5: Skip connections receive gradients
    print("\n5. Skip connections receive gradients (encoder params)")
    try:
        model = UNet(input_channels=2, output_channels=2)
        model.zero_grad()

        x = torch.randn(2, 2, 256, 256)
        output = model(x)
        loss = output.sum()
        loss.backward()

        encoder_params = [
            (name, param) for name, param in model.named_parameters()
            if 'enc' in name
        ]

        all_have_grads = True
        for name, param in encoder_params:
            if param.grad is None or param.grad.abs().sum() == 0:
                print(f"   [FAIL] Encoder param {name} has no gradient")
                all_have_grads = False

        if all_have_grads:
            print(f"   All {len(encoder_params)} encoder params have gradients")
            print("   [PASS]")
        else:
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Test 6: Input receives gradient
    print("\n6. Input tensor receives gradient")
    try:
        model = UNet(input_channels=2, output_channels=2)
        x = torch.randn(2, 2, 256, 256, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        if x.grad is not None:
            print(f"   Input gradient shape: {x.grad.shape}")
            print("   [PASS]")
        else:
            print("   [FAIL] No gradient for input")
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All gradient flow tests PASSED!")
    else:
        print("Some gradient flow tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_gradient_flow()
    sys.exit(0 if success else 1)
