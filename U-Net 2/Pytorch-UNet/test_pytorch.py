#!/usr/bin/env python3
"""
PyTorch Installation Test
=========================

Quick test to verify PyTorch is installed and working.
Run this before training the U-Net model.
"""

import sys

print("="*60)
print("PyTorch Installation Test")
print("="*60)

# Test 1: Import PyTorch
try:
    import torch
    print("✓ PyTorch imported successfully")
    print(f"  Version: {torch.__version__}")
except ImportError as e:
    print("✗ Failed to import PyTorch")
    print(f"  Error: {e}")
    print("\nInstall PyTorch with:")
    print("  uv pip install torch torchvision")
    sys.exit(1)

# Test 2: Check hardware acceleration
print("\n" + "="*60)
print("Hardware Acceleration")
print("="*60)

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()

print(f"CUDA (NVIDIA GPU): {'✓ Available' if cuda_available else '✗ Not available'}")
print(f"MPS (Apple Silicon): {'✓ Available' if mps_available else '✗ Not available'}")

if cuda_available:
    print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
elif mps_available:
    print("  Running on Apple Silicon GPU")
else:
    print("  Running on CPU (training will be slower)")

# Test 3: Basic tensor operations
print("\n" + "="*60)
print("Tensor Operations Test")
print("="*60)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
result = x + y

print(f"x = {x.tolist()}")
print(f"y = {y.tolist()}")
print(f"x + y = {result.tolist()}")
print(f"Expected: [5.0, 7.0, 9.0]")

if result.tolist() == [5.0, 7.0, 9.0]:
    print("✓ Tensor math works correctly")
else:
    print("✗ Unexpected result - something is wrong")
    sys.exit(1)

# Test 4: Matrix multiplication
print("\n" + "="*60)
print("Matrix Operations Test")
print("="*60)

A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[2.0, 0.0],
                  [1.0, 3.0]])
C = A @ B  # Matrix multiplication

print(f"A @ B =")
print(C)
print(f"Expected:")
print("  [[4.0, 6.0],")
print("   [10.0, 12.0]]")

expected = torch.tensor([[4.0, 6.0], [10.0, 12.0]])
if torch.allclose(C, expected):
    print("✓ Matrix multiplication works correctly")
else:
    print("✗ Unexpected result")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print("✓ All tests passed!")
print("✓ PyTorch is ready to use")
print("\nNext step: Train the U-Net model")
print("  python3 train.py --epochs 10 --batch-size 2 --validation 20")
print("="*60)
