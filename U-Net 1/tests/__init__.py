"""
Test Suite for U-Net Audio Source Separation

Comprehensive tests for validating U-Net architecture:
- Shape validation (forward pass shapes)
- Gradient flow (backward pass works)
- Overfitting (model can learn)
- Memory profiling (reasonable resource usage)
- Audio quality (no NaN/Inf, reasonable values)

Usage:
    # Run all tests
    python -m tests.run_all_tests

    # Run specific test module
    python -m tests.test_shapes
"""

from .test_shapes import test_shapes
from .test_gradients import test_gradient_flow
from .test_overfitting import test_overfitting

__all__ = ['test_shapes', 'test_gradient_flow', 'test_overfitting']
