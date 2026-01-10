"""
Loss Functions for Audio Source Separation

Provides loss functions commonly used for spectrogram-based source separation:
- L1Loss: Mean absolute error (robust to outliers)
- MSELoss: Mean squared error (standard choice)
- CombinedLoss: Weighted combination of L1 and MSE

These operate on spectrogram tensors: (batch, channels, freq, time)

Usage:
    from utils import L1Loss, CombinedLoss

    criterion = L1Loss()
    loss = criterion(prediction, target)

Author: Q1 U-Net Implementation
"""

import torch
import torch.nn as nn
from typing import Optional


class L1Loss(nn.Module):
    """
    L1 Loss (Mean Absolute Error) for spectrogram comparison.

    Robust to outliers, commonly used for source separation.

    Args:
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'

    Shape:
        Input: (batch, channels, freq, time)
        Target: (batch, channels, freq, time)
        Output: scalar if reduction='mean' or 'sum'

    Example:
        >>> criterion = L1Loss()
        >>> prediction = torch.randn(4, 2, 256, 256)
        >>> target = torch.randn(4, 2, 256, 256)
        >>> loss = criterion(prediction, target)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 loss between prediction and target.

        Args:
            prediction: Model output (batch, channels, freq, time)
            target: Ground truth (batch, channels, freq, time)

        Returns:
            L1 loss value
        """
        return self.loss(prediction, target)


class MSELoss(nn.Module):
    """
    MSE Loss (Mean Squared Error) for spectrogram comparison.

    Penalizes large errors more heavily than L1.

    Args:
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'

    Shape:
        Input: (batch, channels, freq, time)
        Target: (batch, channels, freq, time)
        Output: scalar if reduction='mean' or 'sum'

    Example:
        >>> criterion = MSELoss()
        >>> prediction = torch.randn(4, 2, 256, 256)
        >>> target = torch.randn(4, 2, 256, 256)
        >>> loss = criterion(prediction, target)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss between prediction and target.

        Args:
            prediction: Model output (batch, channels, freq, time)
            target: Ground truth (batch, channels, freq, time)

        Returns:
            MSE loss value
        """
        return self.loss(prediction, target)


class CombinedLoss(nn.Module):
    """
    Combined L1 + MSE Loss for spectrogram comparison.

    Balances robustness of L1 with sensitivity of MSE:
    loss = alpha * L1(pred, target) + beta * MSE(pred, target)

    Args:
        alpha: Weight for L1 component. Default: 0.5
        beta: Weight for MSE component. Default: 0.5
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'

    Shape:
        Input: (batch, channels, freq, time)
        Target: (batch, channels, freq, time)
        Output: scalar if reduction='mean' or 'sum'

    Example:
        >>> criterion = CombinedLoss(alpha=0.7, beta=0.3)
        >>> loss = criterion(prediction, target)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss(reduction=reduction)
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined L1 + MSE loss.

        Args:
            prediction: Model output (batch, channels, freq, time)
            target: Ground truth (batch, channels, freq, time)

        Returns:
            Combined loss value: alpha * L1 + beta * MSE
        """
        l1_loss = self.l1(prediction, target)
        mse_loss = self.mse(prediction, target)
        return self.alpha * l1_loss + self.beta * mse_loss


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss for frequency domain comparison.

    Measures the normalized Frobenius norm of the difference:
    SC = ||S_pred - S_target||_F / ||S_target||_F

    Useful for maintaining spectral structure.

    Args:
        eps: Small value to avoid division by zero. Default: 1e-8

    Shape:
        Input: (batch, channels, freq, time)
        Target: (batch, channels, freq, time)
        Output: scalar

    Example:
        >>> criterion = SpectralConvergenceLoss()
        >>> loss = criterion(prediction, target)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral convergence loss.

        Args:
            prediction: Model output (batch, channels, freq, time)
            target: Ground truth (batch, channels, freq, time)

        Returns:
            Spectral convergence loss value
        """
        diff_norm = torch.norm(prediction - target, p='fro')
        target_norm = torch.norm(target, p='fro')
        return diff_norm / (target_norm + self.eps)


def get_loss_function(
    loss_type: str = 'l1',
    alpha: float = 0.5,
    beta: float = 0.5
) -> nn.Module:
    """
    Factory function to get loss function by name.

    Args:
        loss_type: 'l1', 'mse', 'combined', or 'spectral'
        alpha: L1 weight for combined loss
        beta: MSE weight for combined loss

    Returns:
        Loss function module

    Example:
        >>> criterion = get_loss_function('combined', alpha=0.7, beta=0.3)
    """
    if loss_type == 'l1':
        return L1Loss()
    elif loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'combined':
        return CombinedLoss(alpha=alpha, beta=beta)
    elif loss_type == 'spectral':
        return SpectralConvergenceLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_losses():
    """Test all loss functions."""
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    prediction = torch.randn(4, 2, 256, 256)
    target = torch.randn(4, 2, 256, 256)

    # Test L1
    print("\n1. L1 Loss")
    l1 = L1Loss()
    loss = l1(prediction, target)
    print(f"   Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "L1 loss is NaN"
    print("   [PASS]")

    # Test MSE
    print("\n2. MSE Loss")
    mse = MSELoss()
    loss = mse(prediction, target)
    print(f"   Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "MSE loss is NaN"
    print("   [PASS]")

    # Test Combined
    print("\n3. Combined Loss")
    combined = CombinedLoss(alpha=0.7, beta=0.3)
    loss = combined(prediction, target)
    print(f"   Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Combined loss is NaN"
    print("   [PASS]")

    # Test Spectral Convergence
    print("\n4. Spectral Convergence Loss")
    spectral = SpectralConvergenceLoss()
    loss = spectral(prediction, target)
    print(f"   Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Spectral loss is NaN"
    print("   [PASS]")

    # Test factory
    print("\n5. Loss Factory")
    for loss_type in ['l1', 'mse', 'combined', 'spectral']:
        criterion = get_loss_function(loss_type)
        loss = criterion(prediction, target)
        print(f"   {loss_type}: {loss.item():.6f}")
    print("   [PASS]")

    # Test gradient flow
    print("\n6. Gradient Flow")
    prediction.requires_grad = True
    criterion = L1Loss()
    loss = criterion(prediction, target)
    loss.backward()
    assert prediction.grad is not None, "No gradient"
    print("   [PASS]")

    print("\n" + "=" * 60)
    print("All loss function tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()
