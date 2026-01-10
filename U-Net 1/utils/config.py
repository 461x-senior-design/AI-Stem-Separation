"""
Configuration for U-Net Audio Source Separation

Centralized configuration for model architecture, training, and audio parameters.
Designed for Quarter 1 validation and Quarter 2 training.

Usage:
    from utils import get_config, get_device

    config = get_config()
    print(config['model']['initial_filters'])  # 16

    device = get_device()  # 'cuda' if available, else 'cpu'
"""

import torch
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Get default configuration for U-Net training.

    Returns:
        Dictionary with configuration parameters organized by category
    """
    config = {
        # Model architecture
        'model': {
            'input_channels': 2,      # Stereo audio
            'output_channels': 2,     # Stereo output (separated source)
            'initial_filters': 16,    # First encoder filter count
            'depth': 4,               # Number of encoder/decoder levels
            'use_dropout': False,     # Dropout in decoder blocks
            'dropout_prob': 0.5,      # Dropout probability if enabled
        },

        # Audio preprocessing
        'audio': {
            'sample_rate': 44100,     # Audio sample rate (CD quality)
            'n_fft': 2048,            # FFT window size
            'hop_length': 512,        # STFT hop length
            'normalize': 'log',       # Normalization: 'log', 'minmax', 'standardize'
        },

        # Training hyperparameters (for Q2)
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 100,
            'early_stopping_patience': 10,
            'scheduler': 'reduce_on_plateau',  # or 'cosine', 'step'
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
        },

        # Data loading
        'data': {
            'train_split': 0.8,       # Train/val split ratio
            'num_workers': 4,         # DataLoader workers
            'pin_memory': True,       # Pin memory for GPU transfer
            'prefetch_factor': 2,     # Prefetch batches
        },

        # Loss function
        'loss': {
            'type': 'l1',             # 'l1', 'mse', 'combined'
            'combined_alpha': 0.5,    # L1 weight if using combined
            'combined_beta': 0.5,     # MSE weight if using combined
        },

        # Validation (for Q1 testing)
        'validation': {
            'overfitting_iterations': 100,
            'overfitting_lr': 1e-3,
            'min_loss_reduction': 0.5,  # 50% reduction for pass
        },

        # Paths (to be updated per-project)
        'paths': {
            'data_dir': './data',
            'output_dir': './output',
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
        },
    }

    return config


def get_device() -> torch.device:
    """
    Get best available device for PyTorch.

    Returns:
        torch.device for 'cuda' if available, 'mps' for Apple Silicon,
        otherwise 'cpu'
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def print_config(config: Dict[str, Any] = None) -> None:
    """
    Pretty print configuration.

    Args:
        config: Configuration dict. Uses default if None.
    """
    if config is None:
        config = get_config()

    print("=" * 50)
    print("Configuration")
    print("=" * 50)

    for section, params in config.items():
        print(f"\n[{section}]")
        for key, value in params.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Print default configuration
    print_config()

    # Print device info
    device = get_device()
    print(f"\nDevice: {device}")
