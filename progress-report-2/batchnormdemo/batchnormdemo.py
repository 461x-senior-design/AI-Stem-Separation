import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import zoom
import soundfile as sf
import IPython.diisplay as ipd

print("="*70)
print("Batch Normalization Demo")
print("="*70)

# Configuration ADAPTED FROM CAMERON'S PROTO-UNET DEMO
CONFIG = {
    'vocal_path': './input_files/acapella.mp3',
    'output_dir': './output_batch',
    'figures_dir': './figures',
    'sr': 22050,
    'duration': 7.0,
    'n_fft': 2048,
    'hop_length': 512,
}


# Create output directories
Path(CONFIG['output_dir']).mkdir(exist_ok=True)
Path(CONFIG['figures_dir']).mkdir(exist_ok=True)

print("✓ Configuration loaded")

# Simple batch normalization algorithm implemented for demonstration.
# Adapted from Cameron's proto-unet demo.
# Note: This is NOT optimized for performance. We will be using pytorch's BatchNorm in real models.
def batch_normalize(data, eps=1e-5):
    """
    Batch normalization.
    Normalize to mean=0, std=1 across spatial dimensions.
    """
    # Compute statistics
    mean = np.mean(data)
    std = np.std(data)

    # Normalize
    normalized = (data - mean) / (std + eps)

    # Scale and shift (learnable in real networks)
    gamma = 1.0
    beta = 0.0

    output = gamma * normalized + beta

    # Return data + stats for analysis
    stats = {
        'mean': mean,
        'std': std,
        'min': np.min(output),
        'max': np.max(output),
    }
    return output, stats


# Adapted from Cameron's proto-unet demo.
def encoder_block_no_norm(input_data, level, weight_scale=1.0):
    """
    Encoder block WITHOUT normalization
    - Activations can explode or vanish
    - Distribution shifts across layers
    """
    # Convolution with weight scaling (in other words, convolve with padding to maintain size)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16 * weight_scale
    
    padded = np.pad(input_data, 1, mode='reflect')
    conv = signal.convolve2d(padded, kernel, mode='valid')
    
    # ReLU activation (no normalization!)
    activated = np.maximum(conv, 0)
    
    # Downsample
    downsampled = activated[::2, ::2]
    
    # Collect statistics
    stats = {
        'mean': np.mean(activated),
        'std': np.std(activated),
        'min': np.min(activated),
        'max': np.max(activated),
        'zeros': np.sum(activated == 0) / activated.size * 100  # Sparsity
    }
    
    return downsampled, activated, stats


# Adapted from Cameron's proto-unet demo.
def encoder_block_with_batchnorm(input_data, level, weight_scale=1.0):
    """
    Encoder block WITH BatchNorm
    - Stabilized activations
    - Consistent distributions
    """
    # Convolution
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16 * weight_scale
    
    padded = np.pad(input_data, 1, mode='reflect')
    conv = signal.convolve2d(padded, kernel, mode='valid')
    
    # **KEY ADDITION** BatchNorm BEFORE activation
    normalized, norm_stats = batch_normalize(conv)
    
    # ReLU activation
    activated = np.maximum(normalized, 0)
    
    # Downsample
    downsampled = activated[::2, ::2]
    
    # Collect statistics
    stats = {
        'mean': np.mean(activated),
        'std': np.std(activated),
        'min': np.min(activated),
        'max': np.max(activated),
        'zeros': np.sum(activated == 0) / activated.size * 100,
        'norm_mean': norm_stats['mean'],
        'norm_std': norm_stats['std']
    }
    
    return downsampled, activated, stats


# Simulate Training instability WITHOUT BatchNorm
# Adapted from Cameron's proto-unet demo.
def simulate_training_instability(spectrogram, use_batchnorm=False):
    """
    Simulate what happens during  training with/without BatchNorm.
    We'll use increasing weight scales to simulate learning dynamics
    """

    print("="*50)
    print(f"{'WITH' if use_batchnorm else 'WITHOUT'} BATCHNORM")
    print("="*50)

    # Choose encoder function
    encoder_fn = encoder_block_with_batchnorm if use_batchnorm else encoder_block_no_norm

    # Initialize storage
    outputs = {}
    activations = {}
    all_stats = {}

    # Process through levels (input)
    current = spectrogram
    weight_scales = [1.0, 1.2, 1.5, 1.8]

    for level in range(1, 5):
        weight_scale = weight_scales[level - 1]
        current, act, stats = encoder_fn(current, level, weight_scales[level-1])

        # Store for analysis
        outputs[level] = current
        activations[level] = act
        all_stats[level] = stats

        # Print progress
        print(f"\n  Level {level} (weight_scale={weight_scale}):")
        print(f"    Shape: {current.shape}")
        print(f"    Mean:  {stats['mean']:.4f}")
        print(f"    Std:   {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    Zeros: {stats['zeros']:.1f}%")
        
        if use_batchnorm:
            print(f"    (Pre-norm mean: {stats['norm_mean']:.4f}, std: {stats['norm_std']:.4f})")
    
    return outputs, activations, all_stats


