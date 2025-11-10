import numpy as np
import librosa
# import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
# from scipy.ndimage import zoom
# import soundfile as sf
# import IPython.display as ipd

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


# Estimate gradient magnitudes during backpropagation
# Adapted from Cameron's proto-unet demo.
def simulate_gradient_flow(all_stats, use_batchnorm=False):
    """
    Simulate gradient magnitudes flowing backward
    Stable gradients are crucial for training
    """
    print(f"\n  Gradient Flow Analysis:")
    
    gradient_magnitudes = []
    
    for level in range(1, 5):
        stats = all_stats[level]
        
        # Gradient magnitude proportional to activation std
        if use_batchnorm:
            # BatchNorm keeps gradients stable (GOOD)
            grad_mag = 1.0 / (level * 0.5)  # Gradual decay
        else:
            # Without BatchNorm, gradients can explode or vanish (BAD)
            grad_mag = stats['std'] * (1.5 ** level)  # Exponential growth
        
        gradient_magnitudes.append(grad_mag)
        
        status = "✓ STABLE" if 0.1 < grad_mag < 10 else "✗ UNSTABLE"
        print(f"    Level {level}: grad_magnitude = {grad_mag:.4f} {status}")
    
    return gradient_magnitudes


# Now we load audio
# ========================
# Main Processing Pipeline
# ========================
# Adapted from Cameron's proto-unet demo.

# Load audio
print("[LOADING AUDIO]")
audio, sr = librosa.load(CONFIG['vocal_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
print(f"  Loaded: {len(audio)} samples @ {sr}Hz")

# Compute spectrogram
print("\n[CREATING SPECTROGRAM]")
stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
magnitude = np.abs(stft)
print(f"  Spectrogram: {magnitude.shape} (freq x time)")

# Normalize input to [0, 1]
magnitude_norm = magnitude / (np.max(magnitude) + 1e-8)

# ============================================
# Comparison: WITH vs WITHOUT BatchNorm
# ============================================

print("\n" + "="*70)
print("SIMULATING TRAINING DYNAMICS")
print("="*70)

# Path 1: WITHOUT BatchNorm
outputs_no_bn, acts_no_bn, stats_no_bn = simulate_training_instability(magnitude_norm, use_batchnorm=False)
grads_no_bn = simulate_gradient_flow(stats_no_bn, use_batchnorm=False)

# Path 2: WITH BatchNorm
outputs_with_bn, acts_with_bn, stats_with_bn = simulate_training_instability(magnitude_norm, use_batchnorm=True)
grads_with_bn = simulate_gradient_flow(stats_with_bn, use_batchnorm=True)


# ============================================
# VISUALIZATION
# ============================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Figure 1: Activation distributions at each level
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Activation Distributions Across Encoder Levels', fontsize=14, fontweight='bold')

for level in range(1, 5):
    # WITHOUT BatchNorm
    ax_left = axes[level - 1, 0]
    act_data = acts_no_bn[level].flatten()
    ax_left.hist(act_data, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax_left.set_title(f'Level {level} WITHOUT BatchNorm')
    ax_left.set_xlabel('Activation Value')
    ax_left.set_ylabel('Frequency')
    ax_left.axvline(x=np.mean(act_data), color='blue', linestyle='--', label=f'Mean: {np.mean(act_data):.3f}')
    ax_left.axvline(x=np.std(act_data), color='green', linestyle='--', label=f'Std: {np.std(act_data):.3f}')
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)

    # WITH BatchNorm
    ax_right = axes[level - 1, 1]
    act_data_bn = acts_with_bn[level].flatten()
    ax_right.hist(act_data_bn, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax_right.set_title(f'Level {level} WITH BatchNorm')
    ax_right.set_xlabel('Activation Value')
    ax_right.set_ylabel('Frequency')
    ax_right.axvline(x=np.mean(act_data_bn), color='blue', linestyle='--', label=f'Mean: {np.mean(act_data_bn):.3f}')
    ax_right.axvline(x=np.std(act_data_bn), color='red', linestyle='--', label=f'Std: {np.std(act_data_bn):.3f}')
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo4_activation_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {CONFIG['figures_dir']}/demo4_activation_distributions.png")

# Figure 2: Activation statistics across levels
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Activation Statistics Across Layers', fontsize=14, fontweight='bold')

levels = [1, 2, 3, 4]

# Mean activations
means_no_bn = [stats_no_bn[l]['mean'] for l in levels]
means_with_bn = [stats_with_bn[l]['mean'] for l in levels]
axes[0, 0].plot(levels, means_no_bn, 'o-', linewidth=2, markersize=8, label='Without BatchNorm', color='red')
axes[0, 0].plot(levels, means_with_bn, 's-', linewidth=2, markersize=8, label='With BatchNorm', color='green')
axes[0, 0].set_title('Mean Activation')
axes[0, 0].set_xlabel('Encoder Level')
axes[0, 0].set_ylabel('Mean')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Std deviation
stds_no_bn = [stats_no_bn[l]['std'] for l in levels]
stds_with_bn = [stats_with_bn[l]['std'] for l in levels]
axes[0, 1].plot(levels, stds_no_bn, 'o-', linewidth=2, markersize=8, label='Without BatchNorm', color='red')
axes[0, 1].plot(levels, stds_with_bn, 's-', linewidth=2, markersize=8, label='With BatchNorm', color='green')
axes[0, 1].set_title('Standard Deviation')
axes[0, 1].set_xlabel('Encoder Level')
axes[0, 1].set_ylabel('Std')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Max activation (indicator of explosion)
maxs_no_bn = [stats_no_bn[l]['max'] for l in levels]
maxs_with_bn = [stats_with_bn[l]['max'] for l in levels]
axes[1, 0].plot(levels, maxs_no_bn, 'o-', linewidth=2, markersize=8, label='Without BatchNorm', color='red')
axes[1, 0].plot(levels, maxs_with_bn, 's-', linewidth=2, markersize=8, label='With BatchNorm', color='green')
axes[1, 0].set_title('Max Activation (explosion indicator)')
axes[1, 0].set_xlabel('Encoder Level')
axes[1, 0].set_ylabel('Max')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Gradient flow
axes[1, 1].plot(levels, grads_no_bn, 'o-', linewidth=2, markersize=8, label='Without BatchNorm', color='red')
axes[1, 1].plot(levels, grads_with_bn, 's-', linewidth=2, markersize=8, label='With BatchNorm', color='green')
axes[1, 1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Vanishing threshold')
axes[1, 1].axhline(y=10, color='purple', linestyle='--', alpha=0.5, label='Exploding threshold')
axes[1, 1].set_title('Gradient Magnitude (backward pass)')
axes[1, 1].set_xlabel('Encoder Level')
axes[1, 1].set_ylabel('Gradient Magnitude')
axes[1, 1].set_yscale('log')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo4_statistics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {CONFIG['figures_dir']}/demo4_statistics_comparison.png")

# Figure 3: Heatmap comparison
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Feature Maps: With vs Without BatchNorm', fontsize=14, fontweight='bold')

for level in range(1, 5):
    # WITHOUT BatchNorm
    ax_left = axes[level - 1, 0]
    data_no_bn = acts_no_bn[level]
    im1 = ax_left.imshow(np.log10(np.abs(data_no_bn) + 1e-8), aspect='auto', origin='lower',
                         cmap='viridis', interpolation='nearest')
    ax_left.set_title(f'Level {level} WITHOUT BatchNorm')
    ax_left.set_ylabel('Frequency')
    plt.colorbar(im1, ax=ax_left, label='Log Magnitude')

    # WITH BatchNorm
    ax_right = axes[level - 1, 1]
    data_with_bn = acts_with_bn[level]
    im2 = ax_right.imshow(np.log10(np.abs(data_with_bn) + 1e-8), aspect='auto', origin='lower',
                          cmap='viridis', interpolation='nearest')
    ax_right.set_title(f'Level {level} WITH BatchNorm')
    ax_right.set_ylabel('Frequency')
    plt.colorbar(im2, ax=ax_right, label='Log Magnitude')

    if level == 4:
        ax_left.set_xlabel('Time')
        ax_right.set_xlabel('Time')

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo4_feature_maps.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {CONFIG['figures_dir']}/demo4_feature_maps.png")

# ============================================
# QUANTITATIVE COMPARISON
# ============================================

print("\n" + "="*70)
print("QUANTITATIVE ANALYSIS")
print("="*70)

print("\nActivation Stability (std across levels):")
std_variance_no_bn = np.var([stats_no_bn[l]['std'] for l in levels])
std_variance_with_bn = np.var([stats_with_bn[l]['std'] for l in levels])
print(f"  WITHOUT BatchNorm: variance = {std_variance_no_bn:.6f} (unstable)")
print(f"  WITH BatchNorm:    variance = {std_variance_with_bn:.6f} (stable)")
print(f"  Improvement: {(1 - std_variance_with_bn / std_variance_no_bn) * 100:.1f}% more stable")

print("\nGradient Flow Health:")
grad_variance_no_bn = np.var(grads_no_bn)
grad_variance_with_bn = np.var(grads_with_bn)
print(f"  WITHOUT BatchNorm: variance = {grad_variance_no_bn:.6f} (erratic)")
print(f"  WITH BatchNorm:    variance = {grad_variance_with_bn:.6f} (smooth)")
print(f"  Improvement: {(1 - grad_variance_with_bn / grad_variance_no_bn) * 100:.1f}% smoother")

print("\nDead Neurons (ReLU zeros):")
avg_zeros_no_bn = np.mean([stats_no_bn[l]['zeros'] for l in levels])
avg_zeros_with_bn = np.mean([stats_with_bn[l]['zeros'] for l in levels])
print(f"  WITHOUT BatchNorm: {avg_zeros_no_bn:.1f}% dead")
print(f"  WITH BatchNorm:    {avg_zeros_with_bn:.1f}% dead")

# ============================================
# INSIGHTS
# ============================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. INTERNAL COVARIATE SHIFT:")
print("   - WITHOUT BatchNorm: Input distributions shift dramatically per layer")
print("   - WITH BatchNorm: Distributions stay consistent (mean~0, std~1)")

print("\n2. GRADIENT FLOW:")
print("   - WITHOUT BatchNorm: Gradients can explode or vanish")
print("   - WITH BatchNorm: Gradients flow smoothly backward")

print("\n3. TRAINING STABILITY:")
print("   - WITHOUT BatchNorm: Sensitive to weight initialization and learning rate")
print("   - WITH BatchNorm: More robust, can train with higher learning rates")

print("\n4. ACTIVATION DYNAMICS:")
print("   - WITHOUT BatchNorm: Activations grow/shrink unpredictably")
print("   - WITH BatchNorm: Activations stay in healthy range")

print("\n5. WHY IT MATTERS FOR U-NET:")
print("   - Deep encoder-decoder needs stable activations")
print("   - Skip connections benefit from normalized features")
print("   - Enables faster convergence during training")

print("\n" + "="*70)
print("✓ DEMO 4 COMPLETE")
print("="*70)
print("\nKey Insight: BatchNorm stabilizes training!")
print("Without it, deep networks struggle with gradient flow and activation explosions.")
print("\nView visualizations:")
print("  - demo4_activation_distributions.png (histograms per layer)")
print("  - demo4_statistics_comparison.png (quantitative metrics)")
print("  - demo4_feature_maps.png (spatial patterns)")

