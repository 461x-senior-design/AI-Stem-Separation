#!/usr/bin/env python3
"""
PROTO-UNET DEMO 2: Encoder Architecture
=========================================

Hierarchical feature extraction through progressive downsampling.

This demo explores HOW encoders work:
1. Multi-scale representation: Each level captures different details
2. Receptive field growth: Deeper layers "see" more context
3. Feature abstraction: Low-level edges -> high-level patterns
4. Compression trade-offs: Resolution vs. semantic information

We'll process your acapella vocal and visualize each encoder level:
- Level 1: Raw spectral details (edges, onsets)
- Level 2: Local patterns (formants, transients)
- Level 3: Mid-level structures (phonemes, pitch contours)
- Level 4: High-level features (vocal characteristics)

Expected: See hierarchical abstraction from fine to coarse
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, ndimage

print("="*70)
print("PROTO-UNET DEMO 2: ENCODER HIERARCHY")
print("="*70)
print("\nExploring: Multi-scale feature extraction\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': '../PreviousResearch/librosa-demo 2/pre/rtg/100-window/isolated_vocal.wav',
    'output_dir': 'output',
    'figures_dir': 'figures',
    'sr': 22050,
    'duration': 4.7,
    'n_fft': 2048,
    'hop_length': 512,
}

Path(CONFIG['output_dir']).mkdir(exist_ok=True)
Path(CONFIG['figures_dir']).mkdir(exist_ok=True)

# ============================================
# ENCODER COMPONENTS
# ============================================

def apply_convolution(input_data, kernel_type='edge'):
    """
    Apply different convolution kernels to extract features
    Args:
        input_data: Input spectrogram
        kernel_type: Type of feature to extract
    Returns:
        features: Extracted feature map
    """
    if kernel_type == 'edge':
        # Horizontal edge detection (sustained notes)
        kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
    elif kernel_type == 'vertical':
        # Vertical edge detection (onsets)
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
    elif kernel_type == 'blur':
        # Smoothing (noise reduction)
        kernel = np.ones((3, 3), dtype=np.float32) / 9
    elif kernel_type == 'sharpen':
        # Sharpening (enhance details)
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)
    else:  # 'identity'
        kernel = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.float32)

    # Apply convolution
    padded = np.pad(input_data, ((1, 1), (1, 1)), mode='reflect')
    features = signal.convolve2d(padded, kernel, mode='valid')

    return features

def encoder_level(input_data, level, extract_multiple_features=False):
    """
    Single encoder level with optional multi-feature extraction
    Args:
        input_data: Input from previous level
        level: Current depth level
        extract_multiple_features: If True, extract multiple feature types
    Returns:
        output: Downsampled output for next level
        feature_maps: Dictionary of extracted features (for visualization)
    """
    print(f"\n  Level {level}: Input shape {input_data.shape}")

    feature_maps = {}

    if extract_multiple_features:
        # Extract multiple types of features
        feature_maps['edge'] = apply_convolution(input_data, 'edge')
        feature_maps['vertical'] = apply_convolution(input_data, 'vertical')
        feature_maps['blur'] = apply_convolution(input_data, 'blur')
        feature_maps['sharpen'] = apply_convolution(input_data, 'sharpen')

        # Combine features (simple average for this demo)
        combined = np.mean([feature_maps['edge'],
                           feature_maps['vertical'],
                           feature_maps['blur'],
                           feature_maps['sharpen']], axis=0)
    else:
        # Single feature type (blur for simplicity)
        combined = apply_convolution(input_data, 'blur')
        feature_maps['blur'] = combined

    # Activation (ReLU-like)
    activated = np.maximum(combined, 0)

    # Downsample by 2x (MaxPooling)
    downsampled = activated[::2, ::2]

    print(f"    Features extracted: {list(feature_maps.keys())}")
    print(f"    Output shape: {downsampled.shape}")

    # Calculate receptive field size (approximate)
    receptive_field = 3 * (2 ** (level - 1))
    print(f"    Receptive field: ~{receptive_field}x{receptive_field} pixels")

    return downsampled, feature_maps

# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def analyze_feature_statistics(feature_map, level):
    """Compute and display statistics for a feature map"""
    mean_activation = np.mean(feature_map)
    std_activation = np.std(feature_map)
    sparsity = np.sum(feature_map == 0) / feature_map.size * 100
    max_activation = np.max(feature_map)

    print(f"\n    Statistics:")
    print(f"      Mean activation: {mean_activation:.4f}")
    print(f"      Std deviation:   {std_activation:.4f}")
    print(f"      Sparsity:        {sparsity:.1f}%")
    print(f"      Max activation:  {max_activation:.4f}")

    return {
        'mean': mean_activation,
        'std': std_activation,
        'sparsity': sparsity,
        'max': max_activation
    }

def compute_frequency_emphasis(feature_map, sr):
    """Analyze which frequency bands are emphasized"""
    freq_bins = feature_map.shape[0]

    # Divide into frequency bands
    low_band = feature_map[:freq_bins//4, :]
    mid_band = feature_map[freq_bins//4:3*freq_bins//4, :]
    high_band = feature_map[3*freq_bins//4:, :]

    low_energy = np.sum(low_band**2)
    mid_energy = np.sum(mid_band**2)
    high_energy = np.sum(high_band**2)

    total = low_energy + mid_energy + high_energy + 1e-8

    print(f"    Frequency emphasis:")
    print(f"      Low (0-{sr//8}Hz):     {low_energy/total*100:.1f}%")
    print(f"      Mid ({sr//8}-{3*sr//8}Hz):  {mid_energy/total*100:.1f}%")
    print(f"      High ({3*sr//8}+Hz):    {high_energy/total*100:.1f}%")

# ============================================
# MAIN PROCESSING
# ============================================

print("\n[LOADING AUDIO]")
audio, sr = librosa.load(CONFIG['vocal_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
print(f"  Loaded: {len(audio)} samples @ {sr}Hz")

print("\n[CREATING SPECTROGRAM]")
stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
magnitude = np.abs(stft)
print(f"  Spectrogram: {magnitude.shape} (freq x time)")

# Normalize
magnitude_norm = magnitude / (np.max(magnitude) + 1e-8)

# ============================================
# ENCODER PYRAMID
# ============================================

print("\n" + "="*70)
print("BUILDING ENCODER PYRAMID")
print("="*70)

encoder_outputs = {}
encoder_features = {}
stats_per_level = {}

# Level 0: Input (original spectrogram)
encoder_outputs[0] = magnitude_norm
print(f"\n[INPUT LEVEL 0]")
print(f"  Shape: {magnitude_norm.shape}")
print(f"  This is the raw spectrogram - finest detail")

# Level 1: First encoder
print(f"\n[ENCODER LEVEL 1]")
enc1, features1 = encoder_level(magnitude_norm, 1, extract_multiple_features=True)
encoder_outputs[1] = enc1
encoder_features[1] = features1
stats_per_level[1] = analyze_feature_statistics(enc1, 1)
compute_frequency_emphasis(enc1, sr)

# Level 2: Second encoder
print(f"\n[ENCODER LEVEL 2]")
enc2, features2 = encoder_level(enc1, 2, extract_multiple_features=True)
encoder_outputs[2] = enc2
encoder_features[2] = features2
stats_per_level[2] = analyze_feature_statistics(enc2, 2)
compute_frequency_emphasis(enc2, sr)

# Level 3: Third encoder
print(f"\n[ENCODER LEVEL 3]")
enc3, features3 = encoder_level(enc2, 3, extract_multiple_features=False)
encoder_outputs[3] = enc3
encoder_features[3] = features3
stats_per_level[3] = analyze_feature_statistics(enc3, 3)
compute_frequency_emphasis(enc3, sr)

# Level 4: Fourth encoder (bottleneck)
print(f"\n[ENCODER LEVEL 4 - BOTTLENECK]")
enc4, features4 = encoder_level(enc3, 4, extract_multiple_features=False)
encoder_outputs[4] = enc4
encoder_features[4] = features4
stats_per_level[4] = analyze_feature_statistics(enc4, 4)
compute_frequency_emphasis(enc4, sr)

print(f"\n  Bottleneck reached: {enc4.shape}")
print(f"  Compression ratio: {magnitude_norm.size / enc4.size:.1f}x")

# ============================================
# VISUALIZATION: ENCODER PYRAMID
# ============================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Figure 1: Encoder pyramid showing progressive downsampling
fig, axes = plt.subplots(5, 1, figsize=(14, 12))

for level in range(5):
    data = encoder_outputs[level]

    # Convert to dB scale for visualization
    if level == 0:
        data_db = librosa.amplitude_to_db(data, ref=np.max)
        title = f"Level {level}: Input (Original Spectrogram)"
    else:
        # For feature maps, just use log scale
        data_db = np.log10(np.abs(data) + 1e-8)
        title = f"Level {level}: Encoder Output (Receptive field ~{3 * (2 ** (level-1))}x)"

    im = axes[level].imshow(data_db, aspect='auto', origin='lower',
                             cmap='viridis', interpolation='nearest')
    axes[level].set_title(title)
    axes[level].set_ylabel(f'Freq\n{data.shape[0]} bins')

    if level == 4:
        axes[level].set_xlabel('Time (frames)')

    # Add colorbar
    plt.colorbar(im, ax=axes[level], label='Magnitude (dB)')

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo2_encoder_pyramid.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {CONFIG['figures_dir']}/demo2_encoder_pyramid.png")

# Figure 2: Feature diversity at Level 1 (multiple feature types)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Level 1: Multiple Feature Types', fontsize=14, fontweight='bold')

feature_types = ['edge', 'vertical', 'blur', 'sharpen']
for idx, feat_type in enumerate(feature_types):
    ax = axes[idx // 2, idx % 2]
    feat_data = encoder_features[1][feat_type]
    feat_db = np.log10(np.abs(feat_data) + 1e-8)

    im = ax.imshow(feat_db, aspect='auto', origin='lower',
                   cmap='viridis', interpolation='nearest')
    ax.set_title(f'{feat_type.capitalize()} Detection')
    ax.set_ylabel('Frequency')
    if idx >= 2:
        ax.set_xlabel('Time')
    plt.colorbar(im, ax=ax, label='Magnitude')

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo2_feature_diversity.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {CONFIG['figures_dir']}/demo2_feature_diversity.png")

# Figure 3: Statistics comparison across levels
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Encoder Statistics Across Levels', fontsize=14, fontweight='bold')

levels = list(range(1, 5))

# Mean activation
axes[0, 0].plot(levels, [stats_per_level[l]['mean'] for l in levels], 'o-', linewidth=2, markersize=8)
axes[0, 0].set_title('Mean Activation')
axes[0, 0].set_xlabel('Encoder Level')
axes[0, 0].set_ylabel('Mean')
axes[0, 0].grid(True, alpha=0.3)

# Std deviation
axes[0, 1].plot(levels, [stats_per_level[l]['std'] for l in levels], 'o-', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_title('Standard Deviation')
axes[0, 1].set_xlabel('Encoder Level')
axes[0, 1].set_ylabel('Std')
axes[0, 1].grid(True, alpha=0.3)

# Sparsity
axes[1, 0].plot(levels, [stats_per_level[l]['sparsity'] for l in levels], 'o-', linewidth=2, markersize=8, color='green')
axes[1, 0].set_title('Sparsity (% zeros)')
axes[1, 0].set_xlabel('Encoder Level')
axes[1, 0].set_ylabel('Sparsity (%)')
axes[1, 0].grid(True, alpha=0.3)

# Max activation
axes[1, 1].plot(levels, [stats_per_level[l]['max'] for l in levels], 'o-', linewidth=2, markersize=8, color='red')
axes[1, 1].set_title('Max Activation')
axes[1, 1].set_xlabel('Encoder Level')
axes[1, 1].set_ylabel('Max')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo2_encoder_statistics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {CONFIG['figures_dir']}/demo2_encoder_statistics.png")

# ============================================
# INSIGHTS
# ============================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. MULTI-SCALE REPRESENTATION:")
print("   - Level 1: Fine details (edges, onsets)")
print("   - Level 2: Local patterns (formants)")
print("   - Level 3: Mid-level structures (phonemes)")
print("   - Level 4: High-level features (global characteristics)")

print("\n2. RECEPTIVE FIELD GROWTH:")
level1_rf = 3
level2_rf = 3 * 2
level3_rf = 3 * 4
level4_rf = 3 * 8
print(f"   - Level 1 sees: ~{level1_rf}x{level1_rf} pixels")
print(f"   - Level 2 sees: ~{level2_rf}x{level2_rf} pixels")
print(f"   - Level 3 sees: ~{level3_rf}x{level3_rf} pixels")
print(f"   - Level 4 sees: ~{level4_rf}x{level4_rf} pixels")
print("   Deeper layers have more context!")

print("\n3. COMPRESSION:")
print(f"   - Input size:      {encoder_outputs[0].size} values")
print(f"   - Bottleneck size: {encoder_outputs[4].size} values")
print(f"   - Compression:     {encoder_outputs[0].size / encoder_outputs[4].size:.1f}x")

print("\n4. FEATURE ABSTRACTION:")
print("   - Early layers: Low-level features (edges, textures)")
print("   - Deep layers:  High-level features (semantic patterns)")

print("\n" + "="*70)
print("✓ DEMO 2 COMPLETE")
print("="*70)
print("\nKey Insight: Encoders create hierarchical representations!")
print("Each level captures different scales of information.")
print("\nView visualizations:")
print("  - demo2_encoder_pyramid.png (shows progressive compression)")
print("  - demo2_feature_diversity.png (different feature types)")
print("  - demo2_encoder_statistics.png (quantitative analysis)")
