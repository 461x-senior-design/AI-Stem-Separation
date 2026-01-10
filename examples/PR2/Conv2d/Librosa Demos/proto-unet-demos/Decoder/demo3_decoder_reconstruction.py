#!/usr/bin/env python3
"""
PROTO-UNET DEMO 3: Decoder Architecture
=========================================

Learned reconstruction through progressive upsampling.

This demo explores HOW decoders work:
1. Upsampling strategies: Nearest, bilinear, transposed convolution
2. Feature refinement: Combining coarse + fine information
3. Reconstruction quality: Different approaches yield different results
4. Decoder symmetry: Mirror structure to encoder

We'll process your acapella vocal through a complete encode-decode cycle:
- Compress through encoder (downsampling)
- Store bottleneck representation
- Reconstruct through decoder (upsampling)
- Compare different upsampling methods

Expected: See how upsampling strategy affects reconstruction quality
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, ndimage

print("="*70)
print("PROTO-UNET DEMO 3: DECODER RECONSTRUCTION")
print("="*70)
print("\nExploring: Learned reconstruction strategies\n")

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
# UPSAMPLING STRATEGIES
# ============================================

def upsample_nearest(encoded, target_shape):
    """
    Nearest-neighbor upsampling (simplest, blocky)
    """
    freq_bins, time_frames = target_shape

    # Repeat each value
    upsampled = np.repeat(np.repeat(encoded, 2, axis=0), 2, axis=1)

    # Crop/pad to target
    upsampled = adjust_to_target_shape(upsampled, target_shape)

    return upsampled

def upsample_bilinear(encoded, target_shape):
    """
    Bilinear interpolation upsampling (smoother)
    """
    from scipy.ndimage import zoom

    freq_bins, time_frames = target_shape
    current_freq, current_time = encoded.shape

    # Calculate zoom factors
    zoom_freq = freq_bins / current_freq
    zoom_time = time_frames / current_time

    # Apply bilinear interpolation
    upsampled = zoom(encoded, (zoom_freq, zoom_time), order=1)

    # Ensure exact target shape
    upsampled = adjust_to_target_shape(upsampled, target_shape)

    return upsampled

def upsample_transposed_conv(encoded, target_shape):
    """
    Transposed convolution upsampling (learnable filters)
    This simulates learnable upsampling with fixed filters
    """
    # First, do nearest neighbor upsampling
    upsampled = upsample_nearest(encoded, target_shape)

    # Apply a smoothing kernel (simulates learned transposed conv)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16

    # Convolve
    padded = np.pad(upsampled, ((1, 1), (1, 1)), mode='reflect')
    refined = signal.convolve2d(padded, kernel, mode='valid')

    # Ensure target shape
    refined = adjust_to_target_shape(refined, target_shape)

    return refined

def upsample_subpixel(encoded, target_shape):
    """
    Sub-pixel convolution (pixel shuffle) upsampling
    Rearranges features for better quality
    """
    # Upsample by 2x using pixel rearrangement idea
    freq, time = encoded.shape

    # Create 4 shifted copies (simulating sub-pixel arrangement)
    upsampled = np.zeros((freq * 2, time * 2))

    upsampled[0::2, 0::2] = encoded
    upsampled[0::2, 1::2] = encoded
    upsampled[1::2, 0::2] = encoded
    upsampled[1::2, 1::2] = encoded

    # Apply smoothing
    kernel = np.ones((2, 2)) / 4
    padded = np.pad(upsampled, ((1, 1), (1, 1)), mode='reflect')
    smoothed = signal.convolve2d(padded, kernel, mode='valid')

    # Adjust to target
    smoothed = adjust_to_target_shape(smoothed, target_shape)

    return smoothed

def adjust_to_target_shape(data, target_shape):
    """Crop or pad to match exact target shape"""
    freq_bins, time_frames = target_shape
    current_freq, current_time = data.shape

    # Crop if needed
    if current_freq > freq_bins:
        data = data[:freq_bins, :]
    if current_time > time_frames:
        data = data[:, :time_frames]

    # Pad if needed
    if current_freq < freq_bins:
        data = np.pad(data, ((0, freq_bins - current_freq), (0, 0)), mode='reflect')
    if current_time < time_frames:
        data = np.pad(data, ((0, 0), (0, time_frames - current_time)), mode='reflect')

    return data

# ============================================
# ENCODER/DECODER BLOCKS
# ============================================

def simple_encoder(spectrogram):
    """
    Simple 4-level encoder
    Returns all intermediate representations
    """
    print("\n[ENCODER PATH]")

    # Simple blur + downsample at each level
    kernel = np.ones((3, 3)) / 9

    enc1 = signal.convolve2d(np.pad(spectrogram, 1, mode='reflect'), kernel, mode='valid')[::2, ::2]
    print(f"  Level 1: {spectrogram.shape} -> {enc1.shape}")

    enc2 = signal.convolve2d(np.pad(enc1, 1, mode='reflect'), kernel, mode='valid')[::2, ::2]
    print(f"  Level 2: {enc1.shape} -> {enc2.shape}")

    enc3 = signal.convolve2d(np.pad(enc2, 1, mode='reflect'), kernel, mode='valid')[::2, ::2]
    print(f"  Level 3: {enc2.shape} -> {enc3.shape}")

    enc4 = signal.convolve2d(np.pad(enc3, 1, mode='reflect'), kernel, mode='valid')[::2, ::2]
    print(f"  Level 4 (bottleneck): {enc3.shape} -> {enc4.shape}")

    return enc4, [spectrogram, enc1, enc2, enc3]

def decoder_with_strategy(bottleneck, encoder_shapes, strategy='nearest'):
    """
    Decoder using specified upsampling strategy
    """
    print(f"\n[DECODER PATH - {strategy.upper()}]")

    upsample_fn = {
        'nearest': upsample_nearest,
        'bilinear': upsample_bilinear,
        'transposed': upsample_transposed_conv,
        'subpixel': upsample_subpixel,
    }[strategy]

    # Decode level by level
    dec4 = upsample_fn(bottleneck, encoder_shapes[2].shape)
    print(f"  Level 4: {bottleneck.shape} -> {dec4.shape}")

    dec3 = upsample_fn(dec4, encoder_shapes[1].shape)
    print(f"  Level 3: {dec4.shape} -> {dec3.shape}")

    dec2 = upsample_fn(dec3, encoder_shapes[0].shape)
    print(f"  Level 2: {dec3.shape} -> {dec2.shape}")

    dec1 = upsample_fn(dec2, encoder_shapes[0].shape)
    print(f"  Level 1: {dec2.shape} -> {dec1.shape}")

    # Final reconstruction to original size
    reconstruction = adjust_to_target_shape(dec1, encoder_shapes[0].shape)

    return reconstruction

# ============================================
# QUALITY METRICS
# ============================================

def compute_reconstruction_metrics(original, reconstructed):
    """Compute various quality metrics"""

    # Mean Squared Error
    mse = np.mean((original - reconstructed)**2)

    # Peak Signal-to-Noise Ratio
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')

    # Structural similarity (simple version)
    mean_orig = np.mean(original)
    mean_recon = np.mean(reconstructed)
    std_orig = np.std(original)
    std_recon = np.std(reconstructed)
    covariance = np.mean((original - mean_orig) * (reconstructed - mean_recon))

    c1 = 0.01
    c2 = 0.03
    ssim = ((2 * mean_orig * mean_recon + c1) * (2 * covariance + c2)) / \
           ((mean_orig**2 + mean_recon**2 + c1) * (std_orig**2 + std_recon**2 + c2))

    # High-frequency preservation
    high_freq_orig = np.sum(original[original.shape[0]//2:, :]**2)
    high_freq_recon = np.sum(reconstructed[original.shape[0]//2:, :]**2)
    hf_preservation = high_freq_recon / (high_freq_orig + 1e-8)

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'hf_preservation': hf_preservation
    }

# ============================================
# MAIN PROCESSING
# ============================================

print("\n[LOADING AUDIO]")
audio, sr = librosa.load(CONFIG['vocal_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
print(f"  Loaded: {len(audio)} samples @ {sr}Hz")

print("\n[CREATING SPECTROGRAM]")
stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
magnitude = np.abs(stft)
phase = np.angle(stft)
print(f"  Spectrogram: {magnitude.shape} (freq x time)")

# Normalize
magnitude_norm = magnitude / (np.max(magnitude) + 1e-8)

# ============================================
# ENCODE ONCE
# ============================================

print("\n" + "="*70)
print("ENCODING TO BOTTLENECK")
print("="*70)

bottleneck, encoder_shapes = simple_encoder(magnitude_norm)
print(f"\nBottleneck: {bottleneck.shape}")
print(f"Compression: {magnitude_norm.size / bottleneck.size:.1f}x")

# ============================================
# DECODE WITH DIFFERENT STRATEGIES
# ============================================

print("\n" + "="*70)
print("TESTING DECODER STRATEGIES")
print("="*70)

strategies = ['nearest', 'bilinear', 'transposed', 'subpixel']
reconstructions = {}
metrics_all = {}
audio_reconstructions = {}

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*70}")

    # Decode
    reconstruction = decoder_with_strategy(bottleneck, encoder_shapes, strategy)
    reconstructions[strategy] = reconstruction

    # Compute metrics
    metrics = compute_reconstruction_metrics(magnitude_norm, reconstruction)
    metrics_all[strategy] = metrics

    print(f"\n  Quality Metrics:")
    print(f"    MSE:              {metrics['mse']:.6f}")
    print(f"    PSNR:             {metrics['psnr']:.2f} dB")
    print(f"    SSIM:             {metrics['ssim']:.4f}")
    print(f"    HF Preservation:  {metrics['hf_preservation']:.2%}")

    # Reconstruct audio
    magnitude_recon = reconstruction * np.max(magnitude)
    stft_recon = magnitude_recon * np.exp(1j * phase)
    audio_recon = librosa.istft(stft_recon, hop_length=CONFIG['hop_length'], n_fft=CONFIG['n_fft'])
    audio_recon = audio_recon / (np.max(np.abs(audio_recon)) + 1e-8)
    audio_reconstructions[strategy] = audio_recon

    # Save audio
    sf.write(f"{CONFIG['output_dir']}/demo3_{strategy}.wav", audio_recon, sr)
    print(f"    Saved: demo3_{strategy}.wav")

# Save original for comparison
audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
sf.write(f"{CONFIG['output_dir']}/demo3_original.wav", audio_norm, sr)

# ============================================
# VISUALIZATION
# ============================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Figure 1: Compare all reconstructions
fig, axes = plt.subplots(5, 1, figsize=(14, 14))

# Original
axes[0].imshow(librosa.amplitude_to_db(magnitude, ref=np.max),
               aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title('Original Spectrogram')
axes[0].set_ylabel('Frequency')

# Each strategy
for idx, strategy in enumerate(strategies):
    recon = reconstructions[strategy]
    magnitude_recon = recon * np.max(magnitude)

    axes[idx + 1].imshow(librosa.amplitude_to_db(magnitude_recon, ref=np.max),
                          aspect='auto', origin='lower', cmap='viridis')

    metrics = metrics_all[strategy]
    title = f'{strategy.capitalize()} (PSNR: {metrics["psnr"]:.1f}dB, SSIM: {metrics["ssim"]:.3f})'
    axes[idx + 1].set_title(title)
    axes[idx + 1].set_ylabel('Frequency')

    if idx == len(strategies) - 1:
        axes[idx + 1].set_xlabel('Time (frames)')

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo3_decoder_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {CONFIG['figures_dir']}/demo3_decoder_comparison.png")

# Figure 2: Metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Decoder Strategy Comparison', fontsize=14, fontweight='bold')

strategies_labels = [s.capitalize() for s in strategies]

# MSE (lower is better)
mse_values = [metrics_all[s]['mse'] for s in strategies]
axes[0, 0].bar(strategies_labels, mse_values, color='steelblue')
axes[0, 0].set_title('Mean Squared Error (lower is better)')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# PSNR (higher is better)
psnr_values = [metrics_all[s]['psnr'] for s in strategies]
axes[0, 1].bar(strategies_labels, psnr_values, color='coral')
axes[0, 1].set_title('Peak Signal-to-Noise Ratio (higher is better)')
axes[0, 1].set_ylabel('PSNR (dB)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# SSIM (higher is better)
ssim_values = [metrics_all[s]['ssim'] for s in strategies]
axes[1, 0].bar(strategies_labels, ssim_values, color='seagreen')
axes[1, 0].set_title('Structural Similarity (higher is better)')
axes[1, 0].set_ylabel('SSIM')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# High-frequency preservation (closer to 1.0 is better)
hf_values = [metrics_all[s]['hf_preservation'] for s in strategies]
axes[1, 1].bar(strategies_labels, hf_values, color='mediumpurple')
axes[1, 1].set_title('High-Frequency Preservation (closer to 1.0 is better)')
axes[1, 1].set_ylabel('Preservation Ratio')
axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo3_metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {CONFIG['figures_dir']}/demo3_metrics_comparison.png")

# ============================================
# INSIGHTS
# ============================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. UPSAMPLING STRATEGIES:")
print("   - Nearest: Fast but blocky artifacts")
print("   - Bilinear: Smooth but can blur details")
print("   - Transposed: Learnable, good for complex patterns")
print("   - Subpixel: Efficient, preserves sharpness")

print("\n2. QUALITY TRADE-OFFS:")
best_psnr = max(strategies, key=lambda s: metrics_all[s]['psnr'])
best_ssim = max(strategies, key=lambda s: metrics_all[s]['ssim'])
best_hf = max(strategies, key=lambda s: metrics_all[s]['hf_preservation'])
print(f"   - Best PSNR: {best_psnr} ({metrics_all[best_psnr]['psnr']:.2f}dB)")
print(f"   - Best SSIM: {best_ssim} ({metrics_all[best_ssim]['ssim']:.4f})")
print(f"   - Best HF preservation: {best_hf} ({metrics_all[best_hf]['hf_preservation']:.2%})")

print("\n3. DECODER SYMMETRY:")
print("   - Decoder mirrors encoder structure")
print("   - Each decoder level reverses corresponding encoder level")
print("   - Symmetric architecture enables reconstruction")

print("\n4. RECONSTRUCTION CHALLENGE:")
print("   - Bottleneck loses information (compression)")
print("   - Decoder must 'hallucinate' missing details")
print("   - Better upsampling = better hallucination")

print("\n" + "="*70)
print("✓ DEMO 3 COMPLETE")
print("="*70)
print("\nKey Insight: Decoder upsampling strategy matters!")
print("Different methods have different trade-offs for reconstruction quality.")
print("\nListen to compare:")
for strategy in strategies:
    print(f"  - demo3_{strategy}.wav")
print("\nView visualizations:")
print("  - demo3_decoder_comparison.png (spectrograms)")
print("  - demo3_metrics_comparison.png (quality metrics)")
