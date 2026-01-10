#!/usr/bin/env python3
"""
PROTO-UNET DEMO 1: Skip Connections
====================================

The defining feature of U-Net architecture.

This demo explores WHY skip connections matter:
1. Without skips: Information loss through bottleneck
2. With skips: Fine details preserved via lateral connections
3. Visualization: Compare reconstructions side-by-side

We'll process your acapella vocal through:
- Encoder path (downsampling)
- Bottleneck (compressed representation)
- Decoder path (upsampling) WITH and WITHOUT skip connections

Expected: Skip connections preserve high-frequency vocal details
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

print("="*70)
print("PROTO-UNET DEMO 1: SKIP CONNECTIONS")
print("="*70)
print("\nExploring: Why skip connections are the defining U-Net feature\n")

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
# PROTO U-NET COMPONENTS
# ============================================

def encoder_block(spectrogram, level):
    """
    Single encoder block: Extract features and downsample
    Args:
        spectrogram: Input magnitude spectrogram (freq x time)
        level: Encoder depth level (1-4)
    Returns:
        downsampled: Reduced resolution representation
        skip_features: High-res features to skip to decoder
    """
    freq_bins, time_frames = spectrogram.shape

    # Feature extraction (simple averaging convolution)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Pad for valid convolution
    padded = np.pad(spectrogram, ((1, 1), (1, 1)), mode='reflect')
    features = signal.convolve2d(padded, kernel, mode='valid')

    # Store high-resolution features for skip connection
    skip_features = features.copy()

    # Downsample by 2x (MaxPool operation)
    downsampled = features[::2, ::2]

    print(f"  Encoder L{level}: {spectrogram.shape} -> {downsampled.shape} (skip: {skip_features.shape})")

    return downsampled, skip_features

def bottleneck(encoded):
    """
    Bottleneck: Most compressed representation
    This is where information is most constrained
    """
    print(f"  Bottleneck: {encoded.shape} (most compressed)")
    return encoded

def decoder_block_no_skip(encoded, target_shape, level):
    """
    Decoder WITHOUT skip connections (information loss!)
    Args:
        encoded: Compressed representation
        target_shape: Shape to upsample to
        level: Decoder depth level
    Returns:
        upsampled: Reconstructed representation (lossy)
    """
    # Upsample by 2x (bilinear interpolation)
    freq_bins, time_frames = target_shape

    # Simple repeat upsampling
    upsampled = np.repeat(np.repeat(encoded, 2, axis=0), 2, axis=1)

    # Crop/pad to exact target shape
    if upsampled.shape[0] > freq_bins:
        upsampled = upsampled[:freq_bins, :]
    if upsampled.shape[1] > time_frames:
        upsampled = upsampled[:, :time_frames]
    if upsampled.shape[0] < freq_bins:
        upsampled = np.pad(upsampled, ((0, freq_bins - upsampled.shape[0]), (0, 0)), mode='reflect')
    if upsampled.shape[1] < time_frames:
        upsampled = np.pad(upsampled, ((0, 0), (0, time_frames - upsampled.shape[1])), mode='reflect')

    print(f"  Decoder L{level} (NO skip): {encoded.shape} -> {upsampled.shape}")

    return upsampled

def decoder_block_with_skip(encoded, skip_features, level):
    """
    Decoder WITH skip connections (details preserved!)
    Args:
        encoded: Compressed representation from lower level
        skip_features: High-res features from encoder at same level
        level: Decoder depth level
    Returns:
        upsampled: Reconstructed representation (with fine details)
    """
    # Upsample by 2x
    freq_bins, time_frames = skip_features.shape

    # Simple repeat upsampling
    upsampled = np.repeat(np.repeat(encoded, 2, axis=0), 2, axis=1)

    # Crop/pad to match skip features shape
    if upsampled.shape[0] > freq_bins:
        upsampled = upsampled[:freq_bins, :]
    if upsampled.shape[1] > time_frames:
        upsampled = upsampled[:, :time_frames]
    if upsampled.shape[0] < freq_bins:
        upsampled = np.pad(upsampled, ((0, freq_bins - upsampled.shape[0]), (0, 0)), mode='reflect')
    if upsampled.shape[1] < time_frames:
        upsampled = np.pad(upsampled, ((0, 0), (0, time_frames - upsampled.shape[1])), mode='reflect')

    # SKIP CONNECTION: Concatenate/add high-res features
    # This is the key difference! We recover lost details
    combined = (upsampled + skip_features) / 2  # Simple averaging

    print(f"  Decoder L{level} (WITH skip): {encoded.shape} + skip -> {combined.shape}")

    return combined

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

# Normalize to [0, 1] for processing
magnitude_norm = magnitude / (np.max(magnitude) + 1e-8)

# ============================================
# PATH 1: NO SKIP CONNECTIONS
# ============================================

print("\n" + "="*70)
print("PATH 1: WITHOUT SKIP CONNECTIONS")
print("="*70)

# Encoder path
print("\n[ENCODER PATH]")
enc1, skip1 = encoder_block(magnitude_norm, 1)
enc2, skip2 = encoder_block(enc1, 2)
enc3, skip3 = encoder_block(enc2, 3)
enc4, skip4 = encoder_block(enc3, 4)

# Bottleneck
print("\n[BOTTLENECK]")
bottleneck_repr = bottleneck(enc4)

# Decoder path WITHOUT skips
print("\n[DECODER PATH - NO SKIPS]")
dec4_no_skip = decoder_block_no_skip(bottleneck_repr, enc3.shape, 4)
dec3_no_skip = decoder_block_no_skip(dec4_no_skip, enc2.shape, 3)
dec2_no_skip = decoder_block_no_skip(dec3_no_skip, enc1.shape, 2)
dec1_no_skip = decoder_block_no_skip(dec2_no_skip, magnitude_norm.shape, 1)

print(f"\n  Final reconstruction (no skip): {dec1_no_skip.shape}")

# ============================================
# PATH 2: WITH SKIP CONNECTIONS
# ============================================

print("\n" + "="*70)
print("PATH 2: WITH SKIP CONNECTIONS")
print("="*70)

# Same encoder path (we already have skip features saved)
print("\n[ENCODER PATH - Using same features]")
print(f"  Skip connections stored at each level")

# Bottleneck (same)
print("\n[BOTTLENECK - Same]")

# Decoder path WITH skips
print("\n[DECODER PATH - WITH SKIPS]")
dec4_with_skip = decoder_block_with_skip(bottleneck_repr, skip4, 4)
dec3_with_skip = decoder_block_with_skip(dec4_with_skip, skip3, 3)
dec2_with_skip = decoder_block_with_skip(dec3_with_skip, skip2, 2)
dec1_with_skip = decoder_block_with_skip(dec2_with_skip, skip1, 1)

print(f"\n  Final reconstruction (with skip): {dec1_with_skip.shape}")

# ============================================
# RECONSTRUCTION & COMPARISON
# ============================================

print("\n" + "="*70)
print("AUDIO RECONSTRUCTION")
print("="*70)

# Denormalize
magnitude_no_skip = dec1_no_skip * np.max(magnitude)
magnitude_with_skip = dec1_with_skip * np.max(magnitude)

# Reconstruct complex STFT
stft_no_skip = magnitude_no_skip * np.exp(1j * phase)
stft_with_skip = magnitude_with_skip * np.exp(1j * phase)

# Inverse STFT
print("\n  Converting back to audio...")
audio_no_skip = librosa.istft(stft_no_skip, hop_length=CONFIG['hop_length'], n_fft=CONFIG['n_fft'])
audio_with_skip = librosa.istft(stft_with_skip, hop_length=CONFIG['hop_length'], n_fft=CONFIG['n_fft'])

# Normalize
audio_no_skip = audio_no_skip / (np.max(np.abs(audio_no_skip)) + 1e-8)
audio_with_skip = audio_with_skip / (np.max(np.abs(audio_with_skip)) + 1e-8)
audio_original = audio / (np.max(np.abs(audio)) + 1e-8)

print("  ✓ Audio reconstructed for both paths")

# Save outputs
print("\n[SAVING RESULTS]")
sf.write(f"{CONFIG['output_dir']}/demo1_original.wav", audio_original, sr)
sf.write(f"{CONFIG['output_dir']}/demo1_no_skip.wav", audio_no_skip, sr)
sf.write(f"{CONFIG['output_dir']}/demo1_with_skip.wav", audio_with_skip, sr)
print(f"  ✓ {CONFIG['output_dir']}/demo1_original.wav")
print(f"  ✓ {CONFIG['output_dir']}/demo1_no_skip.wav")
print(f"  ✓ {CONFIG['output_dir']}/demo1_with_skip.wav")

# ============================================
# VISUALIZATION
# ============================================

print("\n[CREATING VISUALIZATIONS]")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Row 1: Original
axes[0, 0].imshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
axes[0, 0].set_title('Original Vocal Spectrogram')
axes[0, 0].set_ylabel('Frequency (bins)')

axes[0, 1].plot(audio_original[:2000])
axes[0, 1].set_title('Original Waveform (2000 samples)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_ylim([-1.1, 1.1])
axes[0, 1].grid(True, alpha=0.3)

# Row 2: Without skip connections
axes[1, 0].imshow(librosa.amplitude_to_db(magnitude_no_skip, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
axes[1, 0].set_title('Reconstructed WITHOUT Skip Connections')
axes[1, 0].set_ylabel('Frequency (bins)')

axes[1, 1].plot(audio_no_skip[:2000])
axes[1, 1].set_title('Waveform WITHOUT Skips (lossy)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].set_ylim([-1.1, 1.1])
axes[1, 1].grid(True, alpha=0.3)

# Row 3: With skip connections
axes[2, 0].imshow(librosa.amplitude_to_db(magnitude_with_skip, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
axes[2, 0].set_title('Reconstructed WITH Skip Connections')
axes[2, 0].set_ylabel('Frequency (bins)')
axes[2, 0].set_xlabel('Time (frames)')

axes[2, 1].plot(audio_with_skip[:2000])
axes[2, 1].set_title('Waveform WITH Skips (details preserved)')
axes[2, 1].set_ylabel('Amplitude')
axes[2, 1].set_xlabel('Samples')
axes[2, 1].set_ylim([-1.1, 1.1])
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['figures_dir']}/demo1_skip_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {CONFIG['figures_dir']}/demo1_skip_comparison.png")

# Compute quality metrics
print("\n" + "="*70)
print("QUALITY METRICS")
print("="*70)

mse_no_skip = np.mean((magnitude - magnitude_no_skip)**2)
mse_with_skip = np.mean((magnitude - magnitude_with_skip)**2)

print(f"\nMean Squared Error (lower is better):")
print(f"  WITHOUT skip connections: {mse_no_skip:.6f}")
print(f"  WITH skip connections:    {mse_with_skip:.6f}")
print(f"  Improvement: {((mse_no_skip - mse_with_skip) / mse_no_skip * 100):.1f}%")

# Frequency range preservation
high_freq_energy_orig = np.sum(magnitude[magnitude.shape[0]//2:, :]**2)
high_freq_energy_no_skip = np.sum(magnitude_no_skip[magnitude.shape[0]//2:, :]**2)
high_freq_energy_with_skip = np.sum(magnitude_with_skip[magnitude.shape[0]//2:, :]**2)

print(f"\nHigh-Frequency Energy Preservation:")
print(f"  Original:      {high_freq_energy_orig:.1f}")
print(f"  WITHOUT skip:  {high_freq_energy_no_skip:.1f} ({high_freq_energy_no_skip/high_freq_energy_orig*100:.1f}%)")
print(f"  WITH skip:     {high_freq_energy_with_skip:.1f} ({high_freq_energy_with_skip/high_freq_energy_orig*100:.1f}%)")

print("\n" + "="*70)
print("✓ DEMO 1 COMPLETE")
print("="*70)
print("\nKey Insight: Skip connections preserve fine details lost in bottleneck!")
print("\nListen to:")
print("  1. demo1_no_skip.wav   - Muffled, details lost")
print("  2. demo1_with_skip.wav - Crisp, details preserved")
print("\nThis is WHY U-Net works for image/audio tasks requiring fine detail.")
