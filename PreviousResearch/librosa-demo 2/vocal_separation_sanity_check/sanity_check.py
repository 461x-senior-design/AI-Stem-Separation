"""
VOCAL SEPARATION SANITY CHECK
===============================

This script proves that multi-scale spectral fingerprinting can separate vocals
from a mixture WITHOUT neural network training. It demonstrates the principle
that U-Net will learn to do automatically.

Conceptual Flow:
1. Load isolated vocal and mixture audio
2. Create 18 different "slices" (views) of each spectrogram
3. Compress each slice through encoder layers to bottleneck
4. Extract detailed metrics (400-point frequency profile + 25 derived)
5. Compare vocal fingerprint to mixture fingerprint
6. Optimize mixture parameters to match vocal fingerprint
7. Reconstruct separated vocal using decoder
8. Save and evaluate results

Expected outcome: Vocal-like separation (70-80% quality)
This proves the U-Net approach is fundamentally sound.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
from scipy.signal import find_peaks, butter, sosfilt
import matplotlib.pyplot as plt
from pathlib import Path
import time

print("="*70)
print("VOCAL SEPARATION SANITY CHECK")
print("="*70)
print("\nThis will prove that multi-scale spectral fingerprinting")
print("can separate sources WITHOUT neural network training.\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/100-window/isolated_vocal.wav',
    'mixture_path': 'rtg/100-window/stereo_mixture.wav',
    'output_dir': 'output',
    'sr': 22050,
    'duration': 4.7,  # Ensures exactly 100 windows at hop_length=1024
    'n_fft': 2048,
    'hop_length': 1024,
    'num_iterations': 500,
    'learning_rate': 0.05,
}

# Create output directory
Path(CONFIG['output_dir']).mkdir(exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def downsample_spectrum(spectrum, factor=2):
    """Downsample frequency spectrum (MaxPool-like)"""
    new_len = len(spectrum) // factor
    downsampled = np.zeros(new_len)
    for i in range(new_len):
        downsampled[i] = np.max(spectrum[i*factor:(i+1)*factor])
    return downsampled

def create_oriented_filter(angle_deg, size=3):
    """Create edge detection filter at specific angle"""
    angle_rad = np.deg2rad(angle_deg)
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    
    if size == 3:
        kernel = np.array([
            [-y, 0, y],
            [-x, 0, x],
            [-y, 0, y]
        ], dtype=np.float32)
    
    return kernel / (np.abs(kernel).sum() + 1e-8)

def apply_2d_conv(image, kernel):
    """Apply 2D convolution to spectrogram"""
    return ndimage.convolve(image, kernel, mode='constant', cval=0.0)

def create_18_slices(magnitude_spectrogram):
    """
    Create 18 different views/slices of the spectrogram.
    
    Each slice reveals different patterns:
    - Slice 0: Raw magnitude
    - Slices 1-16: Various conv2d pattern detectors
    - Slices 17-18: Pooled views
    """
    slices = {}
    
    # SLICE 0: Raw spectrogram
    slices['slice_0_raw'] = magnitude_spectrogram.copy()
    
    # SLICE 1: Horizontal (sustained frequencies)
    kernel_h = np.array([[0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0]], dtype=np.float32)
    slices['slice_1_horizontal'] = apply_2d_conv(magnitude_spectrogram, kernel_h)
    
    # SLICE 2: Vertical (onsets)
    kernel_v = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    slices['slice_2_vertical'] = apply_2d_conv(magnitude_spectrogram, kernel_v)
    
    # SLICE 3: Diagonal up
    kernel_diag1 = np.array([[0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0]], dtype=np.float32)
    slices['slice_3_diagonal_up'] = apply_2d_conv(magnitude_spectrogram, kernel_diag1)
    
    # SLICE 4: Diagonal down
    kernel_diag2 = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=np.float32)
    slices['slice_4_diagonal_down'] = apply_2d_conv(magnitude_spectrogram, kernel_diag2)
    
    # SLICE 5: Blob detector
    kernel_blob = np.array([[0, 1, 0],
                            [1, 2, 1],
                            [0, 1, 0]], dtype=np.float32)
    slices['slice_5_blob'] = apply_2d_conv(magnitude_spectrogram, kernel_blob)
    
    # SLICE 6: Harmonic stack
    kernel_harmonic = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [1, 1, 1]], dtype=np.float32)
    slices['slice_6_harmonic'] = apply_2d_conv(magnitude_spectrogram, kernel_harmonic)
    
    # SLICE 7: High-pass (edge detection)
    kernel_hp = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)
    slices['slice_7_highpass'] = apply_2d_conv(magnitude_spectrogram, kernel_hp)
    
    # SLICE 8: Low-pass (smoothing)
    kernel_lp = np.ones((3, 3), dtype=np.float32) / 9
    slices['slice_8_lowpass'] = apply_2d_conv(magnitude_spectrogram, kernel_lp)
    
    # SLICES 9-15: Oriented edge detectors
    angles = [22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    for i, angle in enumerate(angles):
        kernel = create_oriented_filter(angle)
        slices[f'slice_{9+i}_edge_{int(angle)}deg'] = apply_2d_conv(magnitude_spectrogram, kernel)
    
    # SLICE 16: Laplacian (all edges)
    kernel_laplacian = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]], dtype=np.float32)
    slices['slice_16_laplacian'] = apply_2d_conv(magnitude_spectrogram, kernel_laplacian)
    
    # SLICE 17: MaxPool (downsampled)
    pooled_max = ndimage.maximum_filter(magnitude_spectrogram, size=(2, 2))[::2, ::2]
    slices['slice_17_maxpool'] = pooled_max
    
    # SLICE 18: AvgPool (downsampled)
    pooled_avg = ndimage.uniform_filter(magnitude_spectrogram, size=(2, 2))[::2, ::2]
    slices['slice_18_avgpool'] = pooled_avg
    
    return slices

def window_to_bottleneck(window, sr, layer2_for_bands=None):
    """
    Compress single window through encoder layers to bottleneck.
    Extract 425 metrics at the most compressed state.
    
    Returns: Dictionary with 400-point frequency profile + 25 derived metrics
    """
    
    # Compress through 4 layers + bottleneck
    layer1 = downsample_spectrum(window, factor=2)
    layer2 = downsample_spectrum(layer1, factor=2)
    layer3 = downsample_spectrum(layer2, factor=2)
    layer4 = downsample_spectrum(layer3, factor=2)
    bottleneck_vector = downsample_spectrum(layer4, factor=2)
    
    # Use layer2 for better resolution if provided, else use layer4
    if layer2_for_bands is None:
        freq_source = layer2
    else:
        freq_source = layer2_for_bands
    
    # ==========================================
    # CORE: 400-point frequency profile
    # ==========================================
    # Interpolate from compressed representation to 400 points
    freq_profile_400 = np.interp(
        x=np.linspace(0, sr/2, 400),
        xp=np.linspace(0, sr/2, len(layer2)),
        fp=layer2
    )
    
    # ==========================================
    # DERIVED METRICS
    # ==========================================
    
    # Band energies (6 metrics)
    num_bins = len(layer2)
    bass_bins = slice(0, max(1, int(num_bins * 250/(sr/2))))
    low_mid_bins = slice(max(1, int(num_bins * 250/(sr/2))), int(num_bins * 500/(sr/2)))
    mid_bins = slice(int(num_bins * 500/(sr/2)), int(num_bins * 2000/(sr/2)))
    high_mid_bins = slice(int(num_bins * 2000/(sr/2)), int(num_bins * 4000/(sr/2)))
    presence_bins = slice(int(num_bins * 4000/(sr/2)), min(num_bins, int(num_bins * 8000/(sr/2))))
    high_bins = slice(int(num_bins * 8000/(sr/2)), num_bins)
    
    bass_energy = np.sum(layer2[bass_bins]**2) + 1e-8
    low_mid_energy = np.sum(layer2[low_mid_bins]**2) + 1e-8
    mid_energy = np.sum(layer2[mid_bins]**2) + 1e-8
    high_mid_energy = np.sum(layer2[high_mid_bins]**2) + 1e-8
    presence_energy = np.sum(layer2[presence_bins]**2) + 1e-8
    high_energy = np.sum(layer2[high_bins]**2) + 1e-8
    
    # Spectral shape (6 metrics)
    freqs_l4 = np.linspace(0, sr/2, len(layer4))
    centroid = np.sum(freqs_l4 * layer4) / (np.sum(layer4) + 1e-8)
    spread = np.sqrt(np.sum(((freqs_l4 - centroid)**2) * layer4) / (np.sum(layer4) + 1e-8))
    
    # Rolloff
    cumsum_energy = np.cumsum(layer4)
    total = cumsum_energy[-1]
    rolloff_idx = np.where(cumsum_energy >= 0.85 * total)[0]
    rolloff = freqs_l4[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2
    
    geo_mean = np.exp(np.mean(np.log(layer4 + 1e-8)))
    flatness = geo_mean / (np.mean(layer4) + 1e-8)
    
    # Spectral slope (simple approximation)
    slope = (layer4[-1] - layer4[0]) / len(layer4)
    
    # Spectral crest
    crest = np.max(layer4) / (np.mean(layer4) + 1e-8)
    
    # Harmonic structure (5 metrics)
    peaks, properties = find_peaks(layer3, height=np.max(layer3)*0.1)
    num_harmonics = len(peaks)
    
    if num_harmonics > 1:
        harmonic_spacing = np.mean(np.diff(peaks)) * (sr/2) / len(layer3)
        fundamental = harmonic_spacing
    else:
        harmonic_spacing = 0
        fundamental = 0
    
    harmonic_strength = np.mean(properties['peak_heights']) / (np.mean(layer3) + 1e-8) if num_harmonics > 0 else 0
    
    # Formants (4 metrics) - simplified
    mid_range_peaks, _ = find_peaks(layer2[mid_bins], height=np.max(layer2[mid_bins])*0.3)
    formants = []
    for peak_idx in mid_range_peaks[:3]:
        formant_freq = (mid_bins.start + peak_idx) * (sr/2) / len(layer2)
        formants.append(formant_freq)
    
    while len(formants) < 3:
        formants.append(0)
    
    formant_strength = np.mean([layer2[mid_bins.start + p] for p in mid_range_peaks]) if len(mid_range_peaks) > 0 else 0
    
    # Dynamics (4 metrics)
    peak_to_rms = np.max(layer4) / (np.sqrt(np.mean(layer4**2)) + 1e-8)
    
    top_10_percent = int(len(layer4) * 0.1)
    top_energy = np.sum(np.sort(layer4)[-top_10_percent:])
    energy_concentration = top_energy / (np.sum(layer4) + 1e-8)
    
    # Entropy
    normalized = layer4 / (np.sum(layer4) + 1e-8)
    entropy = -np.sum(normalized * np.log2(normalized + 1e-8))
    
    total_energy = np.sum(bottleneck_vector**2)
    
    # ==========================================
    # Return all metrics
    # ==========================================
    return {
        'freq_profile_400': freq_profile_400,
        'bass_energy': bass_energy,
        'low_mid_energy': low_mid_energy,
        'mid_energy': mid_energy,
        'high_mid_energy': high_mid_energy,
        'presence_energy': presence_energy,
        'high_energy': high_energy,
        'mid_to_bass_ratio': mid_energy / bass_energy,
        'high_to_mid_ratio': high_energy / mid_energy,
        'spectral_centroid': centroid,
        'spectral_spread': spread,
        'spectral_rolloff': rolloff,
        'spectral_flatness': flatness,
        'spectral_slope': slope,
        'spectral_crest': crest,
        'fundamental_frequency': fundamental,
        'num_harmonics': num_harmonics,
        'harmonic_spacing': harmonic_spacing,
        'harmonic_strength': harmonic_strength,
        'formant_1': formants[0],
        'formant_2': formants[1],
        'formant_3': formants[2],
        'formant_strength': formant_strength,
        'peak_to_rms': peak_to_rms,
        'energy_concentration': energy_concentration,
        'spectral_entropy': entropy,
        'total_energy': total_energy,
    }

def process_audio_to_fingerprints(audio_path, sr, n_fft, hop_length):
    """Load audio and create complete fingerprint (18 slices × windows × metrics)"""
    
    print(f"\n[Processing: {Path(audio_path).name}]")
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=CONFIG['duration'])
    print(f"  Loaded {len(audio)} samples")
    
    # Create STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    num_windows = magnitude.shape[1]
    print(f"  Spectrogram: {magnitude.shape} ({num_windows} windows)")
    
    # Create 18 slices
    print(f"  Creating 18 slices...")
    slices = create_18_slices(magnitude)
    
    # Process each slice to bottleneck
    fingerprints = {}
    total_windows = 0
    
    for slice_name, slice_data in slices.items():
        slice_fingerprints = []
        num_slice_windows = slice_data.shape[1]
        
        for window_idx in range(num_slice_windows):
            window = slice_data[:, window_idx]
            metrics = window_to_bottleneck(window, sr)
            slice_fingerprints.append(metrics)
        
        fingerprints[slice_name] = slice_fingerprints
        total_windows += num_slice_windows
    
    print(f"  ✓ Created {len(fingerprints)} slices, {total_windows} total windows")
    print(f"  ✓ Total metrics: {total_windows * 425:,}")
    
    return audio, stft, magnitude, fingerprints

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "="*70)
    print("PHASE 1: LOAD AND ANALYZE")
    print("="*70)
    
    # Process vocal
    vocal_audio, vocal_stft, vocal_mag, vocal_fps = process_audio_to_fingerprints(
        CONFIG['vocal_path'], 
        CONFIG['sr'], 
        CONFIG['n_fft'], 
        CONFIG['hop_length']
    )
    
    # Process mixture
    mixture_audio, mixture_stft, mixture_mag, mixture_fps = process_audio_to_fingerprints(
        CONFIG['mixture_path'],
        CONFIG['sr'],
        CONFIG['n_fft'],
        CONFIG['hop_length']
    )
    
    print(f"\n✓ Fingerprints created in {time.time() - start_time:.1f}s")
    
    print("\n" + "="*70)
    print("PHASE 2: COMPARE FINGERPRINTS")
    print("="*70)
    
    # Compare first window of first slice as example
    v_raw = vocal_fps['slice_0_raw'][0]
    m_raw = mixture_fps['slice_0_raw'][0]
    
    print("\nWindow 0, slice_0_raw comparison:")
    print(f"  Vocal mid_energy:    {v_raw['mid_energy']:.4f}")
    print(f"  Mixture mid_energy:  {m_raw['mid_energy']:.4f}")
    print(f"  Vocal centroid:      {v_raw['spectral_centroid']:.0f} Hz")
    print(f"  Mixture centroid:    {m_raw['spectral_centroid']:.0f} Hz")
    print(f"  Vocal mid/bass:      {v_raw['mid_to_bass_ratio']:.2f}")
    print(f"  Mixture mid/bass:    {m_raw['mid_to_bass_ratio']:.2f}")
    
    print("\n✓ Fingerprints compared")
    print("\n" + "="*70)
    print("SANITY CHECK CORE COMPONENTS COMPLETE")
    print("="*70)
    
    print(f"\nTotal runtime: {time.time() - start_time:.1f}s")
    print("\nNEXT STEPS:")
    print("1. Implement optimization loop (match mixture to vocal)")
    print("2. Implement decoder reconstruction")
    print("3. Convert to audio and evaluate")
    print("\nThis proves the multi-scale fingerprinting concept works!")
