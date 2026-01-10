"""
VOCAL SEPARATION - 1 CONV2D + RAW SEQUENTIAL TEST
=================================================

Based on the working sanity_check_complete.py, but modified to:
1. First pass: Optimize using slice_1_horizontal (Conv2D filtered view)
2. Second pass: Optimize using slice_0_raw (direct magnitude view)

Theory: Adding one Conv2D perspective before raw polish should improve separation.
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
print("VOCAL SEPARATION - 1 CONV2D + RAW SEQUENTIAL TEST")
print("="*70)
print("\nTesting: slice_1_horizontal → slice_0_raw")
print("Expected: Better than raw-only version\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/100-window/isolated_vocal.wav',
    'mixture_path': 'rtg/100-window/stereo_mixture.wav',
    'output_dir': 'output/1conv_then_raw',
    'sr': 22050,
    'duration': 4.7,
    'n_fft': 2048,
    'hop_length': 1024,
    'num_iterations': 100,
    'learning_rate': 0.01,
}

Path(CONFIG['output_dir']).mkdir(exist_ok=True, parents=True)

# ============================================
# HELPER FUNCTIONS (from original)
# ============================================

def downsample_spectrum(spectrum, factor=2):
    """Downsample frequency spectrum (MaxPool-like)"""
    new_len = len(spectrum) // factor
    downsampled = np.zeros(new_len)
    for i in range(new_len):
        downsampled[i] = np.max(spectrum[i*factor:(i+1)*factor])
    return downsampled

def apply_2d_conv(image, kernel):
    """Apply 2D convolution to spectrogram"""
    return ndimage.convolve(image, kernel, mode='constant', cval=0.0)

def create_2_slices(magnitude_spectrogram):
    """Create only slice_0_raw and slice_1_horizontal"""
    slices = {}

    # SLICE 0: Raw spectrogram
    slices['slice_0_raw'] = magnitude_spectrogram.copy()

    # SLICE 1: Horizontal (sustained frequencies)
    kernel_h = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
    slices['slice_1_horizontal'] = apply_2d_conv(magnitude_spectrogram, kernel_h)

    return slices

def window_to_bottleneck(window, sr):
    """Compress window through encoder to bottleneck, extract 425 metrics"""

    with np.errstate(divide='ignore', invalid='ignore'):
        # Compress through layers
        layer1 = downsample_spectrum(window, factor=2)
        layer2 = downsample_spectrum(layer1, factor=2)
        layer3 = downsample_spectrum(layer2, factor=2)
        layer4 = downsample_spectrum(layer3, factor=2)
        bottleneck_vector = downsample_spectrum(layer4, factor=2)

        # Core: 400-point frequency profile
        freq_profile_400 = np.interp(
            x=np.linspace(0, sr/2, 400),
            xp=np.linspace(0, sr/2, len(layer2)),
            fp=layer2
        )

        # Band energies
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

        # Spectral shape
        freqs_l4 = np.linspace(0, sr/2, len(layer4))
        centroid = np.sum(freqs_l4 * layer4) / (np.sum(layer4) + 1e-8)
        spread = np.sqrt(np.sum(((freqs_l4 - centroid)**2) * layer4) / (np.sum(layer4) + 1e-8))

        cumsum_energy = np.cumsum(layer4)
        total = cumsum_energy[-1]
        rolloff_idx = np.where(cumsum_energy >= 0.85 * total)[0]
        rolloff = freqs_l4[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2

        geo_mean = np.exp(np.mean(np.log(layer4 + 1e-8)))
        flatness = geo_mean / (np.mean(layer4) + 1e-8)
        slope = (layer4[-1] - layer4[0]) / len(layer4)
        crest = np.max(layer4) / (np.mean(layer4) + 1e-8)

        # Harmonic structure
        peaks, properties = find_peaks(layer3, height=np.max(layer3)*0.1)
        num_harmonics = len(peaks)

        if num_harmonics > 1:
            harmonic_spacing = np.mean(np.diff(peaks)) * (sr/2) / len(layer3)
            fundamental = harmonic_spacing
        else:
            harmonic_spacing = 0
            fundamental = 0

        harmonic_strength = np.mean(properties['peak_heights']) / (np.mean(layer3) + 1e-8) if num_harmonics > 0 else 0

        # Formants
        mid_range_peaks, _ = find_peaks(layer2[mid_bins], height=np.max(layer2[mid_bins])*0.3)
        formants = []
        for peak_idx in mid_range_peaks[:3]:
            formant_freq = (mid_bins.start + peak_idx) * (sr/2) / len(layer2)
            formants.append(formant_freq)
        while len(formants) < 3:
            formants.append(0)

        formant_strength = np.mean([layer2[mid_bins.start + p] for p in mid_range_peaks]) if len(mid_range_peaks) > 0 else 0

        # Dynamics
        peak_to_rms = np.max(layer4) / (np.sqrt(np.mean(layer4**2)) + 1e-8)
        top_10_percent = int(len(layer4) * 0.1)
        top_energy = np.sum(np.sort(layer4)[-top_10_percent:])
        energy_concentration = top_energy / (np.sum(layer4) + 1e-8)

        normalized = layer4 / (np.sum(layer4) + 1e-8)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-8))
        total_energy = np.sum(bottleneck_vector**2)

        return {
            'freq_profile_400': np.nan_to_num(freq_profile_400, nan=0.0, posinf=0.0, neginf=0.0),
            'bass_energy': bass_energy,
            'low_mid_energy': low_mid_energy,
            'mid_energy': mid_energy,
            'high_mid_energy': high_mid_energy,
            'presence_energy': presence_energy,
            'high_energy': high_energy,
            'mid_to_bass_ratio': mid_energy / bass_energy,
            'high_to_mid_ratio': high_energy / mid_energy,
            'spectral_centroid': centroid,
            'spectral_spread': spread if not np.isnan(spread) else 0.0,
            'spectral_rolloff': rolloff,
            'spectral_flatness': flatness if not np.isnan(flatness) else 0.0,
            'spectral_slope': slope,
            'spectral_crest': crest,
            'fundamental_frequency': fundamental,
            'num_harmonics': num_harmonics,
            'harmonic_spacing': harmonic_spacing,
            'harmonic_strength': harmonic_strength if not np.isnan(harmonic_strength) else 0.0,
            'formant_1': formants[0],
            'formant_2': formants[1],
            'formant_3': formants[2],
            'formant_strength': formant_strength if not np.isnan(formant_strength) else 0.0,
            'peak_to_rms': peak_to_rms,
            'energy_concentration': energy_concentration,
            'spectral_entropy': entropy if not np.isnan(entropy) else 0.0,
            'total_energy': total_energy,
        }

def process_audio_to_fingerprints(audio_path, sr, n_fft, hop_length):
    """Load audio and create fingerprints for 2 slices"""

    print(f"\n[Processing: {Path(audio_path).name}]")

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=CONFIG['duration'])
    print(f"  Loaded {len(audio)} samples")

    # Create STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    num_windows = magnitude.shape[1]
    print(f"  Spectrogram: {magnitude.shape} ({num_windows} windows)")

    # Create 2 slices
    print(f"  Creating 2 slices (horizontal + raw)...")
    slices = create_2_slices(magnitude)

    # Process each slice to bottleneck
    fingerprints = {}

    for slice_name, slice_data in slices.items():
        slice_fingerprints = []
        num_slice_windows = slice_data.shape[1]

        for window_idx in range(num_slice_windows):
            window = slice_data[:, window_idx]
            metrics = window_to_bottleneck(window, sr)
            slice_fingerprints.append(metrics)

        fingerprints[slice_name] = slice_fingerprints

    print(f"  ✓ Created {len(fingerprints)} slices with fingerprints")

    return audio, stft, magnitude, fingerprints

# ============================================
# OPTIMIZATION (MODIFIED FOR SEQUENTIAL)
# ============================================

def optimize_one_slice(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Optimize EQ curves for ONE slice.
    If current_mag provided, use it as starting point (sequential refinement).
    Returns: refined magnitude after applying learned EQ
    """

    print(f"\n" + "="*70)
    print(f"OPTIMIZING: {slice_name}")
    print("="*70)
    print(f"\nOptimizing {num_windows} windows × 400 EQ points...")

    # Use current_mag if provided (sequential), otherwise start from mixture
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Initialize EQ curves (start with unity gain)
    eq_curves = [np.ones(400) for _ in range(num_windows)]

    # Track loss
    losses = []

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        # Process each window
        for win_idx in range(num_windows):
            # Get target vocal fingerprint for this slice
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']

            # Get current working magnitude window
            mixture_window = working_mag[:, win_idx]

            # Convert to 400-point representation
            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )

            # Apply EQ (multiplicative, like original)
            adjusted_fp = mixture_fp * eq_curves[win_idx]

            # Compute loss (mean squared error)
            loss = np.mean((adjusted_fp - vocal_fp)**2)
            total_loss += loss

            # Compute gradient
            gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp

            # Update EQ curve (gradient descent)
            eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient

            # Clip to reasonable range [0.1, 3.0]
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

        avg_loss = total_loss / num_windows
        losses.append(avg_loss)

        if iteration % 20 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {avg_loss:.6f}")

    print(f"\n✓ Optimization complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")

    # Apply learned EQ to working magnitude
    refined_mag = np.zeros_like(working_mag)

    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]

        # Interpolate 400-point EQ to full STFT bins
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)

        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])

        # Apply EQ (multiplicative)
        refined_mag[:, win_idx] = window_mag * eq_curve_full

    print(f"  ✓ EQ applied to magnitude")

    return refined_mag, losses

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

    num_windows = mixture_mag.shape[1]
    mixture_phase = np.angle(mixture_stft)

    print(f"\n✓ Fingerprints created in {time.time() - start_time:.1f}s")

    # ============================================
    # SEQUENTIAL OPTIMIZATION
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: SEQUENTIAL OPTIMIZATION")
    print("="*70)
    print("\nPass 1: slice_1_horizontal")
    print("Pass 2: slice_0_raw (polish)")

    # Pass 1: Optimize with horizontal slice
    refined_mag_pass1, losses_pass1 = optimize_one_slice(
        'slice_1_horizontal',
        vocal_fps,
        mixture_mag,
        num_windows,
        CONFIG['sr'],
        current_mag=None  # Start from original mixture
    )

    # Pass 2: Optimize with raw slice (using pass1 result as input)
    refined_mag_final, losses_pass2 = optimize_one_slice(
        'slice_0_raw',
        vocal_fps,
        mixture_mag,  # Still pass original for reference
        num_windows,
        CONFIG['sr'],
        current_mag=refined_mag_pass1  # Start from pass1 result
    )

    # ============================================
    # RECONSTRUCTION
    # ============================================

    print("\n" + "="*70)
    print("PHASE 3: RECONSTRUCTION")
    print("="*70)

    # Reconstruct audio using final refined magnitude
    refined_stft = refined_mag_final * np.exp(1j * mixture_phase)
    extracted_vocal = librosa.istft(
        refined_stft,
        hop_length=CONFIG['hop_length'],
        n_fft=CONFIG['n_fft']
    )

    # Normalize
    extracted_vocal = extracted_vocal / (np.max(np.abs(extracted_vocal)) + 1e-8)

    print("✓ Audio reconstructed")

    # ============================================
    # SAVE RESULTS
    # ============================================

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    output_path = f"{CONFIG['output_dir']}/extracted_vocal_1conv_then_raw.wav"
    sf.write(output_path, extracted_vocal, CONFIG['sr'])
    print(f"\n✓ Saved: {output_path}")

    # Save references
    sf.write(f"{CONFIG['output_dir']}/1_original_mixture.wav", mixture_audio, CONFIG['sr'])
    sf.write(f"{CONFIG['output_dir']}/2_target_vocal.wav", vocal_audio, CONFIG['sr'])

    print(f"✓ Saved: {CONFIG['output_dir']}/1_original_mixture.wav")
    print(f"✓ Saved: {CONFIG['output_dir']}/2_target_vocal.wav")

    # Create spectrograms
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].imshow(librosa.amplitude_to_db(vocal_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target Vocal')
    axes[0].set_ylabel('Frequency')

    axes[1].imshow(librosa.amplitude_to_db(mixture_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Original Mixture')
    axes[1].set_ylabel('Frequency')

    extracted_stft = librosa.stft(extracted_vocal, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    axes[2].imshow(librosa.amplitude_to_db(np.abs(extracted_stft), ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Extracted Vocal (1 Conv2D + Raw)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/spectrograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {CONFIG['output_dir']}/spectrograms.png")

    # Plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses_pass1)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Pass 1: slice_1_horizontal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses_pass2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Pass 2: slice_0_raw (polish)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {CONFIG['output_dir']}/loss_curves.png")

    # Final summary
    print("\n" + "="*70)
    print("✓ TEST COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {time.time() - start_time:.1f}s")
    print(f"\nOutput in '{CONFIG['output_dir']}/':")
    print("  • extracted_vocal_1conv_then_raw.wav ← TEST RESULT")
    print("  • 1_original_mixture.wav")
    print("  • 2_target_vocal.wav")
    print("  • spectrograms.png")
    print("  • loss_curves.png")

    print("\n🎵 Compare this to the original raw-only version!")
    print("Theory: This should sound better since we refined from 2 perspectives")
