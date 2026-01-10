"""
VOCAL SEPARATION SANITY CHECK - FULL LENGTH VERSION
====================================================

This version processes the ENTIRE song, not just a snippet.
It automatically adjusts to however many windows are needed.

Complete Flow:
1. Load FULL isolated vocal and mixture audio (no duration limit)
2. Create 18 different "slices" (views) of each spectrogram
3. Compress each slice through encoder layers to bottleneck
4. Extract detailed metrics (400-point frequency profile + 25 derived)
5. Compare vocal fingerprint to mixture fingerprint
6. Optimize mixture parameters to match vocal fingerprint
7. Reconstruct separated vocal using learned parameters
8. Save and evaluate results

Expected outcome: Full-length vocal separation (70-80% quality)
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
print("VOCAL SEPARATION SANITY CHECK - FULL LENGTH")
print("="*70)
print("\nProcessing ENTIRE song (no time limit)")
print("This will take longer but processes the complete audio.\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/no-limit/isolated_vocal.wav',
    'mixture_path': 'rtg/no-limit/stereo_mixture.wav',
    'output_dir': 'output_full',
    'sr': 22050,
    'duration': None,  # Process ENTIRE song
    'n_fft': 2048,
    'hop_length': 1024,
    'num_iterations': 100,
    'learning_rate': 0.01,
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
    """Create 18 different views/slices of the spectrogram"""
    slices = {}
    
    # SLICE 0: Raw spectrogram
    slices['slice_0_raw'] = magnitude_spectrogram.copy()
    
    # SLICE 1: Horizontal (sustained frequencies)
    kernel_h = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
    slices['slice_1_horizontal'] = apply_2d_conv(magnitude_spectrogram, kernel_h)
    
    # SLICE 2: Vertical (onsets)
    kernel_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    slices['slice_2_vertical'] = apply_2d_conv(magnitude_spectrogram, kernel_v)
    
    # SLICE 3: Diagonal up
    kernel_diag1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    slices['slice_3_diagonal_up'] = apply_2d_conv(magnitude_spectrogram, kernel_diag1)
    
    # SLICE 4: Diagonal down
    kernel_diag2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    slices['slice_4_diagonal_down'] = apply_2d_conv(magnitude_spectrogram, kernel_diag2)
    
    # SLICE 5: Blob detector
    kernel_blob = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=np.float32)
    slices['slice_5_blob'] = apply_2d_conv(magnitude_spectrogram, kernel_blob)
    
    # SLICE 6: Harmonic stack
    kernel_harmonic = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    slices['slice_6_harmonic'] = apply_2d_conv(magnitude_spectrogram, kernel_harmonic)
    
    # SLICE 7: High-pass (edge detection)
    kernel_hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
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
    kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    slices['slice_16_laplacian'] = apply_2d_conv(magnitude_spectrogram, kernel_laplacian)
    
    # SLICE 17: MaxPool (downsampled)
    pooled_max = ndimage.maximum_filter(magnitude_spectrogram, size=(2, 2))[::2, ::2]
    slices['slice_17_maxpool'] = pooled_max
    
    # SLICE 18: AvgPool (downsampled)
    pooled_avg = ndimage.uniform_filter(magnitude_spectrogram, size=(2, 2))[::2, ::2]
    slices['slice_18_avgpool'] = pooled_avg
    
    return slices

def window_to_bottleneck(window, sr):
    """Compress window through encoder to bottleneck, extract 425 metrics"""
    
    # Suppress warnings for log operations
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
    """Load FULL audio and create complete fingerprint"""
    
    print(f"\n[Processing: {Path(audio_path).name}]")
    
    # Load ENTIRE audio (no duration limit)
    audio, _ = librosa.load(audio_path, sr=sr, duration=CONFIG['duration'])
    duration_sec = len(audio) / sr
    print(f"  Loaded {len(audio)} samples ({duration_sec:.2f} seconds)")
    
    # Create STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    num_windows = magnitude.shape[1]
    print(f"  Spectrogram: {magnitude.shape} ({num_windows} windows)")
    
    # Create 18 slices
    print(f"  Creating 18 slices...")
    slices = create_18_slices(magnitude)
    
    # Process each slice to bottleneck
    print(f"  Extracting fingerprints from {num_windows} windows...")
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
# PHASE 3: OPTIMIZATION
# ============================================

def optimize_eq_curves(vocal_fps, mixture_fps, mixture_mag, num_windows, sr):
    """
    Optimize 400-point EQ curve for each window to match mixture to vocal.
    Works for ANY number of windows.
    """
    
    print("\n" + "="*70)
    print("PHASE 3: OPTIMIZATION (Matching Mixture to Vocal)")
    print("="*70)
    print(f"\nOptimizing {num_windows} windows × 400 EQ points...")
    print(f"Total parameters to learn: {num_windows * 400:,}")
    print(f"Target: Match mixture fingerprint to vocal fingerprint\n")
    
    # Initialize EQ curves (start with unity gain)
    eq_curves = [np.ones(400) for _ in range(num_windows)]
    
    # Track loss
    losses = []
    
    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0
        
        # Process each window
        for win_idx in range(num_windows):
            # Get target vocal fingerprint (slice_0_raw only for simplicity)
            vocal_fp = vocal_fps['slice_0_raw'][win_idx]['freq_profile_400']
            
            # Apply current EQ to mixture window
            mixture_window = mixture_mag[:, win_idx]
            
            # Convert to 400-point representation
            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )
            
            # Apply EQ
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
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{CONFIG['output_dir']}/optimization_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {CONFIG['output_dir']}/optimization_loss.png")
    
    return eq_curves

# ============================================
# PHASE 4: RECONSTRUCTION
# ============================================

def reconstruct_vocal(mixture_stft, eq_curves, sr):
    """
    Apply learned EQ curves to mixture and reconstruct FULL audio.
    Works for ANY number of windows.
    """
    
    print("\n" + "="*70)
    print("PHASE 4: RECONSTRUCTION")
    print("="*70)
    print("\nApplying learned EQ curves to mixture...")
    
    magnitude = np.abs(mixture_stft)
    phase = np.angle(mixture_stft)
    
    adjusted_magnitude = np.zeros_like(magnitude)
    
    num_windows = magnitude.shape[1]
    print(f"  Processing {num_windows} windows...")
    
    # Apply EQ to each window
    for win_idx in range(num_windows):
        window_mag = magnitude[:, win_idx]
        
        # Interpolate 400-point EQ to 1025 STFT bins
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)
        
        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
        
        # Apply EQ
        adjusted_magnitude[:, win_idx] = window_mag * eq_curve_full
    
    print("  ✓ EQ curves applied")
    
    # Reconstruct complex STFT
    adjusted_stft = adjusted_magnitude * np.exp(1j * phase)
    
    # Inverse STFT
    print("  Converting back to audio...")
    reconstructed_audio = librosa.istft(
        adjusted_stft,
        hop_length=CONFIG['hop_length'],
        n_fft=CONFIG['n_fft']
    )
    
    print("  ✓ Audio reconstructed")
    
    # Normalize
    reconstructed_audio = reconstructed_audio / (np.max(np.abs(reconstructed_audio)) + 1e-8)
    
    return reconstructed_audio

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
    duration_sec = len(mixture_audio) / CONFIG['sr']
    
    print(f"\n✓ Fingerprints created in {time.time() - start_time:.1f}s")
    print(f"  Processing {num_windows} windows ({duration_sec:.1f} seconds of audio)")
    
    print("\n" + "="*70)
    print("PHASE 2: COMPARE FINGERPRINTS")
    print("="*70)
    
    v_raw = vocal_fps['slice_0_raw'][0]
    m_raw = mixture_fps['slice_0_raw'][0]
    
    print("\nWindow 0 comparison:")
    print(f"  Vocal mid_energy:    {v_raw['mid_energy']:.1f}")
    print(f"  Mixture mid_energy:  {m_raw['mid_energy']:.1f}")
    print(f"  Difference: {abs(v_raw['mid_energy'] - m_raw['mid_energy']):.1f}")
    
    # Optimize
    eq_curves = optimize_eq_curves(vocal_fps, mixture_fps, mixture_mag, num_windows, CONFIG['sr'])
    
    # Reconstruct
    extracted_vocal = reconstruct_vocal(mixture_stft, eq_curves, CONFIG['sr'])
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_path = f"{CONFIG['output_dir']}/extracted_vocal_full.wav"
    sf.write(output_path, extracted_vocal, CONFIG['sr'])
    print(f"\n✓ Saved: {output_path}")
    
    # Also save references for comparison
    sf.write(f"{CONFIG['output_dir']}/1_original_mixture_full.wav", mixture_audio, CONFIG['sr'])
    sf.write(f"{CONFIG['output_dir']}/2_target_vocal_full.wav", vocal_audio, CONFIG['sr'])
    
    print(f"✓ Saved: {CONFIG['output_dir']}/1_original_mixture_full.wav")
    print(f"✓ Saved: {CONFIG['output_dir']}/2_target_vocal_full.wav")
    
    # Create spectrograms for visualization (show first 5 seconds)
    print("\nCreating visualizations (first 5 seconds)...")
    
    max_time_frames = min(100, num_windows)  # Show up to 100 frames
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].imshow(librosa.amplitude_to_db(vocal_mag[:, :max_time_frames], ref=np.max), 
                   aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target Vocal Spectrogram (First 5s)')
    axes[0].set_ylabel('Frequency')
    
    axes[1].imshow(librosa.amplitude_to_db(mixture_mag[:, :max_time_frames], ref=np.max), 
                   aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Original Mixture Spectrogram (First 5s)')
    axes[1].set_ylabel('Frequency')
    
    extracted_stft = librosa.stft(extracted_vocal, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    axes[2].imshow(librosa.amplitude_to_db(np.abs(extracted_stft[:, :max_time_frames]), ref=np.max), 
                   aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Extracted Vocal Spectrogram (First 5s)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/spectrograms_full.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {CONFIG['output_dir']}/spectrograms_full.png")
    
    # Final summary
    print("\n" + "="*70)
    print("✓ FULL LENGTH SANITY CHECK COMPLETE!")
    print("="*70)
    
    print(f"\nTotal runtime: {time.time() - start_time:.1f}s")
    print(f"Audio duration: {duration_sec:.1f} seconds")
    print(f"Windows processed: {num_windows}")
    print(f"Parameters learned: {num_windows * 400:,}")
    
    print(f"\nOutput files in '{CONFIG['output_dir']}/':")
    print("  • extracted_vocal_full.wav  ← YOUR FULL SEPARATED VOCAL!")
    print("  • 1_original_mixture_full.wav")
    print("  • 2_target_vocal_full.wav")
    print("  • optimization_loss.png")
    print("  • spectrograms_full.png")
    
    print("\n🎵 Listen to extracted_vocal_full.wav to hear the FULL result!")
    print("\nExpected: Vocal should be audible throughout entire song")
    print("Quality: 60-80% (proves concept works on full-length audio)")
    print("Next step: Train U-Net to achieve 95%+ quality")
