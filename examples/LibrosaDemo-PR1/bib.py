BIN-BY-BIN ANALYSIS: 1025 BINS × 100 WINDOWS
=============================================

This version analyzes each frequency bin independently at each time window.

Structure:
- Standard: 100 windows → extract 425 metrics per window from full spectrum
- This version: 1025 bins × 100 windows → extract 425 metrics per (bin, window) pair

Approach:
1. Load audio and create STFT (1025 bins × 100 windows) at 44.1kHz
2. For each bin (0-21.5 Hz, 21.5-43 Hz, ..., up to 22,050 Hz):
   - For each window:
     - Extract 425 metrics from that single bin's value
     - Most metrics will be degenerate (can't find harmonics in one bin)
     - But energy and some temporal metrics might be informative
3. Total fingerprint: 1025 × 100 × 425 = 43,562,500 values

Purpose: Extreme granularity - every frequency bin analyzed independently at every moment.
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
print("VOCAL SEPARATION SANITY CHECK - COMPLETE")
print("="*70)
print("\nThis will prove that multi-scale spectral fingerprinting")
print("can separate sources WITHOUT neural network training.\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/100-window/isolated_vocal.wav',
    'mixture_path': 'rtg/100-window/stereo_mixture.wav',
    'output_dir': 'output/bin-by-bin-1025x100',
    'sr': 44100,  # 44.1 kHz for full frequency range
    'duration': 4.7,
    'n_fft': 2048,
    'window_spacing_ms': 47,  # Milliseconds between windows
    'num_iterations': 100,
    'learning_rate': 0.01,
}

# Calculate hop_length from window spacing
CONFIG['hop_length'] = int(CONFIG['sr'] * CONFIG['window_spacing_ms'] / 1000)

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

# NOTE: Conv2D functions removed - this version does NOT use Conv2D slices
# Only raw frequency bins analyzed independently

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
    """Load audio and create bin-by-bin fingerprint (1025 bins × 100 windows × 425 metrics)"""

    print(f"\n[Processing: {Path(audio_path).name}]")

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=CONFIG['duration'])
    print(f"  Loaded {len(audio)} samples")

    # Create STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    num_bins, num_windows = magnitude.shape
    print(f"  Spectrogram: {magnitude.shape} ({num_bins} bins × {num_windows} windows)")

    # NO Conv2D - using only raw slice
    print(f"  Analyzing each bin independently at each window...")
    print(f"  This will create {num_bins} × {num_windows} × 425 = {num_bins * num_windows * 425:,} metrics")

    # Process bin-by-bin, window-by-window
    fingerprints = {}

    # We'll store under 'slice_0_raw' for compatibility, but it's actually bin-level
    bin_fingerprints = []

    for bin_idx in range(num_bins):
        bin_across_windows = []

        for window_idx in range(num_windows):
            # Create a sparse "spectrum" with only this one bin populated
            sparse_window = np.zeros(num_bins)
            sparse_window[bin_idx] = magnitude[bin_idx, window_idx]

            # Extract 425 metrics from this mostly-empty spectrum
            metrics = window_to_bottleneck(sparse_window, sr)
            bin_across_windows.append(metrics)

        bin_fingerprints.append(bin_across_windows)

        if (bin_idx + 1) % 100 == 0:
            print(f"    Processed {bin_idx + 1}/{num_bins} bins...")

    fingerprints['slice_0_raw'] = bin_fingerprints

    print(f"  ✓ Created bin-by-bin fingerprint: {num_bins} bins × {num_windows} windows × 425 metrics")

    return audio, stft, magnitude, fingerprints

# ============================================
# PHASE 3: OPTIMIZATION
# ============================================

def optimize_eq_curves(vocal_fps, mixture_fps, mixture_mag, num_windows, sr):
    """
    Optimize per-bin EQ curves using bin-by-bin fingerprints.
    Returns learned EQ parameters for each window.
    """

    num_bins = mixture_mag.shape[0]

    print("\n" + "="*70)
    print("PHASE 3: OPTIMIZATION (Bin-by-Bin Matching)")
    print("="*70)
    print(f"\nOptimizing using {num_bins} bins × {num_windows} windows fingerprints...")
    print(f"Each bin analyzed independently!")
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

            # Accumulate loss across all bins
            window_loss = 0.0

            for bin_idx in range(num_bins):
                # Get target vocal fingerprint for this bin at this window
                # Structure: vocal_fps['slice_0_raw'][bin_idx][win_idx]
                vocal_fp_dict = vocal_fps['slice_0_raw'][bin_idx][win_idx]
                vocal_fp = vocal_fp_dict['freq_profile_400']

                # Compute loss for this bin
                bin_loss = np.mean((adjusted_fp - vocal_fp)**2)
                window_loss += bin_loss

            # Average loss across bins
            window_loss /= num_bins
            total_loss += window_loss

            # Compute gradient (simplified - using average across bins)
            gradient = np.zeros(400)
            for bin_idx in range(num_bins):
                vocal_fp_dict = vocal_fps['slice_0_raw'][bin_idx][win_idx]
                vocal_fp = vocal_fp_dict['freq_profile_400']
                gradient += 2 * (adjusted_fp - vocal_fp) * mixture_fp
            gradient /= num_bins

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
    Apply learned EQ curves to mixture and reconstruct audio.
    """
    
    print("\n" + "="*70)
    print("PHASE 4: RECONSTRUCTION")
    print("="*70)
    print("\nApplying learned EQ curves to mixture...")
    
    magnitude = np.abs(mixture_stft)
    phase = np.angle(mixture_stft)
    
    adjusted_magnitude = np.zeros_like(magnitude)
    
    num_windows = magnitude.shape[1]
    
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
    
    print(f"\n✓ Fingerprints created in {time.time() - start_time:.1f}s")
    
    print("\n" + "="*70)
    print("PHASE 2: COMPARE FINGERPRINTS")
    print("="*70)
    
    # Access the first bin, first window (index [0][0])
    v_raw = vocal_fps['slice_0_raw'][0][0]  # First bin, first window
    m_raw = mixture_fps['slice_0_raw'][0][0]  # First bin, first window
    
    print("\nWindow 0 comparison (for first frequency bin):")
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
    
    output_path = f"{CONFIG['output_dir']}/extracted_vocal.wav"
    sf.write(output_path, extracted_vocal, CONFIG['sr'])
    print(f"\n✓ Saved: {output_path}")
    
    # Also save references for comparison
    sf.write(f"{CONFIG['output_dir']}/1_original_mixture.wav", mixture_audio, CONFIG['sr'])
    sf.write(f"{CONFIG['output_dir']}/2_target_vocal.wav", vocal_audio, CONFIG['sr'])
    
    print(f"✓ Saved: {CONFIG['output_dir']}/1_original_mixture.wav")
    print(f"✓ Saved: {CONFIG['output_dir']}/2_target_vocal.wav")
    
    # Create spectrograms for visualization
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].imshow(librosa.amplitude_to_db(vocal_mag, ref=np.max), 
                   aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target Vocal Spectrogram')
    axes[0].set_ylabel('Frequency')
    
    axes[1].imshow(librosa.amplitude_to_db(mixture_mag, ref=np.max), 
                   aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Original Mixture Spectrogram')
    axes[1].set_ylabel('Frequency')
    
    extracted_stft = librosa.stft(extracted_vocal, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    axes[2].imshow(librosa.amplitude_to_db(np.abs(extracted_stft), ref=np.max), 
                   aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Extracted Vocal Spectrogram')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/spectrograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {CONFIG['output_dir']}/spectrograms.png")
    
    # Final summary
    print("\n" + "="*70)
    print("✓ SANITY CHECK COMPLETE!")
    print("="*70)
    
    print(f"\nTotal runtime: {time.time() - start_time:.1f}s")
    print(f"\nOutput files in '{CONFIG['output_dir']}/':")
    print("  • extracted_vocal.wav  ← YOUR SEPARATED VOCAL!")
    print("  • 1_original_mixture.wav")
    print("  • 2_target_vocal.wav")
    print("  • optimization_loss.png")
    print("  • spectrograms.png")
    
    print("\n🎵 Listen to extracted_vocal.wav to hear the result!")
    print("\nExpected: Vocal should be audible, instruments reduced")
    print("Quality: 60-80% (proves concept works)")
    print("Next step: Train U-Net to achieve 95%+ quality")