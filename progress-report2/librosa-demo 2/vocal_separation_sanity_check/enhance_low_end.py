"""
LOW-END ENHANCEMENT - Match bass/sub-bass to a reference track
==================================================================

Takes two tracks:
1. Reference (song with good low-end - like Portishead)
2. Target (your track that needs better bass)

Analyzes the low-end spectral fingerprint (0-200Hz) and applies EQ
to make your track's bass sound like the reference.

Based on the vocal separation approach but focused on low frequencies.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("LOW-END ENHANCEMENT")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'reference_path': 'reference_track.wav',  # Song with GOOD low-end
    'target_path': 'target_track.wav',        # Your track to enhance
    'output_dir': 'output_bass_enhanced',
    'duration': None,  # Process full tracks (or set to number for testing)
    
    # Low-end focused settings
    'n_fft': 8192,  # Higher for better low-freq resolution
    'hop_length': 2048,
    'low_freq_max': 200,  # Focus on 0-200Hz
    'sub_bass_max': 60,   # Sub-bass is 0-60Hz
    
    # Optimization
    'num_iterations': 150,
    'learning_rate': 0.015,
}

Path(CONFIG['output_dir']).mkdir(exist_ok=True)

# ============================================
# LOW-END ANALYSIS
# ============================================

def extract_low_end_fingerprint(audio_path, n_fft, hop_length, duration=None):
    """
    Extract detailed low-end spectral fingerprint (0-200Hz)
    """
    
    print(f"\n[Processing: {Path(audio_path).name}]")
    
    # Load at 44.1kHz
    audio, sr = librosa.load(audio_path, sr=44100, duration=duration)
    print(f"  Loaded: {len(audio)} samples at {sr} Hz")
    
    # High-resolution STFT for low frequencies
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    print(f"  STFT: {magnitude.shape} (high resolution for low-end)")
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Focus on low frequencies only (0-200Hz)
    low_freq_mask = freqs <= CONFIG['low_freq_max']
    low_freq_indices = np.where(low_freq_mask)[0]
    
    # Extract low-end portion
    low_end_mag = magnitude[low_freq_mask, :]
    
    print(f"  Low-end bins: {len(low_freq_indices)} (0-{CONFIG['low_freq_max']}Hz)")
    print(f"  Low-end spectrogram: {low_end_mag.shape}")
    
    # Separate harmonic (bass notes) and percussive (kick drum)
    # Work with STFT instead of audio to avoid shape issues
    stft_harmonic, stft_percussive = librosa.decompose.hpss(stft, margin=3.0)
    
    # Get magnitudes for each component
    harmonic_low = np.abs(stft_harmonic)[low_freq_mask, :]
    percussive_low = np.abs(stft_percussive)[low_freq_mask, :]
    
    print(f"  ✓ Separated harmonic/percussive components")
    
    # Extract low-end features per time window
    num_windows = low_end_mag.shape[1]
    fingerprints = []
    
    for win_idx in range(num_windows):
        window = low_end_mag[:, win_idx]
        harmonic_win = harmonic_low[:, win_idx]
        percussive_win = percussive_low[:, win_idx]
        
        # Sub-bass energy (0-60Hz)
        sub_bass_mask = freqs[low_freq_mask] <= CONFIG['sub_bass_max']
        sub_bass_energy = np.sum(window[sub_bass_mask]**2)
        
        # Bass energy (60-200Hz)
        bass_mask = (freqs[low_freq_mask] > CONFIG['sub_bass_max']) & (freqs[low_freq_mask] <= CONFIG['low_freq_max'])
        bass_energy = np.sum(window[bass_mask]**2)
        
        # Harmonic vs percussive balance
        harmonic_energy = np.sum(harmonic_win**2)
        percussive_energy = np.sum(percussive_win**2)
        
        # RMS energy
        rms = np.sqrt(np.mean(window**2))
        
        # Spectral centroid of low-end
        if np.sum(window) > 1e-10:
            centroid = np.sum(freqs[low_freq_mask] * window) / np.sum(window)
        else:
            centroid = 0
        
        # Peak frequency
        peak_idx = np.argmax(window)
        peak_freq = freqs[low_freq_mask][peak_idx]
        
        fingerprints.append({
            'sub_bass_energy': sub_bass_energy,
            'bass_energy': bass_energy,
            'harmonic_energy': harmonic_energy,
            'percussive_energy': percussive_energy,
            'rms': rms,
            'centroid': centroid,
            'peak_freq': peak_freq,
            'full_spectrum': window,  # Full low-end spectrum for this window
        })
    
    print(f"  ✓ Extracted {len(fingerprints)} window fingerprints")
    
    return {
        'audio': audio,
        'sr': sr,
        'stft': stft,
        'magnitude': magnitude,
        'low_end_mag': low_end_mag,
        'fingerprints': fingerprints,
        'freqs': freqs,
        'low_freq_indices': low_freq_indices,
    }

# ============================================
# OPTIMIZATION
# ============================================

def optimize_low_end(reference_data, target_data):
    """
    Optimize target's low-end to match reference
    """
    
    print("\n" + "="*70)
    print("OPTIMIZING LOW-END")
    print("="*70)
    
    ref_fps = reference_data['fingerprints']
    target_fps = target_data['fingerprints']
    target_low_mag = target_data['low_end_mag']
    
    num_windows = min(len(ref_fps), len(target_fps))
    num_low_bins = target_low_mag.shape[0]
    
    print(f"\nOptimizing {num_windows} windows × {num_low_bins} low-freq bins...")
    print(f"Target: Match target's low-end to reference's character\n")
    
    # Initialize EQ curves for low-end (one per window)
    low_end_eq = [np.ones(num_low_bins) for _ in range(num_windows)]
    
    losses = []
    
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0
        
        for win_idx in range(num_windows):
            ref_fp = ref_fps[win_idx]
            target_fp = target_fps[win_idx]
            
            # Apply current EQ
            target_spectrum = target_fp['full_spectrum'] * low_end_eq[win_idx]
            ref_spectrum = ref_fp['full_spectrum']
            
            # Loss: spectral difference
            spectrum_loss = np.mean((target_spectrum - ref_spectrum)**2)
            
            # Loss: energy matching
            target_sub = np.sum(target_spectrum[:len(target_spectrum)//3]**2)
            ref_sub = ref_fp['sub_bass_energy']
            energy_loss = (target_sub - ref_sub)**2
            
            # Combined loss
            loss = spectrum_loss + 0.1 * energy_loss
            total_loss += loss
            
            # Gradient
            gradient = 2 * (target_spectrum - ref_spectrum) * target_fp['full_spectrum']
            
            # Update EQ
            low_end_eq[win_idx] -= CONFIG['learning_rate'] * gradient
            low_end_eq[win_idx] = np.clip(low_end_eq[win_idx], 0.1, 5.0)
        
        avg_loss = total_loss / num_windows
        losses.append(avg_loss)
        
        if iteration % 30 == 0:
            print(f"  Iteration {iteration:3d}: Loss = {avg_loss:.6f}")
    
    print(f"\n✓ Optimization complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Low-End Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{CONFIG['output_dir']}/optimization_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return low_end_eq

# ============================================
# RECONSTRUCTION
# ============================================

def apply_low_end_enhancement(target_data, low_end_eq):
    """
    Apply learned low-end EQ to target track
    """
    
    print("\n" + "="*70)
    print("APPLYING LOW-END ENHANCEMENT")
    print("="*70)
    
    magnitude = target_data['magnitude'].copy()
    phase = np.angle(target_data['stft'])
    low_freq_indices = target_data['low_freq_indices']
    
    num_windows = magnitude.shape[1]
    num_eq_windows = len(low_end_eq)
    
    print(f"\nApplying EQ to {num_windows} windows...")
    
    # Apply EQ to low-end only
    for win_idx in range(min(num_windows, num_eq_windows)):
        # Get EQ for this window
        eq_curve = low_end_eq[win_idx]
        
        # Apply to low frequencies only
        magnitude[low_freq_indices, win_idx] *= eq_curve
    
    print("  ✓ Low-end EQ applied")
    
    # Reconstruct
    enhanced_stft = magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=CONFIG['hop_length'], n_fft=CONFIG['n_fft'])
    
    print("  ✓ Audio reconstructed")
    
    # Normalize
    enhanced_audio = enhanced_audio / (np.max(np.abs(enhanced_audio)) + 1e-8)
    
    return enhanced_audio

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import time
    start = time.time()
    
    # Load and analyze reference
    print("\n" + "="*70)
    print("ANALYZING REFERENCE TRACK (good low-end)")
    print("="*70)
    
    reference_data = extract_low_end_fingerprint(
        CONFIG['reference_path'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        duration=CONFIG['duration']
    )
    
    # Load and analyze target
    print("\n" + "="*70)
    print("ANALYZING TARGET TRACK (to be enhanced)")
    print("="*70)
    
    target_data = extract_low_end_fingerprint(
        CONFIG['target_path'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        duration=CONFIG['duration']
    )
    
    # Optimize
    low_end_eq = optimize_low_end(reference_data, target_data)
    
    # Apply enhancement
    enhanced_audio = apply_low_end_enhancement(target_data, low_end_eq)
    
    # Save
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_path = f"{CONFIG['output_dir']}/enhanced_low_end.wav"
    sf.write(output_path, enhanced_audio, target_data['sr'])
    print(f"\n✓ Saved: {output_path}")
    
    # Save comparison files
    sf.write(f"{CONFIG['output_dir']}/1_reference.wav", reference_data['audio'], reference_data['sr'])
    sf.write(f"{CONFIG['output_dir']}/2_original_target.wav", target_data['audio'], target_data['sr'])
    print(f"✓ Saved: {CONFIG['output_dir']}/1_reference.wav")
    print(f"✓ Saved: {CONFIG['output_dir']}/2_original_target.wav")
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Reference low-end
    axes[0].imshow(librosa.amplitude_to_db(reference_data['low_end_mag'], ref=np.max),
                   aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title(f'Reference Low-End (0-{CONFIG["low_freq_max"]}Hz)')
    axes[0].set_ylabel('Frequency')
    
    # Original target low-end
    axes[1].imshow(librosa.amplitude_to_db(target_data['low_end_mag'], ref=np.max),
                   aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title('Original Target Low-End')
    axes[1].set_ylabel('Frequency')
    
    # Enhanced target low-end
    enhanced_stft = librosa.stft(enhanced_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    enhanced_low = np.abs(enhanced_stft)[target_data['low_freq_indices'], :]
    axes[2].imshow(librosa.amplitude_to_db(enhanced_low, ref=np.max),
                   aspect='auto', origin='lower', cmap='magma')
    axes[2].set_title('Enhanced Target Low-End')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/low_end_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {CONFIG['output_dir']}/low_end_comparison.png")
    
    print("\n" + "="*70)
    print("✓ LOW-END ENHANCEMENT COMPLETE!")
    print("="*70)
    print(f"\nTotal runtime: {time.time() - start:.1f}s")
    print(f"\nOutput files in '{CONFIG['output_dir']}':")
    print("  • enhanced_low_end.wav  ← YOUR ENHANCED TRACK!")
    print("  • 1_reference.wav")
    print("  • 2_original_target.wav")
    print("  • optimization_loss.png")
    print("  • low_end_comparison.png")
    print("\n🎵 Listen to enhanced_low_end.wav to hear the result!")
