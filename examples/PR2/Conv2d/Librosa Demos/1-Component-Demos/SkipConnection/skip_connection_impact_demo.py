"""
SKIP CONNECTION IMPACT DEMO
===========================

This script demonstrates the effect of skip connections in U-Net-like architectures
by comparing audio reconstructed from:
1. A "coarse" path (downsampled -> processed -> upsampled).
2. The "coarse" path combined with a "fine" path (e.g., original signal details).

Usage:
  python skip_connection_impact_demo.py

The script will:
  1. Load the master audio.
  2. Create a "coarse path" result by downsampling, minimal processing, and upsampling.
  3. Treat the original magnitude (or a simple processed version) as the "fine path".
  4. Combine the "coarse" and "fine" paths (e.g., by addition).
  5. Reconstruct and play audio for the "coarse only" and "coarse + fine (skip)" results.
  6. Highlight how the skip connection restores detail lost in the bottleneck.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ============================================
# CONFIGURATION (Assumes files are in 'rtg/100-window/')
# ============================================

CONFIG = {
    'mixture_path': 'rtg/100-window/stereo_mixture.wav', # Use the mixture as the master
    'output_base_dir': 'output_skip_demo',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'freq_pool_factor': 4, # Factor to downsample frequency axis in bottleneck
    'time_pool_factor': 2,  # Factor to downsample time axis in bottleneck
    'num_iterations': 50,   # Reduced iterations for faster demo
    'learning_rate': 0.02,  # Slightly higher LR for demo
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS
# ============================================

def downsample_freq_time(spectrogram, freq_factor, time_factor):
    """
    Downsample a spectrogram along frequency and time axes using max pooling.
    """
    pooled_freq = ndimage.maximum_filter(spectrogram, size=(freq_factor, 1))[::freq_factor, :]
    pooled_time = ndimage.maximum_filter(pooled_freq, size=(1, time_factor))[:, ::time_factor]
    return pooled_time

def upsample_freq_time_simple(spectrogram, target_shape):
    """
    Upsample a spectrogram back to target_shape using simple repeat_interleave.
    """
    freq_up_factor = target_shape[0] // spectrogram.shape[0]
    time_up_factor = target_shape[1] // spectrogram.shape[1]

    # Use repeat_interleave to upsample
    upsampled = np.repeat(spectrogram, freq_up_factor, axis=0)
    upsampled = np.repeat(upsampled, time_up_factor, axis=1)

    # Ensure the shape matches exactly (might need trimming/padding if not divisible)
    if upsampled.shape[0] > target_shape[0]:
        upsampled = upsampled[:target_shape[0], :]
    if upsampled.shape[1] > target_shape[1]:
        upsampled = upsampled[:, :target_shape[1]]

    # If still smaller, pad with the last value
    if upsampled.shape[0] < target_shape[0]:
        pad_size = target_shape[0] - upsampled.shape[0]
        upsampled = np.pad(upsampled, ((0, pad_size), (0, 0)), mode='edge')
    if upsampled.shape[1] < target_shape[1]:
        pad_size = target_shape[1] - upsampled.shape[1]
        upsampled = np.pad(upsampled, ((0, 0), (0, pad_size)), mode='edge')

    return upsampled

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """Reconstruct audio from magnitude and phase using librosa"""
    reconstructed_stft = magnitude * phase
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

def window_to_bottleneck(window, sr):
    """Extract 400-point frequency profile directly from window (as used in best.py)"""
    freq_profile_400 = np.interp(
        x=np.linspace(0, sr/2, 400),
        xp=np.linspace(0, sr/2, len(window)),
        fp=window
    )
    return {
        'freq_profile_400': np.nan_to_num(freq_profile_400, nan=0.0, posinf=0.0, neginf=0.0),
    }

def optimize_simple_eq(magnitude_spectrogram, target_fingerprints, num_iterations, learning_rate, sr):
    """
    Simplified optimization to learn an EQ curve per window, matching target fingerprints.
    Uses the same dB-space strategy as best.py.
    """
    working_mag_db = librosa.amplitude_to_db(magnitude_spectrogram, ref=1.0, amin=1e-8)
    eq_curves_db = [np.zeros(400) for _ in range(magnitude_spectrogram.shape[1])]
    losses = []

    for iteration in range(num_iterations):
        total_loss = 0.0
        for win_idx in range(magnitude_spectrogram.shape[1]):
            # Get target fingerprint for this window
            vocal_fp = target_fingerprints[win_idx]['freq_profile_400']
            vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)

            # Get current working magnitude window (in dB space)
            mixture_window_db = working_mag_db[:, win_idx]

            # Convert to 400-point representation
            mixture_fp_db = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window_db)),
                fp=mixture_window_db
            )

            # Apply current EQ
            adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]

            # Compute loss
            loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)
            total_loss += loss

            # Compute gradient
            gradient = 2 * (adjusted_fp_db - vocal_fp_db)

            # Update EQ curve
            eq_curves_db[win_idx] -= learning_rate * gradient
            eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20) # Clip to +/- 20 dB

        avg_loss = total_loss / magnitude_spectrogram.shape[1]
        losses.append(avg_loss)

        if iteration % 20 == 0:
            print(f"    Simple EQ Iteration {iteration:3d}: Loss = {avg_loss:.6f}")

    # Apply learned EQ to working magnitude (in dB space)
    refined_mag_db = np.zeros_like(working_mag_db)
    for win_idx in range(magnitude_spectrogram.shape[1]):
        window_mag_db = working_mag_db[:, win_idx]

        freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
        freq_points_eq = np.linspace(0, sr/2, 400)

        eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])

        refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

    # Convert back to linear amplitude
    refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
    refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)

    return refined_mag, losses

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()

    print("="*70)
    print("SKIP CONNECTION IMPACT DEMO")
    print("="*70)
    print(f"Loading master audio: {CONFIG['mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print()

    # Load master audio
    master_audio, _ = librosa.load(CONFIG['mixture_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    print(f"Loaded master audio: {len(master_audio)} samples")

    # Create STFT of master
    master_stft = librosa.stft(master_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    master_mag, master_phase = librosa.magphase(master_stft)
    original_shape = master_mag.shape
    print(f"Original spectrogram shape: {original_shape} (Freq x Time)")
    print()

    # Define downsample factors for bottleneck
    freq_factor = CONFIG['freq_pool_factor']
    time_factor = CONFIG['time_pool_factor']

    print(f"Bottleneck downsampling factors: Freq={freq_factor}x, Time={time_factor}x")
    downsampled_shape = (original_shape[0] // freq_factor, original_shape[1] // time_factor)
    print(f"Bottleneck downsampled shape: {downsampled_shape}")
    print()

    # --- DEMO STEPS ---

    # 1. Create "Coarse Path" (Bottleneck)
    print("--- STEP 1: CREATING COARSE PATH (BOTTLENECK) ---")
    print(f" 1a. Downsample master to {downsampled_shape}...")
    downsampled_mag = downsample_freq_time(master_mag, freq_factor, time_factor)

    print(f" 1b. Apply minimal processing in bottleneck (e.g., slight smoothing)...")
    # Example: Simple smoothing in the bottleneck
    bottleneck_processed = ndimage.gaussian_filter(downsampled_mag, sigma=0.5)

    print(f" 1c. Upsample bottleneck result back to {original_shape}...")
    upsampled_bottleneck_mag = upsample_freq_time_simple(bottleneck_processed, original_shape)

    print(f" 1d. Reconstruct audio from upsampled bottleneck (Coarse Only)...")
    coarse_only_audio = reconstruct_audio_from_magnitude(upsampled_bottleneck_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/01_coarse_only.wav", coarse_only_audio, CONFIG['sr'])
    print(f"     Saved: 01_coarse_only.wav")
    print(f"     This audio is blurred due to information loss in bottleneck.")
    print()

    # 2. Create "Fine Path" (Original Detail)
    print("--- STEP 2: DEFINING FINE PATH (ORIGINAL DETAILS) ---")
    # The "fine path" here is conceptually the original master magnitude.
    # We'll use a simple processed version as the "detail map" to add.
    # For demonstration, let's imagine we have a target fingerprint (e.g., from isolated vocal)
    # We'll use the original master magnitude itself as the "detail" to add back.
    # This is a simplification; in practice, the skip connection carries
    # features from the encoder side.
    fine_detail_mag = master_mag # Conceptually, this is the high-res detail
    print(f"     Fine path magnitude shape: {fine_detail_mag.shape}")
    print(f"     This represents the original high-resolution information.")
    print()

    # 3. Combine "Coarse" and "Fine" Paths (Skip Connection Concept)
    print("--- STEP 3: COMBINING PATHS (SKIP CONNECTION) ---")
    print(f"     Combining 'Coarse Only' result with 'Fine Path' information...")
    # Simple addition (standard U-Net skip connection method)
    combined_mag = upsampled_bottleneck_mag + fine_detail_mag
    # Optional: Normalize combined magnitude if values become too large
    combined_mag = np.clip(combined_mag, 0, np.max(master_mag) * 2) # Prevent extreme values

    print(f"     Reconstructing audio from combined result (Coarse + Fine)...")
    coarse_plus_fine_audio = reconstruct_audio_from_magnitude(combined_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_coarse_plus_fine.wav", coarse_plus_fine_audio, CONFIG['sr'])
    print(f"     Saved: 02_coarse_plus_fine.wav")
    print(f"     This audio should sound sharper/less blurred than 'Coarse Only',")
    print(f"     demonstrating how the skip connection restores fine detail.")
    print()

    # 4. Alternative: Use a learned adjustment on the combined result
    print("--- STEP 4: OPTIONAL - REFINING COMBINED RESULT ---")
    print(f"     Applying a simple learned EQ to the combined result...")
    # Create target fingerprints from the original master magnitude (as a proxy for the target vocal)
    # This simulates learning a final adjustment based on the target.
    target_fingerprints = []
    for win_idx in range(master_mag.shape[1]):
        window = master_mag[:, win_idx]
        metrics = window_to_bottleneck(window, CONFIG['sr'])
        target_fingerprints.append(metrics)

    # Optimize an EQ curve to adjust the combined result towards the target fingerprints
    refined_combined_mag, _ = optimize_simple_eq(
        combined_mag, target_fingerprints, CONFIG['num_iterations'], CONFIG['learning_rate'], CONFIG['sr']
    )
    refined_audio = reconstruct_audio_from_magnitude(refined_combined_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/03_refined_combined.wav", refined_audio, CONFIG['sr'])
    print(f"     Saved: 03_refined_combined.wav")
    print(f"     This is the combined result after a simple learned refinement.")
    print()

    print("="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the impact of the skip connection concept.")
    print("  - 01_coarse_only.wav: Blurred due to bottleneck.")
    print("  - 02_coarse_plus_fine.wav: Blurred + Original detail (sharper than 01).")
    print("  - 03_refined_combined.wav: Further refined version of 02.")
    print("="*70)