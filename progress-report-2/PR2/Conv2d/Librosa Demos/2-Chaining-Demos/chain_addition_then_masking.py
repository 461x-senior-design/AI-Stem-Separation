"""
CHAIN: ADDITION -> MASKING
==========================

This script chains Demo 4 (Addition vs Masking) by applying an additive process first,
then applying a multiplicative mask to the *result* of the additive process.

Usage:
  python chain_addition_then_masking.py

The script will:
  1. Load the master audio.
  2. Apply an additive adjustment (like Demo 4) to the master audio.
  3. Apply a multiplicative mask (like Demo 4) to the result of step 2.
  4. Reconstruct and play the final audio.
  5. Compare against the original and the intermediate additive result.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import ndimage
from pathlib import Path
import time

# ============================================
# CONFIGURATION (Assumes files are in 'rtg/100-window/')
# ============================================

CONFIG = {
    'mixture_path': 'rtg/100-window/stereo_mixture.wav', # Use the mixture as the master
    'output_base_dir': 'output_chain_add_mask',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'slice_to_use_add': 'slice_1_horizontal', # Slice for additive adjustment
    'slice_to_use_mask': 'slice_2_vertical',  # Slice for multiplicative mask
    'num_iterations_add': 30,   # Iterations for additive process
    'num_iterations_mask': 30,  # Iterations for masking process (on intermediate result)
    'learning_rate': 0.02,  # Slightly higher LR for demo
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS (Copied from Demo 4)
# ============================================

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

def apply_2d_conv(image, kernel):
    """Apply 2D convolution to spectrogram (as used in best.py)"""
    return ndimage.convolve(image, kernel, mode='constant', cval=0.0)

def create_18_slices(magnitude_spectrogram):
    """
    Create 18 different views/slices of the spectrogram.
    Each slice represents a different feature extraction perspective.
    """
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

    # SLICES 9-15: Oriented edge detectors at different angles
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

def create_oriented_filter(angle_deg, size=3):
    """Create edge detection filter at specific angle (as used in best.py)"""
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

def optimize_additive_eq(magnitude_spectrogram, target_fingerprints, num_iterations, learning_rate, sr):
    """
    Optimization to learn an additive EQ curve per window, matching target fingerprints.
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

        if iteration % 10 == 0:
            print(f"    Additive EQ Iteration {iteration:3d}: Loss = {avg_loss:.6f}")

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

def create_multiplicative_mask(slice_output, alpha=1.0):
    """
    Creates a multiplicative mask from a slice output using a sigmoid function.
    The alpha parameter controls the steepness of the sigmoid.
    """
    # Normalize slice output to a reasonable range for sigmoid, e.g., [-6, 6]
    normalized_slice = slice_output / (np.max(np.abs(slice_output)) + 1e-8) * 6.0
    mask = 1.0 / (1.0 + np.exp(-alpha * normalized_slice))
    # Ensure mask is within [0, 2] or similar, to avoid extreme amplification
    mask = np.clip(mask, 0.0, 2.0)
    return mask

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """Reconstruct audio from magnitude and phase using librosa"""
    reconstructed_stft = magnitude * phase
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()

    print("="*70)
    print("CHAIN: ADDITION -> MASKING")
    print("="*70)
    print(f"Loading master audio: {CONFIG['mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print(f"Using slice for addition: {CONFIG['slice_to_use_add']}")
    print(f"Using slice for masking: {CONFIG['slice_to_use_mask']}")
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

    # --- CHAIN STEPS ---

    # 1. Apply Additive Adjustment (Demo 4 Concept)
    print("--- STEP 1: APPLYING ADDITIVE ADJUSTMENT (DEMO 4 Concept) ---")
    print(f"  - Applying {CONFIG['slice_to_use_add']} as an additive EQ to master spectrogram...")
    all_slices = create_18_slices(master_mag)
    slice_output_add = all_slices[CONFIG['slice_to_use_add']]

    # Create target fingerprints from the slice output itself
    additive_target_fps = []
    for win_idx in range(slice_output_add.shape[1]):
        window = slice_output_add[:, win_idx]
        metrics = window_to_bottleneck(window, CONFIG['sr'])
        additive_target_fps.append(metrics)

    # Optimize an additive EQ curve to match the slice output fingerprint
    intermediate_mag_additive, _ = optimize_additive_eq(
        master_mag, additive_target_fps, CONFIG['num_iterations_add'], CONFIG['learning_rate'], CONFIG['sr']
    )
    print(f"  - Additive adjustment applied.")
    print(f"  - Reconstructing audio from intermediate result (after addition)...")
    intermediate_audio_additive = reconstruct_audio_from_magnitude(intermediate_mag_additive, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/01_intermediate_additive.wav", intermediate_audio_additive, CONFIG['sr'])
    print(f"     Saved: 01_intermediate_additive.wav")
    print()

    # 2. Apply Multiplicative Mask to the Additive Result (Demo 4 Concept)
    print("--- STEP 2: APPLYING MULTIPLICATIVE MASK (DEMO 4 Concept) ---")
    print(f"  - Applying {CONFIG['slice_to_use_mask']} as a multiplicative mask to the additive result...")
    slice_output_mask = all_slices[CONFIG['slice_to_use_mask']] # Use the mask slice on the original
    mask = create_multiplicative_mask(slice_output_mask, alpha=1.0)
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Applying mask to intermediate additive result...")
    final_mag = intermediate_mag_additive * mask # Apply mask to the *result* of addition
    # Clip to prevent extreme values
    final_mag = np.clip(final_mag, 0, np.max(master_mag) * 2)
    print(f"  - Multiplicative mask applied.")
    print(f"  - Reconstructing final audio from combined result (Addition + Masking)...")
    final_audio = reconstruct_audio_from_magnitude(final_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_final_result.wav", final_audio, CONFIG['sr'])
    print(f"     Saved: 02_final_result.wav")
    print(f"     This audio is the result of applying masking to the additive result.")
    print()

    # 3. Compare with Original
    print("--- STEP 3: SAVING ORIGINAL FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/00_original.wav", master_audio, CONFIG['sr'])
    print(f"     Saved: 00_original.wav")
    print()

    print("="*70)
    print("CHAIN COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the effect of chaining additive and multiplicative adjustments.")
    print("  - 00_original.wav: The original master mix.")
    print(f"  - 01_intermediate_additive.wav: Result of applying {CONFIG['slice_to_use_add']} as an additive EQ.")
    print(f"  - 02_final_result.wav: Result of applying {CONFIG['slice_to_use_mask']} as a multiplicative mask to the additive result.")
    print("The final result combines both types of processing.")
    print("="*70)
