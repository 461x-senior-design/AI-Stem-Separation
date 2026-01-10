"""
MASKING VS ADDITION DEMO
========================

This script demonstrates the difference between applying Conv2d-derived features
as an additive gain versus a multiplicative mask to the original spectrogram.

Usage:
  python masking_vs_addition_demo.py

The script will:
  1. Load the master audio.
  2. Process the master spectrogram through a specific Conv2d slice (e.g., slice_1_horizontal).
  3. Generate an "additive adjustment" based on the slice output (like best.py).
  4. Generate a "multiplicative mask" based on the slice output.
  5. Apply the additive adjustment to the original magnitude.
  6. Apply the multiplicative mask to the original magnitude.
  7. Reconstruct and play audio for both methods.
  8. Highlight how the method of application changes the result.
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
    'output_base_dir': 'output_masking_addition_demo',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'slice_to_use': 'slice_1_horizontal', # Choose the slice to demonstrate with
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
# HELPER FUNCTIONS (Re-used from best.py)
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
        # This function is defined later
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

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """Reconstruct audio from magnitude and phase using librosa"""
    reconstructed_stft = magnitude * phase
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

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

        if iteration % 20 == 0:
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
    # This is a simple heuristic, might need adjustment
    normalized_slice = slice_output / (np.max(np.abs(slice_output)) + 1e-8) * 6.0
    mask = 1.0 / (1.0 + np.exp(-alpha * normalized_slice))
    # Ensure mask is within [0, 2] or similar, to avoid extreme amplification
    mask = np.clip(mask, 0.0, 2.0)
    return mask


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()

    print("="*70)
    print("MASKING VS ADDITION DEMO")
    print("="*70)
    print(f"Loading master audio: {CONFIG['mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print(f"Using slice: {CONFIG['slice_to_use']}")
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

    # --- DEMO STEPS ---

    # 1. Create the specified slice output
    print("--- STEP 1: CREATING SPECIFIED CONV2D SLICE OUTPUT ---")
    print(f"  - Applying {CONFIG['slice_to_use']} kernel to master spectrogram...")
    all_slices = create_18_slices(master_mag)
    slice_output = all_slices[CONFIG['slice_to_use']]
    print(f"  - Slice output shape: {slice_output.shape}")
    print()

    # 2. Generate Additive Adjustment
    print("--- STEP 2: GENERATING ADDITIVE ADJUSTMENT ---")
    print(f"  - Treating slice output as a target fingerprint for additive EQ...")
    # Create target fingerprints from the slice output itself
    additive_target_fps = []
    for win_idx in range(slice_output.shape[1]):
        window = slice_output[:, win_idx]
        metrics = window_to_bottleneck(window, CONFIG['sr'])
        additive_target_fps.append(metrics)

    # Optimize an additive EQ curve to match the slice output fingerprint
    adjusted_mag_additive, _ = optimize_additive_eq(
        master_mag, additive_target_fps, CONFIG['num_iterations'], CONFIG['learning_rate'], CONFIG['sr']
    )
    print(f"  - Additive adjustment applied.")
    print()

    # 3. Generate Multiplicative Mask
    print("--- STEP 3: GENERATING MULTIPLICATIVE MASK ---")
    print(f"  - Converting slice output to a multiplicative mask using sigmoid...")
    mask = create_multiplicative_mask(slice_output, alpha=1.0) # Adjust alpha if needed
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Mask value range: [{mask.min():.4f}, {mask.max():.4f}]")
    # Apply the mask to the original magnitude
    adjusted_mag_multiplicative = master_mag * mask
    # Clip to prevent extreme values
    adjusted_mag_multiplicative = np.clip(adjusted_mag_multiplicative, 0, np.max(master_mag) * 2)
    print(f"  - Multiplicative mask applied.")
    print()

    # 4. Reconstruct Audio for Additive Method
    print("--- STEP 4: RECONSTRUCTING AUDIO (ADDITIVE) ---")
    audio_additive = reconstruct_audio_from_magnitude(adjusted_mag_additive, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/01_additive_adjustment.wav", audio_additive, CONFIG['sr'])
    print(f"     Saved: 01_additive_adjustment.wav")
    print()

    # 5. Reconstruct Audio for Multiplicative Method
    print("--- STEP 5: RECONSTRUCTING AUDIO (MULTIPLICATIVE) ---")
    audio_multiplicative = reconstruct_audio_from_magnitude(adjusted_mag_multiplicative, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_multiplicative_mask.wav", audio_multiplicative, CONFIG['sr'])
    print(f"     Saved: 02_multiplicative_mask.wav")
    print()

    # 6. Compare with Original
    print("--- STEP 6: SAVING ORIGINAL FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/00_original.wav", master_audio, CONFIG['sr'])
    print(f"     Saved: 00_original.wav")
    print()

    print("="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the difference between additive and multiplicative application.")
    print("  - 00_original.wav: The original master mix.")
    print(f"  - 01_additive_adjustment.wav: Result of applying {CONFIG['slice_to_use']} as an additive EQ.")
    print(f"  - 02_multiplicative_mask.wav: Result of applying {CONFIG['slice_to_use']} as a multiplicative mask.")
    print("The two processed results will likely sound quite different.")
    print("="*70)