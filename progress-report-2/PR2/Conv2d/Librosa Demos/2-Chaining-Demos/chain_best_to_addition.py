"""
CHAIN: BEST PROGRESSIVE -> ADDITIVE ADJUSTMENT
==============================================

This script takes the output magnitude from a progressive refinement (like best.py) and applies
an additive adjustment process (Demo 4 concept) to it.

Usage:
  python chain_best_to_addition.py
  (Assumes you have run best.py and have an output file like 18_slices/extracted_vocal.wav)

The script will:
  1. Load the *refined* audio and its magnitude spectrogram from best.py.
  2. Apply an additive adjustment (like Demo 4) to the *refined magnitude*.
  3. Reconstruct the final audio using the original mixture phase.
  4. Compare against the original refined input and the final result.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import ndimage
from pathlib import Path
import time

# ============================================
# CONFIGURATION (Assumes files are in 'rtg/100-window/' and output_progressive_18slices/)
# ============================================

CONFIG = {
    # Input is the output from best.py
    'refined_input_path': 'output_progressive_8slices/8_slices/extracted_vocal.wav', # The result from best.py
    'original_mixture_path': 'rtg/100-window/stereo_mixture.wav', # Need original phase
    'original_vocal_path': 'rtg/100-window/isolated_vocal.wav', # For comparison
    'output_base_dir': 'output_chain_best_addition',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'slice_to_use': 'slice_1_horizontal', # Choose the slice to demonstrate with
    'num_iterations': 30,   # Reduced iterations for faster demo on refined signal
    'learning_rate': 0.02,  # Slightly higher LR for demo
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS (Copied from Demo 4 and Demo 1)
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
    print("CHAIN: BEST PROGRESSIVE -> ADDITIVE ADJUSTMENT")
    print("="*70)
    print(f"Loading refined input: {CONFIG['refined_input_path']}")
    print(f"Loading original mixture phase: {CONFIG['original_mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print(f"Using slice for addition: {CONFIG['slice_to_use']}")
    print()

    # Load the *refined* audio from best.py
    refined_audio, _ = librosa.load(CONFIG['refined_input_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    print(f"Loaded refined audio: {len(refined_audio)} samples")

    # Load the *original mixture* to get its phase (for final reconstruction)
    mixture_audio, _ = librosa.load(CONFIG['original_mixture_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    mixture_stft = librosa.stft(mixture_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    _, mixture_phase = librosa.magphase(mixture_stft) # We only need the phase
    original_shape = mixture_phase.shape
    print(f"Original mixture phase shape (used for reconstruction): {original_shape}")
    print()

    # Create STFT of the *refined* audio to get its magnitude
    refined_stft = librosa.stft(refined_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    refined_mag, _ = librosa.magphase(refined_stft) # We only need the magnitude
    refined_shape = refined_mag.shape
    print(f"Refined audio spectrogram magnitude shape: {refined_shape}")
    print()

    # --- CHAIN STEPS ---

    # 1. Apply Additive Adjustment (Demo 4 Concept) to the Refined Magnitude
    print("--- STEP 1: APPLYING ADDITIVE ADJUSTMENT (DEMO 4 Concept) ---")
    print(f"  - Applying {CONFIG['slice_to_use']} as an additive EQ to the *refined* magnitude...")
    # Use the *refined magnitude* as the input for creating slices and fingerprints
    all_slices_refined = create_18_slices(refined_mag)
    slice_output_add = all_slices_refined[CONFIG['slice_to_use']]

    # Create target fingerprints from the slice output *on the refined magnitude*
    additive_target_fps = []
    for win_idx in range(slice_output_add.shape[1]):
        window = slice_output_add[:, win_idx]
        metrics = window_to_bottleneck(window, CONFIG['sr'])
        additive_target_fps.append(metrics)

    # Optimize an additive EQ curve to adjust the *refined magnitude* based on its slice features
    final_mag, _ = optimize_additive_eq(
        refined_mag, additive_target_fps, CONFIG['num_iterations'], CONFIG['learning_rate'], CONFIG['sr']
    )
    print(f"  - Additive adjustment applied to refined magnitude.")
    print()

    # 2. Reconstruct Final Audio using Original Mixture Phase
    print("--- STEP 2: RECONSTRUCTING FINAL AUDIO ---")
    print(f"     Reconstructing final audio using adjusted magnitude and original mixture phase...")
    final_audio = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_final_after_addition.wav", final_audio, CONFIG['sr'])
    print(f"     Saved: 02_final_after_addition.wav")
    print(f"     This audio is the result of applying an additive adjustment to the refined output.")
    print()

    # 3. Save the Refined Input for Comparison
    print("--- STEP 3: SAVING REFINED INPUT FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/01_refined_input.wav", refined_audio, CONFIG['sr'])
    print(f"     Saved: 01_refined_input.wav (The audio from best.py)")
    print()

    # 4. Save the Original Mixture and Target for Comparison
    print("--- STEP 4: SAVING ORIGINALS FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/00_original_mixture.wav", mixture_audio, CONFIG['sr'])
    target_vocal, _ = librosa.load(CONFIG['original_vocal_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    sf.write(f"{CONFIG['output_base_dir']}/03_target_vocal.wav", target_vocal, CONFIG['sr'])
    print(f"     Saved: 00_original_mixture.wav, 03_target_vocal.wav")
    print()

    print("="*70)
    print("CHAIN COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the effect of applying an additive adjustment to the progressive refinement output.")
    print("  - 00_original_mixture.wav: The original mix.")
    print("  - 01_refined_input.wav: The refined vocal from best.py (input to this chain).")
    print("  - 02_final_after_addition.wav: The refined vocal after an additive adjustment.")
    print("  - 03_target_vocal.wav: The target vocal.")
    print("="*70)
