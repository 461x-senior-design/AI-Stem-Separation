"""
CONTEXTUAL REFINEMENT DEMO
==========================

This script demonstrates how the "context" in which information is processed
can influence the outcome, mimicking the U-Net encoder-decoder pathway where
information flows through a bottleneck.

Usage:
  python contextual_refinement_demo.py

The script will:
  1. Load the master audio.
  2. Perform "Direct Processing": Apply a specific Conv2d process directly to the original spectrogram.
  3. Perform "Bottleneck + Contextual Processing":
     a. Downsample the original spectrogram significantly.
     b. Apply a simple process in the downsampled space.
     c. Upsample the result back to the original size.
     d. Apply the *same* specific Conv2d process used in "Direct" to this upsampled result.
  4. Reconstruct and play audio for both methods.
  5. Highlight how the pathway/context affects the final application of the process.
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
    'output_base_dir': 'output_contextual_demo',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'freq_pool_factor': 4, # Factor to downsample frequency axis in bottleneck
    'time_pool_factor': 2,  # Factor to downsample time axis in bottleneck
    'num_iterations_direct': 50,   # Iterations for direct processing
    'num_iterations_bottleneck': 20, # Iterations for bottleneck processing
    'num_iterations_refine': 50,   # Iterations for final refinement step
    'learning_rate': 0.02,  # Slightly higher LR for demo
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS (Re-used from best.py and other demos)
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

def downsample_freq_time(spectrogram, freq_factor, time_factor):
    """Downsample a spectrogram along frequency and time axes using max pooling."""
    pooled_freq = ndimage.maximum_filter(spectrogram, size=(freq_factor, 1))[::freq_factor, :]
    pooled_time = ndimage.maximum_filter(pooled_freq, size=(1, time_factor))[:, ::time_factor]
    return pooled_time

def upsample_freq_time_simple(spectrogram, target_shape):
    """Upsample a spectrogram back to target_shape using simple repeat_interleave."""
    freq_up_factor = target_shape[0] // spectrogram.shape[0]
    time_up_factor = target_shape[1] // spectrogram.shape[1]

    upsampled = np.repeat(spectrogram, freq_up_factor, axis=0)
    upsampled = np.repeat(upsampled, time_up_factor, axis=1)

    # Ensure the shape matches exactly
    if upsampled.shape[0] > target_shape[0]:
        upsampled = upsampled[:target_shape[0], :]
    if upsampled.shape[1] > target_shape[1]:
        upsampled = upsampled[:, :target_shape[1]]
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
    print("CONTEXTUAL REFINEMENT DEMO")
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
    bottleneck_shape = (original_shape[0] // freq_factor, original_shape[1] // time_factor)

    print(f"Bottleneck downsampling factors: Freq={freq_factor}x, Time={time_factor}x")
    print(f"Bottleneck downsampled shape: {bottleneck_shape}")
    print()

    # --- DEMO STEPS ---

    # 1. Define the "Refinement Process" (e.g., applying slice_1_horizontal features)
    print("--- STEP 1: DEFINING REFINEMENT PROCESS ---")
    refinement_slice_name = 'slice_1_horizontal' # Choose the process/kernel to apply
    print(f"     Using {refinement_slice_name} as the 'refinement' process.")
    print(f"     This process will be applied in two different contexts.")
    print()

    # Create target fingerprints from the original master magnitude for the refinement process
    target_fingerprints_for_refinement = []
    for win_idx in range(master_mag.shape[1]):
        window = master_mag[:, win_idx]
        metrics = window_to_bottleneck(window, CONFIG['sr'])
        target_fingerprints_for_refinement.append(metrics)
    print(f"     Target fingerprints created for refinement process ({len(target_fingerprints_for_refinement)} windows).")
    print()

    # 2. Direct Processing
    print("--- STEP 2: DIRECT PROCESSING ---")
    print(f"     Applying {refinement_slice_name} refinement process directly to the original spectrogram...")
    direct_mag, _ = optimize_simple_eq(
        master_mag, target_fingerprints_for_refinement, CONFIG['num_iterations_direct'], CONFIG['learning_rate'], CONFIG['sr']
    )
    audio_direct = reconstruct_audio_from_magnitude(direct_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/01_direct_processing.wav", audio_direct, CONFIG['sr'])
    print(f"     Saved: 01_direct_processing.wav")
    print()

    # 3. Bottleneck + Contextual Processing
    print("--- STEP 3: BOTTLENECK + CONTEXTUAL PROCESSING ---")
    print(f" 3a. Downsample original to bottleneck shape {bottleneck_shape}...")
    downsampled_mag = downsample_freq_time(master_mag, freq_factor, time_factor)

    print(f" 3b. Apply a simple process in bottleneck space (e.g., smoothing)...")
    bottleneck_processed = ndimage.gaussian_filter(downsampled_mag, sigma=0.5)

    print(f" 3c. Upsample bottleneck result back to original shape {original_shape}...")
    upsampled_bottleneck_mag = upsample_freq_time_simple(bottleneck_processed, original_shape)

    print(f" 3d. Apply the SAME {refinement_slice_name} refinement process to the upsampled result...")
    # Note: We use the *same* target fingerprints as the direct processing,
    # but now the process operates on the upsampled, potentially altered spectrogram.
    contextual_mag, _ = optimize_simple_eq(
        upsampled_bottleneck_mag, target_fingerprints_for_refinement, CONFIG['num_iterations_refine'], CONFIG['learning_rate'], CONFIG['sr']
    )
    audio_contextual = reconstruct_audio_from_magnitude(contextual_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_contextual_processing.wav", audio_contextual, CONFIG['sr'])
    print(f"     Saved: 02_contextual_processing.wav")
    print()

    # 4. Compare with Original
    print("--- STEP 4: SAVING ORIGINAL FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/00_original.wav", master_audio, CONFIG['sr'])
    print(f"     Saved: 00_original.wav")
    print()

    print("="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the difference between direct and contextual processing.")
    print("  - 00_original.wav: The original master mix.")
    print(f"  - 01_direct_processing.wav: Result of applying {refinement_slice_name} directly to the original.")
    print(f"  - 02_contextual_processing.wav: Result of applying {refinement_slice_name} after the signal has been through a bottleneck and upsampled.")
    print("The two processed results might sound subtly different due to the different 'context' the refinement process operated in.")
    print("="*70)