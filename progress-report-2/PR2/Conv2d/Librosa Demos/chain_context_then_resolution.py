"""
CHAIN: CONTEXTUAL REFINEMENT -> RESOLUTION CHANGE
=================================================

This script chains Demo 5 (Contextual Refinement) and Demo 1 (Resolution Change).
It first applies a bottleneck and refinement process to the master audio,
then applies a downsample/upsample process to the *result* of the refinement.

Usage:
  python chain_context_then_resolution.py

The script will:
  1. Load the master audio.
  2. Apply the contextual refinement process from Demo 5.
  3. Apply the resolution change process from Demo 1 (downsample/upsample) to the result of step 2.
  4. Reconstruct and play the final audio.
  5. Compare against the original and the intermediate refinement result.
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
    'output_base_dir': 'output_chain_context_res',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'freq_pool_factor': 4, # Factor to downsample frequency axis in bottleneck/res change
    'time_pool_factor': 2,  # Factor to downsample time axis in bottleneck/res change
    'num_iterations_bottleneck': 15, # Iterations for bottleneck processing in context
    'num_iterations_refine': 30,   # Iterations for final refinement step in context
    'num_iterations_res_change': 15, # Iterations for potential processing after res change
    'learning_rate': 0.02,  # Slightly higher LR for demo
    'refinement_slice_name': 'slice_1_horizontal', # Slice used for refinement in context demo
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS (Copied from Demo 5 and Demo 1)
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

def downsample_freq_time(spectrogram, freq_factor, time_factor):
    """Downsample a spectrogram along frequency and time axes using max pooling."""
    orig_freq, orig_time = spectrogram.shape
    target_freq = orig_freq // freq_factor
    target_time = orig_time // time_factor
    trimmed_spec = spectrogram[:target_freq * freq_factor, :target_time * time_factor]
    pooled_freq = ndimage.maximum_filter(trimmed_spec, size=(freq_factor, 1))[::freq_factor, :]
    pooled_time = ndimage.maximum_filter(pooled_freq, size=(1, time_factor))[:, ::time_factor]
    return pooled_time

def upsample_freq_time_simple(spectrogram, target_shape):
    """Upsample a spectrogram back to target_shape using simple repeat_interleave."""
    freq_up_factor = target_shape[0] // spectrogram.shape[0]
    time_up_factor = target_shape[1] // spectrogram.shape[1]
    upsampled = np.repeat(spectrogram, freq_up_factor, axis=0)
    upsampled = np.repeat(upsampled, time_up_factor, axis=1)
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

        if iteration % 10 == 0:
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
    print("CHAIN: CONTEXTUAL REFINEMENT -> RESOLUTION CHANGE")
    print("="*70)
    print(f"Loading master audio: {CONFIG['mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print(f"Using slice for refinement: {CONFIG['refinement_slice_name']}")
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

    # Define downsample factors for bottleneck/res change
    freq_factor = CONFIG['freq_pool_factor']
    time_factor = CONFIG['time_pool_factor']
    bottleneck_shape = (original_shape[0] // freq_factor, original_shape[1] // time_factor)

    print(f"Bottleneck/res change downsampling factors: Freq={freq_factor}x, Time={time_factor}x")
    print(f"Bottleneck/res change downsampled shape: {bottleneck_shape}")
    print()

    # --- CHAIN STEPS ---

    # 1. Apply Contextual Refinement Process (Demo 5 Concept)
    print("--- STEP 1: APPLYING CONTEXTUAL REFINEMENT (DEMO 5 Concept) ---")
    print(f" 1a. Downsample original to bottleneck shape {bottleneck_shape}...")
    downsampled_mag = downsample_freq_time(master_mag, freq_factor, time_factor)

    print(f" 1b. Apply a simple process in bottleneck space (e.g., smoothing)...")
    bottleneck_processed = ndimage.gaussian_filter(downsampled_mag, sigma=0.5)

    print(f" 1c. Upsample bottleneck result back to original shape {original_shape}...")
    upsampled_bottleneck_mag = upsample_freq_time_simple(bottleneck_processed, original_shape)

    print(f" 1d. Apply {CONFIG['refinement_slice_name']} refinement process to the upsampled result...")
    # Create target fingerprints from the original master magnitude for the refinement process
    target_fingerprints_for_refinement = []
    for win_idx in range(master_mag.shape[1]):
        window = master_mag[:, win_idx]
        metrics = window_to_bottleneck(window, CONFIG['sr'])
        target_fingerprints_for_refinement.append(metrics)

    contextual_mag, _ = optimize_simple_eq(
        upsampled_bottleneck_mag, target_fingerprints_for_refinement, CONFIG['num_iterations_refine'], CONFIG['learning_rate'], CONFIG['sr']
    )
    print(f" 1e. Reconstructing audio from contextual refinement result (Intermediate)...")
    intermediate_audio_contextual = reconstruct_audio_from_magnitude(contextual_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/01_intermediate_contextual.wav", intermediate_audio_contextual, CONFIG['sr'])
    print(f"     Saved: 01_intermediate_contextual.wav")
    print()

    # 2. Apply Resolution Change (Downsample/Process/Upsample) to the Contextual Result (Demo 1 Concept)
    print("--- STEP 2: APPLYING RESOLUTION CHANGE (DEMO 1 Concept) ---")
    print(f" 2a. Downsample contextual result {original_shape} to {bottleneck_shape}...")
    downsampled_contextual_mag = downsample_freq_time(contextual_mag, freq_factor, time_factor)

    print(f" 2b. Apply a simple process in bottleneck space (e.g., smoothing)...")
    bottleneck_processed_contextual = ndimage.gaussian_filter(downsampled_contextual_mag, sigma=0.5)

    print(f" 2c. Upsample bottleneck result back to original shape {original_shape}...")
    upsampled_contextual_mag = upsample_freq_time_simple(bottleneck_processed_contextual, original_shape)

    print(f" 2d. Reconstructing final audio from resolution-changed contextual result...")
    final_audio = reconstruct_audio_from_magnitude(upsampled_contextual_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_final_result.wav", final_audio, CONFIG['sr'])
    print(f"     Saved: 02_final_result.wav")
    print(f"     This audio is the result of applying resolution change to the contextual refinement output.")
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
    print("Listen to the files to hear the effect of chaining contextual refinement with resolution change.")
    print("  - 00_original.wav: The original master mix.")
    print(f"  - 01_intermediate_contextual.wav: Result of the contextual refinement process (Demo 5 concept applied).")
    print(f"  - 02_final_result.wav: Result of applying resolution change (down/proc/up) to the intermediate result.")
    print("The final result shows how the refined signal behaves after a resolution alteration.")
    print("="*70)
