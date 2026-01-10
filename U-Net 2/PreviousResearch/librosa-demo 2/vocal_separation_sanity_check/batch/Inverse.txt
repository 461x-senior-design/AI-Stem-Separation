"""
PROGRESSIVE VOCAL SEPARATION TEST - INVERSE VERSION (Difference-Based)
======================================================================

⭐ This is the INVERSE version using the difference between mix and instrumental.
⭐ This version uses Strategy 3 (dB-space optimization).

Key features:
- Uses the difference between Full Mix and Instrumental as the target fingerprint
- Uses Strategy 3 dB-space optimization from test_gain_strategies.py
- Uses librosa.magphase() for magnitude extraction
- Direct 400-point interpolation (no encoder/decoder complexity)
- Handles slices 17-18 (maxpool/avgpool) with correct indexing

This script tests vocal separation using progressively more convolutional slices.
The target fingerprint is derived from (Full Mix - Instrumental).

Usage:
  python sanity_check_progressive_INVERSE.py --slices 18    # Test 1 through 18 slices
  python sanity_check_progressive_INVERSE.py --slices 5     # Test 1 through 5 slices
  python sanity_check_progressive_INVERSE.py                # Default: all 18 slices

For each N from 1 to max_slices:
  - Optimizes using slices 1 through N sequentially
  - Always ends with slice_0_raw as final polish
  - Saves: output_progressive_Nslices/N_slices/extracted_vocal.wav

Expected outcome: Hear progressive improvement based on mix-instrumental difference.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
from scipy.signal import find_peaks, butter, sosfilt
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse

# ============================================
# COMMAND-LINE INTERFACE
# ============================================

parser = argparse.ArgumentParser(
    description='Progressive vocal separation test using difference between mix and instrumental.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s --slices 18    Test all 18 slices
  %(prog)s --slices 5     Quick test with 5 slices
  %(prog)s -n 10          Test up to 10 slices
    """
)

parser.add_argument(
    '--slices', '-n',
    type=int,
    default=18,
    choices=range(1, 19),
    metavar='N',
    help='Number of conv slices to test (1-18, default: 18)'
)

args = parser.parse_args()

print("="*70)
print("PROGRESSIVE VOCAL SEPARATION TEST (Difference-Based)")
print("="*70)
print(f"\nTesting progressive separation using 1 to {args.slices} slices")
print("Target fingerprint is derived from the difference between mix and instrumental.")
print("Each iteration adds one more conv slice, always ending with slice_0_raw polish\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'full_mix_path': 'rtg/100-window/stereo_mixture.wav',      # Path to the full mix audio
    'instrumental_path': 'rtg/100-window/stereo_instrumental.wav', # Path to the instrumental audio
    # Note: We no longer need 'vocal_path' as the target is derived from mix and instrumental
    'output_base_dir': f'output_progressive_diff_{args.slices}slices',
    'max_slices': args.slices,
    'sr': 22050,
    'duration': 4.7,
    'n_fft': 2048,
    'hop_length': 1040,
    'num_iterations': 100,
    'learning_rate': 0.01,
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print(f"Configuration:")
print(f"  Max slices: {CONFIG['max_slices']}")
print(f"  Sample rate: {CONFIG['sr']} Hz")
print(f"  Iterations per slice: {CONFIG['num_iterations']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Full Mix: {CONFIG['full_mix_path']}")
print(f"  Instrumental: {CONFIG['instrumental_path']}")
print(f"  Output directory: {CONFIG['output_base_dir']}/")
print()


# ============================================
# HELPER FUNCTIONS
# ============================================

def downsample_spectrum(spectrum, factor=2):
    """Downsample frequency spectrum (MaxPool-like operation)"""
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

def create_19_slices(magnitude_spectrogram): # Note: Now includes slice_0_raw
    """
    Create 19 different views/slices of the spectrogram (including slice_0_raw).
    Each slice represents a different feature extraction perspective.

    Returns:
        dict: Dictionary mapping slice names to 2D arrays
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

def get_slice_order():
    """
    Return the ordered list of slice names for sequential optimization.
    Slices 1-18 are conv/processed views, slice_0 is always the final polish.
    """
    slice_names = [
        'slice_1_horizontal',
        'slice_2_vertical',
        'slice_3_diagonal_up',
        'slice_4_diagonal_down',
        'slice_5_blob',
        'slice_6_harmonic',
        'slice_7_highpass',
        'slice_8_lowpass',
        'slice_9_edge_22deg',
        'slice_10_edge_45deg',
        'slice_11_edge_67deg',
        'slice_12_edge_90deg',
        'slice_13_edge_112deg',
        'slice_14_edge_135deg',
        'slice_15_edge_157deg',
        'slice_16_laplacian',
        'slice_17_maxpool',
        'slice_18_avgpool',
    ]
    return slice_names


# ============================================
# FINGERPRINT EXTRACTION (Modified)
# ============================================

def window_to_bottleneck(window, sr):
    """Extract 400-point frequency profile directly from window"""
    # Directly interpolate to 400 points (no encoder/decoder needed)
    freq_profile_400 = np.interp(
        x=np.linspace(0, sr/2, 400),
        xp=np.linspace(0, sr/2, len(window)),
        fp=window
    )

    return {
        'freq_profile_400': np.nan_to_num(freq_profile_400, nan=0.0, posinf=0.0, neginf=0.0),
    }

def process_audio_to_fingerprints_difference_based(full_mix_path, instrumental_path, sr, n_fft, hop_length):
    """
    Load full mix and instrumental, calculate difference, and create fingerprints.

    Args:
        full_mix_path: Path to full mix audio file
        instrumental_path: Path to instrumental audio file
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        tuple: (full_mix_audio, instrumental_audio, mix_stft, instrumental_stft, mix_mag, instrumental_mag, diff_mag, diff_phase, fingerprints)
            - diff_mag: Magnitude spectrogram derived from (mix - instrumental)
            - diff_phase: Phase spectrogram derived from mix (or instrumental, depending on strategy)
            - fingerprints: Dict of {slice_name: [window_fingerprints]} based on diff_mag
    """
    print(f"\n[Processing: Mix={Path(full_mix_path).name}, Instrumental={Path(instrumental_path).name}]")

    # Load audio
    full_mix_audio, _ = librosa.load(full_mix_path, sr=sr, duration=CONFIG['duration'])
    instrumental_audio, _ = librosa.load(instrumental_path, sr=sr, duration=CONFIG['duration'])
    print(f"  Loaded mix: {len(full_mix_audio)} samples")
    print(f"  Loaded instrumental: {len(instrumental_audio)} samples")

    # Create STFT for both
    mix_stft = librosa.stft(full_mix_audio, n_fft=n_fft, hop_length=hop_length)
    instrumental_stft = librosa.stft(instrumental_audio, n_fft=n_fft, hop_length=hop_length)

    # Get magnitude and phase
    mix_mag, mix_phase = librosa.magphase(mix_stft)
    instrumental_mag, instrumental_phase = librosa.magphase(instrumental_stft)

    num_windows = mix_mag.shape[1]
    print(f"  Mix Spectrogram: {mix_mag.shape} ({num_windows} windows)")
    print(f"  Instrumental Spectrogram: {instrumental_mag.shape} ({num_windows} windows)")

    # Calculate difference magnitude
    # Option 1: Direct subtraction of magnitudes (can go negative)
    # diff_mag_raw = mix_mag - instrumental_mag
    # diff_mag = np.clip(diff_mag_raw, 0, None) # Clip negative values to 0

    # Option 2: Magnitude of the complex difference (more principled, but requires phase)
    # We don't have the true vocal phase. Common approximations:
    # a) Use mix phase for the difference signal
    # b) Use instrumental phase for the difference signal
    # Let's try Option 2a: diff_mag = |mix_stft - instrumental_stft|, phase = mix_phase
    complex_diff = mix_stft - instrumental_stft
    diff_mag = np.abs(complex_diff)
    diff_phase = mix_phase # Use mix phase as approximation for the difference signal

    print(f"  Estimated Vocal (Diff) Spectrogram: {diff_mag.shape} ({num_windows} windows)")

    # Create 19 slices on the DIFFERENCE magnitude spectrogram
    print(f"  Creating 19 spectral slices on DIFFERENCE magnitude...")
    slices = create_19_slices(diff_mag) # Apply slices to the difference magnitude

    # Process each slice to bottleneck fingerprints (based on diff_mag slices)
    fingerprints = {}

    for slice_name, slice_data in slices.items():
        slice_fingerprints = []
        num_slice_windows = slice_data.shape[1] # This varies per slice!

        for window_idx in range(num_slice_windows):
            window = slice_data[:, window_idx]
            metrics = window_to_bottleneck(window, sr)
            slice_fingerprints.append(metrics)

        fingerprints[slice_name] = slice_fingerprints
        print(f"    {slice_name}: {slice_data.shape} -> {len(slice_fingerprints)} fingerprints")

    print(f"  ✓ Created fingerprints for {len(fingerprints)} slices based on difference magnitude")

    return full_mix_audio, instrumental_audio, mix_stft, instrumental_stft, mix_mag, instrumental_mag, diff_mag, diff_phase, fingerprints


# ============================================
# OPTIMIZATION
# ============================================

def optimize_one_slice(slice_name, diff_fps, mix_mag, num_windows, sr, current_mag=None, iteration_offset=0):
    """
    Optimize EQ curves for ONE slice using gradient descent in dB-space.
    The target is derived from the difference fingerprints (diff_fps).

    Args:
        slice_name: Name of the slice to optimize (e.g., 'slice_1_horizontal')
        diff_fps: Difference fingerprints dict (derived from mix - instrumental)
        mix_mag: Original *mix* magnitude spectrogram (used as the base for applying EQ)
        num_windows: Number of time windows in original mix
        sr: Sample rate
        current_mag: Current working magnitude (for sequential refinement). If None, starts from mix_mag
        iteration_offset: Display offset for iteration counter (for multi-pass clarity)

    Returns:
        tuple: (refined_magnitude, losses)
            - refined_magnitude: Updated magnitude after applying learned EQ
            - losses: List of loss values per iteration
    """

    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {slice_name}")
    print(f"{'='*70}")

    # Get the actual number of windows for this slice from diff_fps
    num_slice_windows = len(diff_fps[slice_name])
    print(f"\nOptimizing {num_slice_windows} windows × 400 EQ points...")  # Use actual count

    # Use current_mag if provided (sequential), otherwise start from mix_mag
    if current_mag is None:
        working_mag = mix_mag.copy()
        print(f"  Starting from: original mix magnitude")
    else:
        working_mag = current_mag.copy()
        print(f"  Starting from: previous pass result")

    # Convert to dB space using librosa (Strategy 3: dB-space optimization)
    working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)

    # Initialize EQ curves (dB adjustments, start at 0)
    eq_curves_db = [np.zeros(400) for _ in range(num_slice_windows)]  # Use actual count

    # Track loss
    losses = []

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        # Process each window (for this slice)
        for win_idx in range(num_slice_windows):  # Use actual slice window count
            # Get target fingerprint for this slice (derived from difference)
            target_fp = diff_fps[slice_name][win_idx]['freq_profile_400']
            # Convert target to dB using librosa
            target_fp_db = librosa.amplitude_to_db(target_fp, ref=1.0, amin=1e-8)

            # Get current working magnitude window (in dB space)
            # IMPORTANT: Use the same win_idx as the slice, but from the original mix spectrogram
            # This assumes the EQ is applied to the *mix* spectrogram bins,
            # using the learned curve derived from the *difference* fingerprint.
            # For pooled slices, this means only applying EQ to the first N windows where N = slice_windows.
            # For now, apply to corresponding original window if it exists.
            if win_idx < working_mag_db.shape[1]:
                mix_window_db = working_mag_db[:, win_idx]

                # Convert to 400-point representation
                mix_fp_db = np.interp(
                    x=np.linspace(0, sr/2, 400),
                    xp=np.linspace(0, sr/2, len(mix_window_db)),
                    fp=mix_window_db
                )

                # Apply current EQ (additive in dB space)
                adjusted_fp_db = mix_fp_db + eq_curves_db[win_idx]

                # Compute loss (mean squared error in dB space)
                loss = np.mean((adjusted_fp_db - target_fp_db)**2)
                total_loss += loss

                # Compute gradient (simpler in dB space - additive)
                gradient = 2 * (adjusted_fp_db - target_fp_db)

                # Update EQ curve (gradient descent)
                eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient

                # Clip to symmetric dB range [±20 dB]
                eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)
            else:
                # If slice has fewer windows than original, skip the rest
                break

        avg_loss = total_loss / num_slice_windows  # Use actual count
        losses.append(avg_loss)

        if iteration % 20 == 0:
            display_iter = iteration_offset + iteration
            print(f"  Iteration {display_iter:3d}: Loss = {avg_loss:.6f}")

    print(f"\n✓ Optimization complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")

    # Apply learned EQ to working magnitude (in dB space)
    refined_mag_db = np.zeros_like(working_mag_db)

    for win_idx in range(num_slice_windows):  # Use actual slice window count
        if win_idx < working_mag_db.shape[1]:
            window_mag_db = working_mag_db[:, win_idx]

            # Interpolate 400-point EQ to full STFT bins
            freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
            freq_points_eq = np.linspace(0, sr/2, 400)

            eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])

            # Apply EQ (additive in dB space)
            refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

    # For windows beyond num_slice_windows, keep original magnitude
    if num_slice_windows < working_mag_db.shape[1]:
        refined_mag_db[:, num_slice_windows:] = working_mag_db[:, num_slice_windows:]

    # Convert back to linear amplitude using librosa
    refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
    refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  ✓ EQ curves applied in dB space, converted back to linear")

    return refined_mag, losses

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """Reconstruct audio from magnitude and phase using librosa"""
    # Use librosa.util.phasor for safer complex reconstruction
    # Note: phase here is the phase of the *mix* (or instrumental), approximating the phase of the vocal
    reconstructed_stft = magnitude * phase
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    # Keep manual normalization (sounds best based on testing)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    start_time = time.time()

    # ============================================
    # PHASE 1: LOAD AND ANALYZE
    # ============================================

    print("\n" + "="*70)
    print("PHASE 1: LOAD AND ANALYZE (Difference-Based)")
    print("="*70)

    # Process full mix and instrumental to get difference fingerprints
    full_mix_audio, instrumental_audio, full_mix_stft, instrumental_stft, full_mix_mag, instrumental_mag, diff_mag, diff_phase, diff_fps = process_audio_to_fingerprints_difference_based(
        CONFIG['full_mix_path'],
        CONFIG['instrumental_path'],
        CONFIG['sr'],
        CONFIG['n_fft'],
        CONFIG['hop_length']
    )

    num_windows = full_mix_mag.shape[1]
    # Note: We use diff_phase (which is mix_phase) for reconstruction
    # This is an approximation. Using instrumental_phase might also be tried.
    reconstruction_phase = diff_phase # Use the phase derived from the difference process

    print(f"\n✓ Fingerprints created from difference in {time.time() - start_time:.1f}s")

    # Get ordered slice names (excluding slice_0_raw for the main sequence)
    all_slice_names = get_slice_order()

    # ============================================
    # PHASE 2: PROGRESSIVE TESTING (Difference-Based)
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: PROGRESSIVE TESTING (Difference-Based)")
    print("="*70)
    print(f"\nTesting {CONFIG['max_slices']} configurations:")
    print("  Each test uses slices 1 through N (in original order).")
    print("  Target fingerprint is derived from (Full Mix - Instrumental).")
    print("  Always ends with slice_0_raw as final polish")
    print()

    # Store results for comparison
    all_results = []

    # Progressive loop: test 1 slice, 2 slices, 3 slices, ..., N slices
    for num_slices in range(1, CONFIG['max_slices'] + 1):
        test_start_time = time.time()

        print("\n" + "="*70)
        print(f"TEST {num_slices}/{CONFIG['max_slices']}: Using {num_slices} conv slice(s) + slice_0_raw")
        print("="*70)

        # Create output directory for this test
        test_output_dir = Path(CONFIG['output_base_dir']) / f"{num_slices}_slices"
        test_output_dir.mkdir(exist_ok=True, parents=True)

        # Select slices to use (1 through num_slices)
        slices_to_use = all_slice_names[:num_slices]

        print(f"\nPipeline:")
        for i, slice_name in enumerate(slices_to_use, 1):
            print(f"  Pass {i}: {slice_name}")
        print(f"  Pass {len(slices_to_use) + 1}: slice_0_raw (final polish)")
        print()

        # Sequential optimization through selected slices
        # Start optimization from the *full mix magnitude*
        current_mag = None
        all_losses = []
        iteration_counter = 0

        for slice_idx, slice_name in enumerate(slices_to_use):
            refined_mag, losses = optimize_one_slice(
                slice_name,
                diff_fps, # Use difference fingerprints as target
                full_mix_mag, # Use the original *mix* magnitude as the base to modify
                num_windows,
                CONFIG['sr'],
                current_mag=current_mag, # Pass the current state
                iteration_offset=iteration_counter
            )
            current_mag = refined_mag
            all_losses.extend(losses)
            iteration_counter += CONFIG['num_iterations']

        # Final polish with slice_0_raw (using difference fingerprints)
        print(f"\n{'='*70}")
        print("FINAL POLISH: slice_0_raw")
        print(f"{'='*70}")

        final_mag, final_losses = optimize_one_slice(
            'slice_0_raw',
            diff_fps, # Use difference fingerprints as target
            full_mix_mag, # Use the original *mix* magnitude as the base
            num_windows,
            CONFIG['sr'],
            current_mag=current_mag, # Pass the state after the main sequence
            iteration_offset=iteration_counter
        )
        all_losses.extend(final_losses)

        # Reconstruct audio using the *estimated* vocal magnitude and the *mix* phase
        print(f"\n{'='*70}")
        print("RECONSTRUCTION (using mix phase for estimated vocal)")
        print(f"{'='*70}")

        extracted_vocal = reconstruct_audio_from_magnitude(
            final_mag, # Use the final magnitude after all slices (the estimated vocal magnitude)
            reconstruction_phase, # Use the phase derived from the mix (or instrumental)
            CONFIG['hop_length'],
            CONFIG['n_fft']
        )

        print("✓ Audio reconstructed and normalized")

        # Save audio
        output_audio_path = test_output_dir / "extracted_vocal.wav"
        sf.write(output_audio_path, extracted_vocal, CONFIG['sr'])
        print(f"✓ Saved: {output_audio_path}")

        # Save references (only for first test to avoid duplication)
        if num_slices == 1:
            sf.write(test_output_dir / "1_original_full_mix.wav", full_mix_audio, CONFIG['sr'])
            sf.write(test_output_dir / "2_original_instrumental.wav", instrumental_audio, CONFIG['sr'])
            # Optionally, save the calculated difference signal as a reference
            diff_audio = librosa.istft(diff_mag * reconstruction_phase, hop_length=CONFIG['hop_length'], n_fft=CONFIG['n_fft'])
            diff_audio = diff_audio / (np.max(np.abs(diff_audio)) + 1e-8)
            sf.write(test_output_dir / "3_calculated_difference.wav", diff_audio, CONFIG['sr'])
            print(f"✓ Saved reference files")

        # Store results
        test_duration = time.time() - test_start_time
        all_results.append({
            'num_slices': num_slices,
            'slices_used': slices_to_use + ['slice_0_raw'],
            'losses': all_losses,
            'final_loss': all_losses[-1],
            'duration': test_duration,
            'output_path': output_audio_path,
        })

        print(f"\n✓ Test {num_slices} complete in {test_duration:.1f}s")
        print(f"  Final loss: {all_losses[-1]:.6f}")


    # ============================================
    # PHASE 3: VISUALIZATION AND SUMMARY
    # ============================================

    print("\n" + "="*70)
    print("PHASE 3: VISUALIZATION AND SUMMARY")
    print("="*70)

    # Create comparison plots
    print("\nGenerating visualizations...")

    # Plot 1: Loss progression for all tests
    fig, ax = plt.subplots(figsize=(14, 6))

    for result in all_results:
        ax.plot(result['losses'], label=f"{result['num_slices']} slice(s)", alpha=0.7)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Optimization Loss Progression (All Tests vs. Difference Target)')
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)

    loss_plot_path = Path(CONFIG['output_base_dir']) / "all_tests_loss_progression.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {loss_plot_path}")

    # Plot 2: Final loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    num_slices_list = [r['num_slices'] for r in all_results]
    final_losses = [r['final_loss'] for r in all_results]

    ax.bar(num_slices_list, final_losses, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of Conv Slices Used')
    ax.set_ylabel('Final Loss (MSE)')
    ax.set_title('Final Loss vs Number of Slices (vs. Difference Target)')
    ax.set_xticks(num_slices_list)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate best result
    best_idx = np.argmin(final_losses)
    ax.axhline(y=final_losses[best_idx], color='red', linestyle='--', alpha=0.5, label='Best')
    ax.legend()

    loss_comparison_path = Path(CONFIG['output_base_dir']) / "final_loss_comparison.png"
    plt.savefig(loss_comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {loss_comparison_path}")

    # Plot 3: Sample spectrograms (first, middle, last tests)
    # Use the magnitude spectrograms derived from the processed audio files
    test_indices = [0, len(all_results) // 2, len(all_results) - 1] if len(all_results) >= 3 else range(len(all_results))

    # Use the calculated difference magnitude spectrogram as the "target"
    fig, axes = plt.subplots(len(test_indices) + 2, 1, figsize=(14, 4 * (len(test_indices) + 2)))

    # Target: Calculated difference magnitude (approximation of vocal magnitude)
    axes[0].imshow(librosa.amplitude_to_db(diff_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target (|Mix - Instrumental| Mag)')
    axes[0].set_ylabel('Frequency')

    # Full Mix (reference)
    axes[1].imshow(librosa.amplitude_to_db(full_mix_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Full Mix (Reference)')
    axes[1].set_ylabel('Frequency')

    # Sample extracted results
    for plot_idx, result_idx in enumerate(test_indices):
        result = all_results[result_idx]
        audio_path = result['output_path']

        # Load and compute spectrogram
        audio, _ = librosa.load(audio_path, sr=CONFIG['sr'])
        stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
        mag, _ = librosa.magphase(stft)

        ax = axes[plot_idx + 2]
        ax.imshow(librosa.amplitude_to_db(mag, ref=np.max),
                  aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Estimated Vocal ({result['num_slices']} slice(s), loss={result['final_loss']:.4f})")
        ax.set_ylabel('Frequency')

        if plot_idx == len(test_indices) - 1:
            ax.set_xlabel('Time')

    plt.tight_layout()
    spectrogram_comparison_path = Path(CONFIG['output_base_dir']) / "spectrogram_comparison.png"
    plt.savefig(spectrogram_comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {spectrogram_comparison_path}")

    # ============================================
    # FINAL SUMMARY
    # ============================================

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("✓ PROGRESSIVE TEST (Difference-Based) COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"Tests completed: {len(all_results)}")
    print(f"\nResults summary:")

    for result in all_results:
        print(f"  {result['num_slices']:2d} slice(s): loss={result['final_loss']:.6f}, time={result['duration']:.1f}s")

    # Identify best result
    best_result = all_results[best_idx]
    print(f"\n🎵 Best result: {best_result['num_slices']} slice(s)")
    print(f"   Loss: {best_result['final_loss']:.6f}")
    print(f"   Path: {best_result['output_path']}")

    print(f"\n📁 All outputs saved in: {CONFIG['output_base_dir']}/")
    print(f"\n📊 Visualizations:")
    print(f"  • {loss_plot_path.name}")
    print(f"  • {loss_comparison_path.name}")
    print(f"  • {spectrogram_comparison_path.name}")

    print(f"\n🎧 Listen to the outputs to hear estimated vocals based on mix-instrumental difference!")
    print(f"\nEach test directory contains:")
    print(f"  • extracted_vocal.wav  ← Estimated vocal (from mix - instrumental target)")
    print(f"  • 1_original_full_mix.wav (in 1_slices/ only)")
    print(f"  • 2_original_instrumental.wav (in 1_slices/ only)")
    print(f"  • 3_calculated_difference.wav (in 1_slices/ only - estimated vocal magnitude with mix phase)")

    print("\n" + "="*70)