"""
GAIN STRATEGY COMPARISON TEST
==============================

This script tests 4 different strategies for handling gain reduction
during progressive vocal separation.

Tests 5 slices only, outputs 4 audio files for comparison.

Strategies:
1. Normalize after each pass (prevent cumulative loss)
2. Track total gain and compensate at end
3. dB-space EQ (additive instead of multiplicative)
4. Symmetric clipping range (0.333-3.0 instead of 0.1-3.0)

Output: 4 audio files in output_gain_test/
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
import time

print("="*70)
print("GAIN STRATEGY COMPARISON TEST")
print("="*70)
print("\nTesting 4 different gain-handling strategies")
print("Using 5 conv slices + slice_0_raw polish\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/100-window/isolated_vocal.wav',
    'mixture_path': 'rtg/100-window/stereo_mixture.wav',
    'output_dir': 'output_gain_test',
    'num_slices': 5,  # Test with 5 slices only
    'sr': 22050,
    'duration': 4.7,
    'n_fft': 2048,
    'hop_length': 1024,
    'num_iterations': 100,
    'learning_rate': 0.01,
}

Path(CONFIG['output_dir']).mkdir(exist_ok=True, parents=True)

print(f"Configuration:")
print(f"  Testing {CONFIG['num_slices']} conv slices")
print(f"  Iterations per slice: {CONFIG['num_iterations']}")
print(f"  Output directory: {CONFIG['output_dir']}/")
print()


# ============================================
# HELPER FUNCTIONS (from sanity_check_progressive.py)
# ============================================

def apply_2d_conv(image, kernel):
    """Apply 2D convolution to spectrogram"""
    return ndimage.convolve(image, kernel, mode='constant', cval=0.0)

def create_5_slices(magnitude_spectrogram):
    """Create first 5 slices for testing"""
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

    return slices

def window_to_bottleneck(window, sr):
    """Extract 400-point frequency profile directly from window"""
    with np.errstate(divide='ignore', invalid='ignore'):
        # Directly interpolate to 400 points (no encoder/decoder needed)
        freq_profile_400 = np.interp(
            x=np.linspace(0, sr/2, 400),
            xp=np.linspace(0, sr/2, len(window)),
            fp=window
        )

        return {
            'freq_profile_400': np.nan_to_num(freq_profile_400, nan=0.0, posinf=0.0, neginf=0.0),
        }

def process_audio_to_fingerprints(audio_path, sr, n_fft, hop_length):
    """Load audio and create fingerprints for 5 slices"""
    print(f"\n[Processing: {Path(audio_path).name}]")

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=CONFIG['duration'])
    print(f"  Loaded {len(audio)} samples")

    # Create STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # Use librosa.magphase for safer magnitude extraction
    magnitude, phase = librosa.magphase(stft)
    num_windows = magnitude.shape[1]
    print(f"  Spectrogram: {magnitude.shape} ({num_windows} windows)")

    # Create 5 slices
    print(f"  Creating 5 spectral slices...")
    slices = create_5_slices(magnitude)

    # Process each slice to bottleneck fingerprints
    fingerprints = {}

    for slice_name, slice_data in slices.items():
        slice_fingerprints = []
        num_slice_windows = slice_data.shape[1]

        for window_idx in range(num_slice_windows):
            window = slice_data[:, window_idx]
            metrics = window_to_bottleneck(window, sr)
            slice_fingerprints.append(metrics)

        fingerprints[slice_name] = slice_fingerprints

    print(f"  ✓ Created fingerprints for {len(fingerprints)} slices")

    return audio, stft, magnitude, fingerprints

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """Reconstruct audio from magnitude and phase using librosa"""
    # Use librosa.util.phasor for safer complex reconstruction
    # Note: phase here is already the complex phasor from librosa.magphase
    reconstructed_stft = magnitude * phase
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    # Keep manual normalization (sounds best based on testing)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio


# ============================================
# STRATEGY 1: NORMALIZE AFTER EACH PASS
# ============================================

def optimize_strategy1_normalize(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Strategy 1: Normalize after each pass to prevent cumulative gain loss
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Store original max for normalization reference
    original_max = np.max(working_mag)

    # Initialize EQ curves
    eq_curves = [np.ones(400) for _ in range(num_windows)]

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        for win_idx in range(num_windows):
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            mixture_window = working_mag[:, win_idx]

            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )

            adjusted_fp = mixture_fp * eq_curves[win_idx]
            loss = np.mean((adjusted_fp - vocal_fp)**2)
            total_loss += loss

            gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp
            eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

    # Apply EQ
    refined_mag = np.zeros_like(working_mag)
    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
        refined_mag[:, win_idx] = window_mag * eq_curve_full

    # STRATEGY 1: Normalize to maintain consistent level
    current_max = np.max(refined_mag)
    if current_max > 0:
        refined_mag = refined_mag / current_max * original_max

    return refined_mag


# ============================================
# STRATEGY 2: TRACK TOTAL GAIN AND COMPENSATE
# ============================================

def optimize_strategy2_compensate(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None, accumulated_gains=None):
    """
    Strategy 2: Track cumulative gain changes and compensate at the end
    Returns (refined_mag, accumulated_gains)
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    if accumulated_gains is None:
        accumulated_gains = [np.ones(400) for _ in range(num_windows)]

    # Initialize EQ curves
    eq_curves = [np.ones(400) for _ in range(num_windows)]

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        for win_idx in range(num_windows):
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            mixture_window = working_mag[:, win_idx]

            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )

            adjusted_fp = mixture_fp * eq_curves[win_idx]
            loss = np.mean((adjusted_fp - vocal_fp)**2)
            total_loss += loss

            gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp
            eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.1, 3.0)

    # Apply EQ
    refined_mag = np.zeros_like(working_mag)
    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
        refined_mag[:, win_idx] = window_mag * eq_curve_full

        # STRATEGY 2: Accumulate gains
        eq_curve_400 = eq_curves[win_idx]
        accumulated_gains[win_idx] = accumulated_gains[win_idx] * eq_curve_400

    return refined_mag, accumulated_gains


# ============================================
# STRATEGY 3: DB-SPACE EQ (ADDITIVE)
# ============================================

def optimize_strategy3_db_space(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Strategy 3: Work in dB space (additive) instead of linear space (multiplicative)
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Convert to dB using librosa (safer, handles edge cases better)
    working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)

    # Initialize EQ curves (dB adjustments, start at 0)
    eq_curves_db = [np.zeros(400) for _ in range(num_windows)]

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        for win_idx in range(num_windows):
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            # Convert target to dB using librosa
            vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)

            mixture_window_db = working_mag_db[:, win_idx]

            mixture_fp_db = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window_db)),
                fp=mixture_window_db
            )

            # Apply EQ (additive in dB space)
            adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]

            # Loss in dB space
            loss = np.mean((adjusted_fp_db - vocal_fp_db)**2)
            total_loss += loss

            # Gradient (simpler in dB space - additive)
            gradient = 2 * (adjusted_fp_db - vocal_fp_db)

            # Update EQ curve
            eq_curves_db[win_idx] -= CONFIG['learning_rate'] * gradient

            # STRATEGY 3: Clip in dB space (symmetric)
            eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)  # ±20dB

    # Apply EQ in dB space
    refined_mag_db = np.zeros_like(working_mag_db)
    for win_idx in range(num_windows):
        window_mag_db = working_mag_db[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag_db))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_db_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves_db[win_idx])
        refined_mag_db[:, win_idx] = window_mag_db + eq_curve_db_full

    # Convert back to linear using librosa
    refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
    refined_mag = np.nan_to_num(refined_mag, nan=0.0, posinf=0.0, neginf=0.0)

    return refined_mag


# ============================================
# STRATEGY 4: SYMMETRIC CLIPPING RANGE
# ============================================

def optimize_strategy4_symmetric(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag=None):
    """
    Strategy 4: Use symmetric clipping range [0.333, 3.0] = [-10dB, +10dB]
    Instead of [0.1, 3.0] = [-20dB, +9.5dB]
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Initialize EQ curves
    eq_curves = [np.ones(400) for _ in range(num_windows)]

    # Optimization loop
    for iteration in range(CONFIG['num_iterations']):
        total_loss = 0.0

        for win_idx in range(num_windows):
            vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
            mixture_window = working_mag[:, win_idx]

            mixture_fp = np.interp(
                x=np.linspace(0, sr/2, 400),
                xp=np.linspace(0, sr/2, len(mixture_window)),
                fp=mixture_window
            )

            adjusted_fp = mixture_fp * eq_curves[win_idx]
            loss = np.mean((adjusted_fp - vocal_fp)**2)
            total_loss += loss

            gradient = 2 * (adjusted_fp - vocal_fp) * mixture_fp
            eq_curves[win_idx] -= CONFIG['learning_rate'] * gradient

            # STRATEGY 4: Symmetric clipping in dB space
            # [0.333, 3.0] ≈ [-10dB, +10dB]
            eq_curves[win_idx] = np.clip(eq_curves[win_idx], 0.333, 3.0)

    # Apply EQ
    refined_mag = np.zeros_like(working_mag)
    for win_idx in range(num_windows):
        window_mag = working_mag[:, win_idx]
        freq_bins_stft = np.linspace(0, sr/2, len(window_mag))
        freq_points_eq = np.linspace(0, sr/2, 400)
        eq_curve_full = np.interp(freq_bins_stft, freq_points_eq, eq_curves[win_idx])
        refined_mag[:, win_idx] = window_mag * eq_curve_full

    return refined_mag


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
    # Extract phase as complex phasor using librosa.magphase
    _, mixture_phase = librosa.magphase(mixture_stft)

    # Slice order for testing (1-5)
    slice_names = [
        'slice_1_horizontal',
        'slice_2_vertical',
        'slice_3_diagonal_up',
        'slice_4_diagonal_down',
        'slice_5_blob',
    ]

    print(f"\n✓ Fingerprints created")

    # ============================================
    # PHASE 2: TEST ALL 4 STRATEGIES
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: TEST ALL 4 STRATEGIES")
    print("="*70)
    print(f"\nUsing pipeline: {' → '.join(slice_names)} → slice_0_raw")
    print()

    results = {}

    # Save reference files once at the beginning
    sf.write(Path(CONFIG['output_dir']) / "0_original_mixture.wav", mixture_audio, CONFIG['sr'])
    sf.write(Path(CONFIG['output_dir']) / "0_target_vocal.wav", vocal_audio, CONFIG['sr'])
    print(f"\n✓ Saved reference files in {CONFIG['output_dir']}/")

    # ============================================
    # TEST 1: NORMALIZE AFTER EACH PASS
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 1/4: NORMALIZE AFTER EACH PASS")
    print("="*70)

    current_mag = None
    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag = optimize_strategy1_normalize(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish with slice_0
    print(f"\nFinal polish: slice_0_raw")
    final_mag = optimize_strategy1_normalize(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    audio1 = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy1_normalize'] = audio1

    # SAVE IMMEDIATELY
    output_path1 = Path(CONFIG['output_dir']) / "strategy1_normalize.wav"
    sf.write(output_path1, audio1, CONFIG['sr'])
    print(f"\n✓ Strategy 1 complete!")
    print(f"  📁 Saved: {output_path1}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # TEST 2: TRACK GAIN AND COMPENSATE
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 2/4: TRACK GAIN AND COMPENSATE")
    print("="*70)

    current_mag = None
    accumulated_gains = None

    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag, accumulated_gains = optimize_strategy2_compensate(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag, accumulated_gains
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish
    print(f"\nFinal polish: slice_0_raw")
    final_mag, accumulated_gains = optimize_strategy2_compensate(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag, accumulated_gains
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    # Apply inverse gain compensation
    print("\nApplying gain compensation...")
    compensated_mag = np.zeros_like(final_mag)
    for win_idx in range(num_windows):
        window_mag = final_mag[:, win_idx]

        # Calculate average accumulated gain
        avg_gain = np.mean(accumulated_gains[win_idx])

        # Apply inverse
        if avg_gain > 0:
            compensation_factor = 1.0 / avg_gain
            freq_bins_stft = np.linspace(0, CONFIG['sr']/2, len(window_mag))
            freq_points_eq = np.linspace(0, CONFIG['sr']/2, 400)
            comp_curve = np.ones(400) * compensation_factor
            comp_curve_full = np.interp(freq_bins_stft, freq_points_eq, comp_curve)
            compensated_mag[:, win_idx] = window_mag * comp_curve_full
        else:
            compensated_mag[:, win_idx] = window_mag

    print(f"  ✓ Compensated max magnitude: {np.max(compensated_mag):.2f}")

    audio2 = reconstruct_audio_from_magnitude(compensated_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy2_compensate'] = audio2

    # SAVE IMMEDIATELY
    output_path2 = Path(CONFIG['output_dir']) / "strategy2_compensate.wav"
    sf.write(output_path2, audio2, CONFIG['sr'])
    print(f"\n✓ Strategy 2 complete!")
    print(f"  📁 Saved: {output_path2}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # TEST 3: DB-SPACE EQ
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 3/4: DB-SPACE EQ")
    print("="*70)

    current_mag = None
    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag = optimize_strategy3_db_space(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish
    print(f"\nFinal polish: slice_0_raw")
    final_mag = optimize_strategy3_db_space(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    audio3 = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy3_dbspace'] = audio3

    # SAVE IMMEDIATELY
    output_path3 = Path(CONFIG['output_dir']) / "strategy3_dbspace.wav"
    sf.write(output_path3, audio3, CONFIG['sr'])
    print(f"\n✓ Strategy 3 complete!")
    print(f"  📁 Saved: {output_path3}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # TEST 4: SYMMETRIC CLIPPING
    # ============================================

    print("\n" + "="*70)
    print("STRATEGY 4/4: SYMMETRIC CLIPPING RANGE")
    print("="*70)

    current_mag = None
    for i, slice_name in enumerate(slice_names, 1):
        print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
        current_mag = optimize_strategy4_symmetric(
            slice_name, vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
        )
        print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

    # Final polish
    print(f"\nFinal polish: slice_0_raw")
    final_mag = optimize_strategy4_symmetric(
        'slice_0_raw', vocal_fps, mixture_mag, num_windows, CONFIG['sr'], current_mag
    )
    print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

    audio4 = reconstruct_audio_from_magnitude(final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    results['strategy4_symmetric'] = audio4

    # SAVE IMMEDIATELY
    output_path4 = Path(CONFIG['output_dir']) / "strategy4_symmetric.wav"
    sf.write(output_path4, audio4, CONFIG['sr'])
    print(f"\n✓ Strategy 4 complete!")
    print(f"  📁 Saved: {output_path4}")
    print(f"  🎧 Ready to listen!")

    # ============================================
    # PHASE 3: VISUALIZATION
    # ============================================

    print("\n" + "="*70)
    print("PHASE 3: CREATING COMPARISON VISUALIZATION")
    print("="*70)

    print("\nGenerating comparison spectrograms...")

    fig, axes = plt.subplots(6, 1, figsize=(14, 18))

    # Target vocal
    axes[0].imshow(librosa.amplitude_to_db(vocal_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target Vocal (Reference)')
    axes[0].set_ylabel('Frequency')

    # Original mixture
    axes[1].imshow(librosa.amplitude_to_db(mixture_mag, ref=np.max),
                   aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Original Mixture')
    axes[1].set_ylabel('Frequency')

    # All 4 strategies
    strategy_titles = [
        'Strategy 1: Normalize After Each Pass',
        'Strategy 2: Track Gain & Compensate',
        'Strategy 3: dB-Space EQ',
        'Strategy 4: Symmetric Clipping [0.333, 3.0]'
    ]

    for idx, (strategy_name, title) in enumerate(zip(results.keys(), strategy_titles)):
        audio = results[strategy_name]
        stft = librosa.stft(audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
        mag = np.abs(stft)

        axes[idx + 2].imshow(librosa.amplitude_to_db(mag, ref=np.max),
                            aspect='auto', origin='lower', cmap='viridis')
        axes[idx + 2].set_title(title)
        axes[idx + 2].set_ylabel('Frequency')

    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    viz_path = Path(CONFIG['output_dir']) / "comparison_spectrograms.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {viz_path}")

    # ============================================
    # FINAL SUMMARY
    # ============================================

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("✓ GAIN STRATEGY TEST COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"\nOutput directory: {CONFIG['output_dir']}/")
    print(f"\nGenerated files:")
    print(f"  • strategy1_normalize.wav       ← Normalize after each pass")
    print(f"  • strategy2_compensate.wav      ← Track gain & compensate")
    print(f"  • strategy3_dbspace.wav         ← dB-space EQ")
    print(f"  • strategy4_symmetric.wav       ← Symmetric clipping")
    print(f"  • 0_original_mixture.wav        ← Reference")
    print(f"  • 0_target_vocal.wav            ← Reference")
    print(f"  • comparison_spectrograms.png   ← Visual comparison")

    print("\n🎧 Listen to all 4 outputs and compare:")
    print("   - Which one sounds loudest/clearest?")
    print("   - Which one best preserves vocal quality?")
    print("   - Which one has least artifacts?")

    print("\n" + "="*70)


