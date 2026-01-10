"""
NORMALIZATION STRATEGY COMPARISON TEST
=======================================

This script tests different normalization approaches to see which produces
the best vocal separation quality.

Tests 5 slices with 4 normalization strategies:
1. Manual global normalization (current approach)
2. librosa.util.normalize default (global, norm=np.inf)
3. librosa.util.normalize axis=0 (per-frequency bin)
4. librosa.util.normalize axis=1 (per-time window)

Output: 4 audio files in output_normalization_test/
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
import time

print("="*70)
print("NORMALIZATION STRATEGY COMPARISON TEST")
print("="*70)
print("\nTesting 4 different normalization approaches")
print("Using 5 conv slices + slice_0_raw polish\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/100-window/isolated_vocal.wav',
    'mixture_path': 'rtg/100-window/stereo_mixture.wav',
    'output_dir': 'output_normalization_test',
    'num_slices': 5,
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
    magnitude = np.abs(stft)
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
    """Reconstruct audio from magnitude and phase"""
    reconstructed_stft = magnitude * np.exp(1j * phase)
    audio = librosa.istft(reconstructed_stft, hop_length=hop_length, n_fft=n_fft)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio


# ============================================
# NORMALIZATION STRATEGIES
# ============================================

def optimize_with_normalization(slice_name, vocal_fps, mixture_mag, num_windows, sr, current_mag, norm_strategy):
    """
    Optimize with specified normalization strategy.

    Args:
        norm_strategy: One of 'manual', 'librosa_global', 'librosa_axis0', 'librosa_axis1'
    """
    if current_mag is None:
        working_mag = mixture_mag.copy()
    else:
        working_mag = current_mag.copy()

    # Store original max for manual normalization
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

    # APPLY NORMALIZATION STRATEGY
    if norm_strategy == 'manual':
        # Strategy 1: Manual global normalization
        current_max = np.max(refined_mag)
        if current_max > 0:
            refined_mag = refined_mag / current_max * original_max

    elif norm_strategy == 'librosa_global':
        # Strategy 2: librosa.util.normalize (global, default)
        refined_mag = librosa.util.normalize(refined_mag) * original_max

    elif norm_strategy == 'librosa_axis0':
        # Strategy 3: librosa.util.normalize per-frequency bin (axis=0)
        # Normalize each frequency bin independently, then scale to original max
        refined_mag = librosa.util.normalize(refined_mag, axis=0) * original_max

    elif norm_strategy == 'librosa_axis1':
        # Strategy 4: librosa.util.normalize per-time window (axis=1)
        # Normalize each time window independently, then scale to original max
        refined_mag = librosa.util.normalize(refined_mag, axis=1) * original_max

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
    mixture_phase = np.angle(mixture_stft)

    # Slice order for testing (1-5)
    slice_names = [
        'slice_1_horizontal',
        'slice_2_vertical',
        'slice_3_diagonal_up',
        'slice_4_diagonal_down',
        'slice_5_blob',
    ]

    print(f"\n✓ Fingerprints created")

    # Save reference files once at the beginning
    sf.write(Path(CONFIG['output_dir']) / "0_original_mixture.wav", mixture_audio, CONFIG['sr'])
    sf.write(Path(CONFIG['output_dir']) / "0_target_vocal.wav", vocal_audio, CONFIG['sr'])
    print(f"\n✓ Saved reference files in {CONFIG['output_dir']}/")

    # ============================================
    # PHASE 2: TEST ALL 4 NORMALIZATION STRATEGIES
    # ============================================

    print("\n" + "="*70)
    print("PHASE 2: TEST ALL 4 NORMALIZATION STRATEGIES")
    print("="*70)
    print(f"\nUsing pipeline: {' → '.join(slice_names)} → slice_0_raw")
    print()

    results = {}
    strategies = [
        ('manual', 'Manual Global Normalization'),
        ('librosa_global', 'librosa.util.normalize (global)'),
        ('librosa_axis0', 'librosa.util.normalize (axis=0, per-freq)'),
        ('librosa_axis1', 'librosa.util.normalize (axis=1, per-time)'),
    ]

    for strategy_idx, (strategy_key, strategy_name) in enumerate(strategies, 1):
        print("\n" + "="*70)
        print(f"STRATEGY {strategy_idx}/4: {strategy_name}")
        print("="*70)

        current_mag = None

        # Process through 5 slices
        for i, slice_name in enumerate(slice_names, 1):
            print(f"\nPass {i}/{len(slice_names)}: {slice_name}")
            current_mag = optimize_with_normalization(
                slice_name, vocal_fps, mixture_mag, num_windows,
                CONFIG['sr'], current_mag, strategy_key
            )
            print(f"  ✓ Max magnitude: {np.max(current_mag):.2f}")

        # Final polish with slice_0
        print(f"\nFinal polish: slice_0_raw")
        final_mag = optimize_with_normalization(
            'slice_0_raw', vocal_fps, mixture_mag, num_windows,
            CONFIG['sr'], current_mag, strategy_key
        )
        print(f"  ✓ Max magnitude: {np.max(final_mag):.2f}")

        # Reconstruct audio
        audio = reconstruct_audio_from_magnitude(
            final_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft']
        )
        results[strategy_key] = audio

        # Save immediately
        output_path = Path(CONFIG['output_dir']) / f"{strategy_key}.wav"
        sf.write(output_path, audio, CONFIG['sr'])
        print(f"\n✓ Strategy {strategy_idx} complete!")
        print(f"  📁 Saved: {output_path}")
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
    strategy_titles = [name for _, name in strategies]

    for idx, ((strategy_key, _), title) in enumerate(zip(strategies, strategy_titles)):
        audio = results[strategy_key]
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
    print("✓ NORMALIZATION STRATEGY TEST COMPLETE!")
    print("="*70)

    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"\nOutput directory: {CONFIG['output_dir']}/")
    print(f"\nGenerated files:")
    print(f"  • manual.wav               ← Manual global normalization (current)")
    print(f"  • librosa_global.wav       ← librosa.util.normalize (global)")
    print(f"  • librosa_axis0.wav        ← librosa.util.normalize (per-frequency)")
    print(f"  • librosa_axis1.wav        ← librosa.util.normalize (per-time)")
    print(f"  • 0_original_mixture.wav   ← Reference")
    print(f"  • 0_target_vocal.wav       ← Reference")
    print(f"  • comparison_spectrograms.png  ← Visual comparison")

    print("\n🎧 Listen to all 4 outputs and compare:")
    print("   - Which one sounds clearest?")
    print("   - Which has best vocal isolation?")
    print("   - Which has least artifacts?")

    print("\n" + "="*70)


