"""
CHAIN: BEST PROGRESSIVE -> RESOLUTION CHANGE
============================================

This script takes the output from a progressive refinement (like best.py) and applies
the resolution change process (Demo 1) to it.

Usage:
  python chain_best_to_resolution.py
  (Assumes you have run best.py and have an output file like 18_slices/extracted_vocal.wav)

The script will:
  1. Load the *refined* audio from best.py (or a similar process).
  2. Apply the bottleneck process from Demo 1 (downsample, process, upsample) to this refined audio.
  3. Reconstruct and play the final audio.
  5. Compare against the original refined input and the final result.
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
    'output_base_dir': 'output_chain_best_res',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    'freq_pool_factor': 4, # Factor to downsample frequency axis
    'time_pool_factor': 2,  # Factor to downsample time axis
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS (Copied from Demo 1)
# ============================================

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
    print("CHAIN: BEST PROGRESSIVE -> RESOLUTION CHANGE")
    print("="*70)
    print(f"Loading refined input: {CONFIG['refined_input_path']}")
    print(f"Loading original mixture phase: {CONFIG['original_mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print()

    # Load the *refined* audio from best.py
    refined_audio, _ = librosa.load(CONFIG['refined_input_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    print(f"Loaded refined audio: {len(refined_audio)} samples")

    # Load the *original mixture* to get its phase
    mixture_audio, _ = librosa.load(CONFIG['original_mixture_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    mixture_stft = librosa.stft(mixture_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    _, mixture_phase = librosa.magphase(mixture_stft) # We only need the phase
    original_shape = mixture_phase.shape # Get the shape from the original phase
    print(f"Original mixture phase shape (used for reconstruction): {original_shape}")
    print()

    # Create STFT of the *refined* audio to get its magnitude
    refined_stft = librosa.stft(refined_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    refined_mag, _ = librosa.magphase(refined_stft) # We only need the magnitude
    refined_shape = refined_mag.shape
    print(f"Refined audio spectrogram magnitude shape: {refined_shape}")
    print()

    # Define downsample factors
    freq_factor = CONFIG['freq_pool_factor']
    time_factor = CONFIG['time_pool_factor']
    downsampled_shape = (refined_shape[0] // freq_factor, refined_shape[1] // time_factor)

    print(f"Bottleneck downsampling factors: Freq={freq_factor}x, Time={time_factor}x")
    print(f"Bottleneck downsampled shape: {downsampled_shape}")
    print()

    # --- CHAIN STEPS ---

    # 1. Apply Bottleneck to the Refined Audio (Downsample -> Process -> Upsample)
    print("--- STEP 1: APPLYING BOTTLENECK TO REFINED AUDIO (DEMO 1 Concept) ---")
    print(f" 1a. Downsample refined audio magnitude to {downsampled_shape}...")
    downsampled_refined_mag = downsample_freq_time(refined_mag, freq_factor, time_factor)

    print(f" 1b. Apply minimal processing in bottleneck (e.g., slight smoothing)...")
    bottleneck_processed = ndimage.gaussian_filter(downsampled_refined_mag, sigma=0.5)

    print(f" 1c. Upsample bottleneck result back to original refined shape {refined_shape}...")
    upsampled_bottleneck_mag = upsample_freq_time_simple(bottleneck_processed, refined_shape)

    print(f" 1d. Reconstruct audio from upsampled bottleneck result (Final Result)...")
    # Use the *original mixture phase* for reconstruction, as in other demos
    final_audio = reconstruct_audio_from_magnitude(upsampled_bottleneck_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_final_after_resolution_change.wav", final_audio, CONFIG['sr'])
    print(f"     Saved: 02_final_after_resolution_change.wav")
    print(f"     This audio is the refined signal after a resolution change bottleneck.")
    print()

    # 2. Save the Refined Input for Comparison
    print("--- STEP 2: SAVING REFINED INPUT FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/01_refined_input.wav", refined_audio, CONFIG['sr'])
    print(f"     Saved: 01_refined_input.wav (The audio from best.py)")
    print()

    # 3. Save the Original Mixture and Target for Comparison
    print("--- STEP 3: SAVING ORIGINALS FOR COMPARISON ---")
    sf.write(f"{CONFIG['output_base_dir']}/00_original_mixture.wav", mixture_audio, CONFIG['sr'])
    target_vocal, _ = librosa.load(CONFIG['original_vocal_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    sf.write(f"{CONFIG['output_base_dir']}/03_target_vocal.wav", target_vocal, CONFIG['sr'])
    print(f"     Saved: 00_original_mixture.wav, 03_target_vocal.wav")
    print()

    print("="*70)
    print("CHAIN COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the effect of applying resolution change to the progressive refinement output.")
    print("  - 00_original_mixture.wav: The original mix.")
    print("  - 01_refined_input.wav: The refined vocal from best.py (input to this chain).")
    print("  - 02_final_after_resolution_change.wav: The refined vocal after a bottleneck/res change.")
    print("  - 03_target_vocal.wav: The target vocal.")
    print("="*70)