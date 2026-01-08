"""
CHAIN: RESOLUTION -> SKIP CONNECTION
====================================

This script chains Demo 1 (Resolution Change) and Demo 3 (Skip Connection).
It first applies a bottleneck (downsample/upsample) to the master audio,
then applies the skip connection concept by adding back the original detail.

Usage:
  python chain_resolution_then_skip.py

The script will:
  1. Load the master audio.
  2. Apply the bottleneck process from Demo 1 (e.g., downsample, upsample).
  3. Apply the skip connection process from Demo 3 (add original detail).
  4. Reconstruct and play the final audio.
  5. Compare against the original and the intermediate bottleneck result.
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
    'output_base_dir': 'output_chain_res_skip',
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
# HELPER FUNCTIONS (Copied from Demos 1 & 3)
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
    print("CHAIN: RESOLUTION -> SKIP CONNECTION")
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

    # Define downsample factors
    freq_factor = CONFIG['freq_pool_factor']
    time_factor = CONFIG['time_pool_factor']
    downsampled_shape = (original_shape[0] // freq_factor, original_shape[1] // time_factor)

    print(f"Bottleneck downsampling factors: Freq={freq_factor}x, Time={time_factor}x")
    print(f"Bottleneck downsampled shape: {downsampled_shape}")
    print()

    # --- CHAIN STEPS ---

    # 1. Apply Bottleneck (Downsample -> Process -> Upsample)
    print("--- STEP 1: APPLYING BOTTLENECK (DEMO 1 Concept) ---")
    print(f" 1a. Downsample master to {downsampled_shape}...")
    downsampled_mag = downsample_freq_time(master_mag, freq_factor, time_factor)

    print(f" 1b. Apply minimal processing in bottleneck (e.g., slight smoothing)...")
    bottleneck_processed = ndimage.gaussian_filter(downsampled_mag, sigma=0.5)

    print(f" 1c. Upsample bottleneck result back to {original_shape}...")
    upsampled_bottleneck_mag = upsample_freq_time_simple(bottleneck_processed, original_shape)

    print(f" 1d. Reconstruct audio from upsampled bottleneck (Intermediate Result)...")
    intermediate_audio = reconstruct_audio_from_magnitude(upsampled_bottleneck_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/01_intermediate_bottleneck.wav", intermediate_audio, CONFIG['sr'])
    print(f"     Saved: 01_intermediate_bottleneck.wav")
    print(f"     This audio is blurred due to information loss in bottleneck.")
    print()

    # 2. Apply Skip Connection (Add Original Detail to Bottleneck Result)
    print("--- STEP 2: APPLYING SKIP CONNECTION (DEMO 3 Concept) ---")
    print(f"     Combining 'Intermediate Bottleneck' result with 'Original Master' detail...")
    # Simple addition (standard U-Net skip connection method)
    combined_mag = upsampled_bottleneck_mag + master_mag # Add original detail
    # Optional: Normalize combined magnitude if values become too large
    combined_mag = np.clip(combined_mag, 0, np.max(master_mag) * 2) # Prevent extreme values

    print(f"     Reconstructing final audio from combined result (Bottleneck + Original Detail)...")
    final_audio = reconstruct_audio_from_magnitude(combined_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_final_result.wav", final_audio, CONFIG['sr'])
    print(f"     Saved: 02_final_result.wav")
    print(f"     This audio should sound sharper than '01_intermediate_bottleneck.wav',")
    print(f"     demonstrating how the skip connection restores detail after the bottleneck.")
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
    print("Listen to the files to hear the effect of chaining resolution change with skip connection.")
    print("  - 00_original.wav: The original master mix.")
    print("  - 01_intermediate_bottleneck.wav: Result of bottleneck process (blurred).")
    print("  - 02_final_result.wav: Result of adding original detail back to the bottleneck result (sharper than 01).")
    print("="*70)