"""
FREQUENCY & TIME RESOLUTION DEMO
=================================

This script demonstrates the effect of changing frequency and time resolution
on audio spectrograms, mimicking the core operations of U-Net encoder/decoder paths.

Usage:
  python freq_time_resolution_demo.py

The script will:
  1. Load the master audio.
  2. Show spectrograms and play audio for:
     - Original (full resolution)
     - Low Frequency Resolution (e.g., 4x freq pooling)
     - Low Time Resolution (e.g., 2x time pooling)
     - Low Freq & Time Resolution (e.g., 4x freq & 2x time pooling)
  3. Show spectrograms and play audio after upsampling the low-res versions back
     to the original size using simple methods.
  4. Highlight the loss of information during downsampling and the inability
     of upsampling alone to recover it.
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
    'output_base_dir': 'output_resolution_demo',
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
# HELPER FUNCTIONS
# ============================================

def downsample_freq_time(spectrogram, freq_factor, time_factor):
    """
    Downsample a spectrogram along frequency and time axes using max pooling.
    This version ensures the output dimensions are divisible by the factors.
    """
    orig_freq, orig_time = spectrogram.shape
    # Calculate target sizes based on factors
    target_freq = orig_freq // freq_factor
    target_time = orig_time // time_factor

    # Trim the spectrogram to make its dimensions divisible by the factors
    trimmed_spec = spectrogram[:target_freq * freq_factor, :target_time * time_factor]

    # Apply pooling
    pooled_freq = ndimage.maximum_filter(trimmed_spec, size=(freq_factor, 1))[::freq_factor, :]
    pooled_time = ndimage.maximum_filter(pooled_freq, size=(1, time_factor))[:, ::time_factor]

    return pooled_time

def upsample_freq_time_simple(spectrogram, target_shape):
    """
    Upsample a spectrogram back to target_shape using simple repeat_interleave.
    """
    freq_up_factor = target_shape[0] // spectrogram.shape[0]
    time_up_factor = target_shape[1] // spectrogram.shape[1]

    # Use repeat_interleave to upsample
    upsampled = np.repeat(spectrogram, freq_up_factor, axis=0)
    upsampled = np.repeat(upsampled, time_up_factor, axis=1)

    # Ensure the shape matches exactly (might need trimming/padding if not divisible)
    if upsampled.shape[0] > target_shape[0]:
        upsampled = upsampled[:target_shape[0], :]
    if upsampled.shape[1] > target_shape[1]:
        upsampled = upsampled[:, :target_shape[1]]

    # If still smaller, pad with the last value
    if upsampled.shape[0] < target_shape[0]:
        pad_size = target_shape[0] - upsampled.shape[0]
        upsampled = np.pad(upsampled, ((0, pad_size), (0, 0)), mode='edge')
    if upsampled.shape[1] < target_shape[1]:
        pad_size = target_shape[1] - upsampled.shape[1]
        upsampled = np.pad(upsampled, ((0, 0), (0, pad_size)), mode='edge')

    return upsampled

def reconstruct_audio_from_magnitude(magnitude, phase, hop_length, n_fft):
    """Reconstruct audio from magnitude and phase using librosa"""
    # Ensure magnitude matches phase shape for reconstruction
    if magnitude.shape != phase.shape:
        print(f"Warning: Magnitude shape {magnitude.shape} does not match phase shape {phase.shape}")
        # Crop or pad magnitude to match phase if necessary
        min_freq = min(magnitude.shape[0], phase.shape[0])
        min_time = min(magnitude.shape[1], phase.shape[1])
        magnitude = magnitude[:min_freq, :min_time]
        # Pad if needed
        target_freq, target_time = phase.shape
        if magnitude.shape[0] < target_freq:
             magnitude = np.pad(magnitude, ((0, target_freq - magnitude.shape[0]), (0, 0)), mode='edge')
        if magnitude.shape[1] < target_time:
             magnitude = np.pad(magnitude, ((0, 0), (0, target_time - magnitude.shape[1])), mode='edge')
        print(f"  Adjusted magnitude shape to match phase: {magnitude.shape}")

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
    print("FREQUENCY & TIME RESOLUTION DEMO")
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

    # Define downsample factors
    freq_factor = CONFIG['freq_pool_factor']
    time_factor = CONFIG['time_pool_factor']

    print(f"\nDownsampling factors: Freq={freq_factor}x, Time={time_factor}x")
    print(f"Target downsampled shape: ({original_shape[0]//freq_factor}, {original_shape[1]//time_factor})")
    print()

    # --- DEMO STEPS ---

    # 1. Original
    print("--- DEMO 1: ORIGINAL ---")
    print(f"Spectrogram shape: {original_shape}")
    print("Playing original master audio...")
    sf.write(f"{CONFIG['output_base_dir']}/01_original.wav", master_audio, CONFIG['sr'])
    # Show original spectrogram (optional visual confirmation)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.amplitude_to_db(master_mag, ref=np.max), sr=CONFIG['sr'], hop_length=CONFIG['hop_length'], x_axis='time', y_axis='linear')
    # plt.title('Original')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.savefig(f"{CONFIG['output_base_dir']}/01_original_spect.png")
    # plt.close()
    print("Saved: 01_original.wav")
    print()

    # 2. Low Frequency Resolution
    print("--- DEMO 2: LOW FREQUENCY RESOLUTION ---")
    low_freq_mag = downsample_freq_time(master_mag, freq_factor, 1) # Only pool freq
    print(f"Spectrogram shape after freq pooling: {low_freq_mag.shape}")
    # Upsample back to original shape for reconstruction
    upsampled_low_freq_mag = upsample_freq_time_simple(low_freq_mag, original_shape)
    print(f"Spectrogram shape after upsampling freq: {upsampled_low_freq_mag.shape}")
    low_freq_audio = reconstruct_audio_from_magnitude(upsampled_low_freq_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    print("Playing low frequency resolution audio (muffled)...")
    sf.write(f"{CONFIG['output_base_dir']}/02_low_freq_res.wav", low_freq_audio, CONFIG['sr'])
    print("Saved: 02_low_freq_res.wav")
    print()

    # 3. Low Time Resolution
    print("--- DEMO 3: LOW TIME RESOLUTION ---")
    low_time_mag = downsample_freq_time(master_mag, 1, time_factor) # Only pool time
    print(f"Spectrogram shape after time pooling: {low_time_mag.shape}")
    # Upsample back to original shape for reconstruction
    upsampled_low_time_mag = upsample_freq_time_simple(low_time_mag, original_shape)
    print(f"Spectrogram shape after upsampling time: {upsampled_low_time_mag.shape}")
    low_time_audio = reconstruct_audio_from_magnitude(upsampled_low_time_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    print("Playing low time resolution audio (temporally blurred)...")
    sf.write(f"{CONFIG['output_base_dir']}/03_low_time_res.wav", low_time_audio, CONFIG['sr'])
    print("Saved: 03_low_time_res.wav")
    print()

    # 4. Low Freq & Time Resolution
    print("--- DEMO 4: LOW FREQ & TIME RESOLUTION ---")
    low_both_mag = downsample_freq_time(master_mag, freq_factor, time_factor) # Pool both
    print(f"Spectrogram shape after both pooling: {low_both_mag.shape}")
    # Upsample back to original shape for reconstruction
    upsampled_low_both_mag = upsample_freq_time_simple(low_both_mag, original_shape)
    print(f"Spectrogram shape after upsampling both: {upsampled_low_both_mag.shape}")
    low_both_audio = reconstruct_audio_from_magnitude(upsampled_low_both_mag, master_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    print("Playing low freq & time resolution audio (muffled AND blurred)...")
    sf.write(f"{CONFIG['output_base_dir']}/04_low_both_res.wav", low_both_audio, CONFIG['sr'])
    print("Saved: 04_low_both_res.wav")
    print()

    # 5. Upsample Low Freq back (already done above for reconstruction)
    # Save the upsampled version directly
    sf.write(f"{CONFIG['output_base_dir']}/05_upsampled_low_freq.wav", low_freq_audio, CONFIG['sr']) # Audio from step 2
    print("--- DEMO 5: UPSAMPLE LOW FREQ BACK (Audio Saved) ---")
    print("Saved: 05_upsampled_low_freq.wav (This is the same as 02, but after upsampling)")
    print()

    # 6. Upsample Low Time back (already done above for reconstruction)
    sf.write(f"{CONFIG['output_base_dir']}/06_upsampled_low_time.wav", low_time_audio, CONFIG['sr']) # Audio from step 3
    print("--- DEMO 6: UPSAMPLE LOW TIME BACK (Audio Saved) ---")
    print("Saved: 06_upsampled_low_time.wav (This is the same as 03, but after upsampling)")
    print()

    # 7. Upsample Low Both back (already done above for reconstruction)
    sf.write(f"{CONFIG['output_base_dir']}/07_upsampled_low_both.wav", low_both_audio, CONFIG['sr']) # Audio from step 4
    print("--- DEMO 7: UPSAMPLE LOW FREQ & TIME BACK (Audio Saved) ---")
    print("Saved: 07_upsampled_low_both.wav (This is the same as 04, but after upsampling)")
    print()

    print("="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Output files saved in: {CONFIG['output_base_dir']}/")
    print("Listen to the files to hear the impact of resolution changes.")
    print("Notice how upsampling does not recover the detail lost during downsampling.")
    print("="*70)