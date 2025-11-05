"""
CHAIN: BEST PROGRESSIVE -> SKIP CONNECTION
==========================================

This script takes the output magnitude from a progressive refinement (like best.py) and applies
the skip connection concept (Demo 3) by adding back the original mixture magnitude.

Usage:
  python chain_best_to_skip.py
  (Assumes you have run best.py and have an output file like 18_slices/extracted_vocal.wav)

The script will:
  1. Load the *refined* audio and its magnitude spectrogram from best.py.
  2. Load the *original mixture* magnitude spectrogram.
  3. Add the original mixture magnitude to the refined magnitude.
  4. Reconstruct the final audio using the original mixture phase.
  5. Compare against the original refined input and the final result.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import time

# ============================================
# CONFIGURATION (Assumes files are in 'rtg/100-window/' and output_progressive_18slices/)
# ============================================

CONFIG = {
    # Input is the output from best.py
    'refined_input_path': 'output_progressive_8slices/8_slices/extracted_vocal.wav', # The result from best.py
    'original_mixture_path': 'rtg/100-window/stereo_mixture.wav', # Need original magnitude and phase
    'original_vocal_path': 'rtg/100-window/isolated_vocal.wav', # For comparison
    'output_base_dir': 'output_chain_best_skip',
    'sr': 22050,
    'duration': 4.7, # Fixed duration to match your baseline
    'n_fft': 2048,
    'hop_length': 1040, # Ensures ~100 windows
    # No specific parameters needed for skip connection itself
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print("CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================
# HELPER FUNCTIONS (Copied from Demo 1 for reconstruction)
# ============================================

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
    print("CHAIN: BEST PROGRESSIVE -> SKIP CONNECTION")
    print("="*70)
    print(f"Loading refined input: {CONFIG['refined_input_path']}")
    print(f"Loading original mixture: {CONFIG['original_mixture_path']}")
    print(f"Target duration: {CONFIG['duration']}s, SR: {CONFIG['sr']}, N_FFT: {CONFIG['n_fft']}, Hop: {CONFIG['hop_length']}")
    print()

    # Load the *refined* audio from best.py
    refined_audio, _ = librosa.load(CONFIG['refined_input_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    print(f"Loaded refined audio: {len(refined_audio)} samples")

    # Load the *original mixture* to get its magnitude and phase
    mixture_audio, _ = librosa.load(CONFIG['original_mixture_path'], sr=CONFIG['sr'], duration=CONFIG['duration'])
    mixture_stft = librosa.stft(mixture_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    mixture_mag, mixture_phase = librosa.magphase(mixture_stft)
    original_shape = mixture_phase.shape
    print(f"Original mixture phase shape (used for reconstruction): {original_shape}")
    print(f"Original mixture magnitude shape (for skip): {mixture_mag.shape}")
    print()

    # Create STFT of the *refined* audio to get its magnitude
    refined_stft = librosa.stft(refined_audio, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    refined_mag, _ = librosa.magphase(refined_stft) # We only need the magnitude
    refined_shape = refined_mag.shape
    print(f"Refined audio spectrogram magnitude shape: {refined_shape}")
    print()

    # --- CHAIN STEPS ---

    # 1. Apply Skip Connection Concept: Add Original Mixture Mag to Refined Mag
    print("--- STEP 1: APPLYING SKIP CONNECTION (DEMO 3 Concept) ---")
    print(f"     Adding original mixture magnitude (skip connection) to refined magnitude...")
    # Ensure shapes match before addition (they should if derived from same STFT params)
    if refined_shape != mixture_mag.shape:
        print(f"Warning: Refined mag shape {refined_shape} != Mixture mag shape {mixture_mag.shape}")
        # Crop or pad if necessary, assuming mixture is the reference shape
        min_freq = min(refined_mag.shape[0], mixture_mag.shape[0])
        min_time = min(refined_mag.shape[1], mixture_mag.shape[1])
        refined_mag = refined_mag[:min_freq, :min_time]
        mixture_mag = mixture_mag[:min_freq, :min_time]
        print(f"  Adjusted shapes to match: {refined_mag.shape}, {mixture_mag.shape}")

    combined_mag = refined_mag + mixture_mag # Add original detail (mixture) to refined result
    # Optional: Normalize combined magnitude if values become too large
    combined_mag = np.clip(combined_mag, 0, np.max([refined_mag, mixture_mag]) * 2) # Prevent extreme values
    print(f"     Combined magnitude shape: {combined_mag.shape}")
    print()

    # 2. Reconstruct Final Audio using Original Mixture Phase
    print("--- STEP 2: RECONSTRUCTING FINAL AUDIO ---")
    print(f"     Reconstructing final audio using combined magnitude and original mixture phase...")
    final_audio = reconstruct_audio_from_magnitude(combined_mag, mixture_phase, CONFIG['hop_length'], CONFIG['n_fft'])
    sf.write(f"{CONFIG['output_base_dir']}/02_final_after_skip.wav", final_audio, CONFIG['sr'])
    print(f"     Saved: 02_final_after_skip.wav")
    print(f"     This audio is the result of adding original mixture magnitude to the refined output.")
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
    print("Listen to the files to hear the effect of adding original mixture detail to the progressive refinement output.")
    print("  - 00_original_mixture.wav: The original mix.")
    print("  - 01_refined_input.wav: The refined vocal from best.py (input to this chain).")
    print("  - 02_final_after_skip.wav: The refined vocal after adding original mixture magnitude (skip connection concept).")
    print("  - 03_target_vocal.wav: The target vocal.")
    print("="*70)