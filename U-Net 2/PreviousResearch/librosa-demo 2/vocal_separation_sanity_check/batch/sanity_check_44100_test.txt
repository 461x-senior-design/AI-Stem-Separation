"""
TEST: 44100 Hz VERSION
======================

Testing if the algorithm works properly at 44100 Hz sample rate.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("44100 Hz TEST VERSION")
print("="*80)

# ============================================
# CONFIGURATION - 44100 Hz
# ============================================

VOCAL_PATH = 'rtg/100-window/isolated_vocal-44.wav'
MIXTURE_PATH = 'rtg/100-window/stereo_mixture-44.wav'
OUTPUT_DIR = 'output/test_44100'
SAMPLE_RATE = 22050  # TESTING 44100 Hz
DURATION = 4.7
N_FFT = 2048
WINDOW_SPACING_MS = 47
HOP_LENGTH = int(SAMPLE_RATE * WINDOW_SPACING_MS / 1000)
LEARNING_RATE = 0.01
NUM_ITERATIONS = 100

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

print(f"\nConfiguration:")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Nyquist frequency: {SAMPLE_RATE/2} Hz")
print(f"  Hop length: {HOP_LENGTH} samples")
print(f"  Window spacing: {WINDOW_SPACING_MS} ms")
print(f"  Learning rate: {LEARNING_RATE}")

# ============================================
# LOAD AUDIO
# ============================================

print("\n" + "="*80)
print("LOADING AUDIO")
print("="*80)

# Load acapella
print("\nLoading acapella...")
acapella_audio, _ = librosa.load(VOCAL_PATH, sr=SAMPLE_RATE, duration=DURATION)
print(f"  Loaded {len(acapella_audio)} samples ({len(acapella_audio)/SAMPLE_RATE:.2f} seconds)")

# Create STFT
print("Creating acapella STFT...")
acapella_stft = librosa.stft(acapella_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
acapella_magnitude = np.abs(acapella_stft)
acapella_phase = np.angle(acapella_stft)
print(f"  STFT shape: {acapella_stft.shape}")
print(f"  Frequency bins: {acapella_stft.shape[0]} (0 to {SAMPLE_RATE/2} Hz)")
print(f"  Time windows: {acapella_stft.shape[1]}")
print(f"  Magnitude range: [{np.min(acapella_magnitude):.3f}, {np.max(acapella_magnitude):.3f}]")

# Load mixture
print("\nLoading mixture...")
mixture_audio, _ = librosa.load(MIXTURE_PATH, sr=SAMPLE_RATE, duration=DURATION)
print(f"  Loaded {len(mixture_audio)} samples ({len(mixture_audio)/SAMPLE_RATE:.2f} seconds)")

# Create STFT
print("Creating mixture STFT...")
mixture_stft = librosa.stft(mixture_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
mixture_magnitude = np.abs(mixture_stft)
mixture_phase = np.angle(mixture_stft)
print(f"  STFT shape: {mixture_stft.shape}")
print(f"  Magnitude range: [{np.min(mixture_magnitude):.3f}, {np.max(mixture_magnitude):.3f}]")

current_mixture_magnitude = mixture_magnitude.copy()

# Save originals
sf.write(f"{OUTPUT_DIR}/original_mixture.wav", mixture_audio, SAMPLE_RATE)
sf.write(f"{OUTPUT_DIR}/target_acapella.wav", acapella_audio, SAMPLE_RATE)

# ============================================
# FINGERPRINT EXTRACTION (FREQUENCY AWARE)
# ============================================

def extract_425_point_fingerprint(spectrogram_window, sr):
    """Extract 425 metrics - properly scaled for sample rate"""

    # Downsample progressively
    layer1 = np.zeros(len(spectrogram_window) // 2)
    for i in range(len(layer1)):
        layer1[i] = np.max(spectrogram_window[i*2:(i+1)*2])

    layer2 = np.zeros(len(layer1) // 2)
    for i in range(len(layer2)):
        layer2[i] = np.max(layer1[i*2:(i+1)*2])

    layer3 = np.zeros(len(layer2) // 2)
    for i in range(len(layer3)):
        layer3[i] = np.max(layer2[i*2:(i+1)*2])

    layer4 = np.zeros(len(layer3) // 2)
    for i in range(len(layer4)):
        layer4[i] = np.max(layer3[i*2:(i+1)*2])

    bottleneck = np.zeros(len(layer4) // 2)
    for i in range(len(bottleneck)):
        bottleneck[i] = np.max(layer4[i*2:(i+1)*2])

    # Extract 400-point frequency profile
    freq_profile_400 = np.interp(
        np.linspace(0, sr/2, 400),
        np.linspace(0, sr/2, len(layer2)),
        layer2
    )

    # Calculate frequency band energies (automatically scaled to sr/2)
    num_bins = len(layer2)
    bass_end = max(1, int(num_bins * 250/(sr/2)))
    mid_end = int(num_bins * 2000/(sr/2))

    bass_energy = np.sum(layer2[0:bass_end]**2) + 1e-8
    mid_energy = np.sum(layer2[bass_end:mid_end]**2) + 1e-8
    high_energy = np.sum(layer2[mid_end:]**2) + 1e-8

    # Build 425-point fingerprint
    fingerprint = np.zeros(425)
    fingerprint[0:400] = freq_profile_400
    fingerprint[400] = bass_energy
    fingerprint[401] = mid_energy
    fingerprint[402] = high_energy

    return np.nan_to_num(fingerprint, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================
# PROCESS ONE SLICE
# ============================================

def process_one_slice(slice_name, kernel, acapella_mag, current_mixture_mag):
    """Process one Conv2D slice"""

    print(f"\n" + "="*80)
    print(f"PROCESSING: {slice_name}")
    print(f"="*80)

    # Apply Conv2D to acapella
    print(f"\nApplying Conv2D to acapella...")
    acapella_conv2d = ndimage.convolve(acapella_mag, kernel, mode='constant', cval=0.0)
    print(f"  Conv2D range: [{np.min(acapella_conv2d):.3f}, {np.max(acapella_conv2d):.3f}]")

    # Extract fingerprints from acapella
    print(f"Extracting fingerprints from acapella...")
    acapella_fingerprints = []
    num_windows = acapella_conv2d.shape[1]

    for win_idx in range(num_windows):
        window = acapella_conv2d[:, win_idx]
        fingerprint = extract_425_point_fingerprint(window, SAMPLE_RATE)
        acapella_fingerprints.append(fingerprint)

    print(f"  Extracted {len(acapella_fingerprints)} fingerprints")

    # Apply Conv2D to mixture
    print(f"Applying Conv2D to mixture...")
    mixture_conv2d = ndimage.convolve(current_mixture_mag, kernel, mode='constant', cval=0.0)
    print(f"  Conv2D range: [{np.min(mixture_conv2d):.3f}, {np.max(mixture_conv2d):.3f}]")

    # Extract fingerprints from mixture
    print(f"Extracting fingerprints from mixture...")
    mixture_fingerprints = []

    for win_idx in range(num_windows):
        window = mixture_conv2d[:, win_idx]
        fingerprint = extract_425_point_fingerprint(window, SAMPLE_RATE)
        mixture_fingerprints.append(fingerprint)

    # Learn EQ
    print(f"Optimizing EQ ({NUM_ITERATIONS} iterations)...")
    eq_curves = []
    for win_idx in range(num_windows):
        eq_curves.append(np.zeros(425))

    for iteration in range(NUM_ITERATIONS):
        total_loss = 0.0

        for win_idx in range(num_windows):
            acapella_fp = acapella_fingerprints[win_idx]
            mixture_fp = mixture_fingerprints[win_idx]

            adjusted_fp = mixture_fp + eq_curves[win_idx]

            difference = adjusted_fp - acapella_fp
            loss = np.mean(difference**2)
            total_loss += loss

            gradient = difference
            eq_curves[win_idx] -= LEARNING_RATE * gradient

        avg_loss = total_loss / num_windows

        if iteration % 20 == 0:
            print(f"    Iteration {iteration:3d}: Loss = {avg_loss:.6f}")

    print(f"  ✓ Final loss: {avg_loss:.6f}")

    # Apply learned EQ to mixture STFT
    print(f"Applying learned EQ to mixture...")
    refined_magnitude = current_mixture_mag.copy()

    for win_idx in range(num_windows):
        eq_freq_profile = eq_curves[win_idx][:400]

        # Interpolate EQ to full frequency resolution
        eq_full = np.interp(
            np.linspace(0, SAMPLE_RATE/2, refined_magnitude.shape[0]),
            np.linspace(0, SAMPLE_RATE/2, 400),
            eq_freq_profile
        )

        # Apply additive EQ
        refined_magnitude[:, win_idx] += eq_full

    print(f"  Refined magnitude range: [{np.min(refined_magnitude):.3f}, {np.max(refined_magnitude):.3f}]")

    return refined_magnitude

# ============================================
# TEST WITH 3 SLICES
# ============================================

print("\n" + "="*80)
print("TESTING WITH 3 SLICES")
print("="*80)

# Slice 1: Horizontal
horizontal_kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
current_mixture_magnitude = process_one_slice(
    "HORIZONTAL",
    horizontal_kernel,
    acapella_magnitude,
    current_mixture_magnitude
)

# Slice 2: Vertical
vertical_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
current_mixture_magnitude = process_one_slice(
    "VERTICAL",
    vertical_kernel,
    acapella_magnitude,
    current_mixture_magnitude
)

# Slice 3: Diagonal
diagonal_kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
current_mixture_magnitude = process_one_slice(
    "DIAGONAL",
    diagonal_kernel,
    acapella_magnitude,
    current_mixture_magnitude
)

# ============================================
# RECONSTRUCT AUDIO
# ============================================

print("\n" + "="*80)
print("RECONSTRUCTING AUDIO")
print("="*80)

# Reconstruct using original phase
print("\nReconstrucing STFT...")
refined_stft = current_mixture_magnitude * np.exp(1j * mixture_phase)
separated_vocal = librosa.istft(refined_stft, hop_length=HOP_LENGTH, n_fft=N_FFT)

# Normalize
separated_vocal = separated_vocal / (np.max(np.abs(separated_vocal)) + 1e-8)

# Save
output_path = f"{OUTPUT_DIR}/separated_vocal_3slices_44100hz.wav"
sf.write(output_path, separated_vocal, SAMPLE_RATE)
print(f"✓ Saved: {output_path}")

# Check if silent
rms_output = np.sqrt(np.mean(separated_vocal**2))
rms_original = np.sqrt(np.mean(mixture_audio**2))

print(f"\nRMS Energy Check:")
print(f"  Original mixture: {rms_original:.6f}")
print(f"  Separated output: {rms_output:.6f}")

if rms_output < 0.001:
    print("  ⚠️  WARNING: Output is nearly silent!")
else:
    print("  ✓ Output has audible content")

# ============================================
# CREATE SPECTROGRAM
# ============================================

print("\n" + "="*80)
print("CREATING SPECTROGRAM")
print("="*80)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original mixture
axes[0].imshow(
    librosa.amplitude_to_db(mixture_magnitude, ref=np.max),
    aspect='auto', origin='lower', cmap='viridis'
)
axes[0].set_title(f'Original Mixture (44100 Hz)')
axes[0].set_ylabel('Frequency')

# Separated
stft_output = librosa.stft(separated_vocal, n_fft=N_FFT, hop_length=HOP_LENGTH)
axes[1].imshow(
    librosa.amplitude_to_db(np.abs(stft_output), ref=np.max),
    aspect='auto', origin='lower', cmap='viridis'
)
axes[1].set_title(f'After 3 Slices (44100 Hz)')
axes[1].set_ylabel('Frequency')

# Target acapella
axes[2].imshow(
    librosa.amplitude_to_db(acapella_magnitude, ref=np.max),
    aspect='auto', origin='lower', cmap='viridis'
)
axes[2].set_title(f'Target Acapella (44100 Hz)')
axes[2].set_ylabel('Frequency')
axes[2].set_xlabel('Time Frame')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/spectrogram_44100hz.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved spectrogram")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nOutput saved to: {OUTPUT_DIR}/")
print(f"Listen to separated_vocal_3slices_44100hz.wav to verify it works at 44100 Hz")
