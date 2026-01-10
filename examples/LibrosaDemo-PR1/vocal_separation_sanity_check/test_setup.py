"""
Quick test to verify setup before running full sanity check.
"""

import sys

print("="*60)
print("TESTING SANITY CHECK SETUP")
print("="*60)

# Test imports
print("\n[1/5] Testing imports...")
try:
    import numpy as np
    print("  ✓ numpy")
except ImportError:
    print("  ✗ numpy - run: uv pip install numpy")
    sys.exit(1)

try:
    import librosa
    print("  ✓ librosa")
except ImportError:
    print("  ✗ librosa - run: uv pip install librosa")
    sys.exit(1)

try:
    import soundfile as sf
    print("  ✓ soundfile")
except ImportError:
    print("  ✗ soundfile - run: uv pip install soundfile")
    sys.exit(1)

try:
    import scipy
    print("  ✓ scipy")
except ImportError:
    print("  ✗ scipy - run: uv pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    print("  ✓ matplotlib")
except ImportError:
    print("  ✗ matplotlib - run: uv pip install matplotlib")
    sys.exit(1)

# Test file existence
print("\n[2/5] Checking for audio files...")
from pathlib import Path

vocal_exists = Path('isolated_vocal.wav').exists()
mixture_exists = Path('stereo_mixture.wav').exists()

if vocal_exists:
    print("  ✓ isolated_vocal.wav found")
else:
    print("  ✗ isolated_vocal.wav NOT found")
    print("    You need a clean isolated vocal track")

if mixture_exists:
    print("  ✓ stereo_mixture.wav found")
else:
    print("  ✗ stereo_mixture.wav NOT found")
    print("    You need a full mix with vocals, drums, bass, etc.")

if not (vocal_exists and mixture_exists):
    print("\n  Place audio files in this directory before running.")
    sys.exit(1)

# Test audio loading
print("\n[3/5] Testing audio loading...")
try:
    vocal, sr = librosa.load('isolated_vocal.wav', sr=22050, duration=1.0)
    print(f"  ✓ Loaded vocal: {len(vocal)} samples at {sr} Hz")
except Exception as e:
    print(f"  ✗ Error loading vocal: {e}")
    sys.exit(1)

try:
    mixture, sr = librosa.load('stereo_mixture.wav', sr=22050, duration=1.0)
    print(f"  ✓ Loaded mixture: {len(mixture)} samples at {sr} Hz")
except Exception as e:
    print(f"  ✗ Error loading mixture: {e}")
    sys.exit(1)

# Test STFT
print("\n[4/5] Testing STFT...")
try:
    stft = librosa.stft(vocal, n_fft=2048, hop_length=1024)
    mag = np.abs(stft)
    print(f"  ✓ Created spectrogram: {mag.shape}")
except Exception as e:
    print(f"  ✗ Error creating STFT: {e}")
    sys.exit(1)

# Test convolution
print("\n[5/5] Testing convolution...")
try:
    from scipy import ndimage
    kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
    filtered = ndimage.convolve(mag, kernel, mode='constant', cval=0.0)
    print(f"  ✓ Convolution works: {filtered.shape}")
except Exception as e:
    print(f"  ✗ Error in convolution: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYou're ready to run the sanity check:")
print("  python sanity_check.py")
