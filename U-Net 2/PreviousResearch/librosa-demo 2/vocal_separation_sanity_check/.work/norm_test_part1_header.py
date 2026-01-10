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
