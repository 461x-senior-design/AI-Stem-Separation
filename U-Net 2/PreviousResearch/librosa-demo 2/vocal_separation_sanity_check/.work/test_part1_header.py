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
