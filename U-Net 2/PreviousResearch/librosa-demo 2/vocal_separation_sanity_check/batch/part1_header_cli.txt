"""
PROGRESSIVE VOCAL SEPARATION TEST
==================================

This script tests vocal separation using progressively more convolutional slices.

Usage:
  python sanity_check_progressive.py --slices 18    # Test 1 through 18 slices
  python sanity_check_progressive.py --slices 5     # Test 1 through 5 slices
  python sanity_check_progressive.py                # Default: all 18 slices

For each N from 1 to max_slices:
  - Optimizes using slices 1 through N sequentially
  - Always ends with slice_0_raw as final polish
  - Saves: output_progressive_Nslices/N_slices/extracted_vocal.wav

Expected outcome: Hear progressive improvement (or identify breakdown point)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal, ndimage
from scipy.signal import find_peaks, butter, sosfilt
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse

# ============================================
# COMMAND-LINE INTERFACE
# ============================================

parser = argparse.ArgumentParser(
    description='Progressive vocal separation test using multi-scale spectral slices',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s --slices 18    Test all 18 slices
  %(prog)s --slices 5     Quick test with 5 slices
  %(prog)s -n 10          Test up to 10 slices
    """
)

parser.add_argument(
    '--slices', '-n',
    type=int,
    default=18,
    choices=range(1, 19),
    metavar='N',
    help='Number of conv slices to test (1-18, default: 18)'
)

args = parser.parse_args()

print("="*70)
print("PROGRESSIVE VOCAL SEPARATION TEST")
print("="*70)
print(f"\nTesting progressive separation using 1 to {args.slices} slices")
print("Each iteration adds one more conv slice, always ending with slice_0_raw polish\n")

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'vocal_path': 'rtg/100-window/isolated_vocal.wav',
    'mixture_path': 'rtg/100-window/stereo_mixture.wav',
    'output_base_dir': f'output_progressive_{args.slices}slices',
    'max_slices': args.slices,
    'sr': 22050,
    'duration': 4.7,
    'n_fft': 2048,
    'hop_length': 1024,
    'num_iterations': 100,
    'learning_rate': 0.01,
}

# Create base output directory
Path(CONFIG['output_base_dir']).mkdir(exist_ok=True, parents=True)

print(f"Configuration:")
print(f"  Max slices: {CONFIG['max_slices']}")
print(f"  Sample rate: {CONFIG['sr']} Hz")
print(f"  Iterations per slice: {CONFIG['num_iterations']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Output directory: {CONFIG['output_base_dir']}/")
print()
