# constants.py
# Audio processing constants for preprocessing pipeline.
# These values must match the model's training configuration.

## Audio Format
TARGET_SAMPLE_RATE: int = 44100  # Hz
TARGET_CHANNELS: int = 2  # Stereo

## STFT Parameters
N_FFT: int = 4096  # FFT window size (samples)
HOP_LENGTH: int = 1024  # Samples between STFT frames
WIN_LENGTH: int = N_FFT  # Window length (same as N_FFT)
WINDOW: str = "hann"  # Window function type

# Derived Constants

N_FREQ_BINS: int = N_FFT // 2 + 1  # = 2049 frequency bins
