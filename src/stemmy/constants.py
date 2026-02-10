"""Project-wide centralized constants.

This module defines the single authoritative defaults for audio I/O, STFT parameters,
model input normalization, and canonical stem ordering.

All other constants modules should import or re-export values from here to keep
preprocessing, training, inference, postprocessing, evaluation, and CLI tools aligned.
"""

# Audio format
TARGET_SAMPLE_RATE: int = 44100
# Hz
TARGET_CHANNELS: int = 2
# Stereo

# STFT parameters
N_FFT: int = 4096
HOP_LENGTH: int = 1024
WIN_LENGTH: int = N_FFT
WINDOW: str = "hann"
N_FREQ_BINS: int = N_FFT // 2 + 1
STFT_CENTER: bool = False

# Default normalization settings
DEFAULT_WAVEFORM_NORM: str = "peak"
DEFAULT_SPECTROGRAM_NORM: str = "freq_minmax"

# Default stem ordering
STEMS_4: list[str] = ["drums", "bass", "vocals", "other"]
