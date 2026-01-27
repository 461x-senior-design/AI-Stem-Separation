# src/postprocessing/constants.py
"""
Output validation constants for postprocessing pipeline - Sprint 1 MVP.

Sprint 1 Scope: Basic validation only (NaN/Inf, shape, sample rate, file I/O)
TODO: Advanced quality checks in future sprints (frequency analysis, perceptual metrics)
"""

from src.preprocessing.constants import TARGET_CHANNELS

# Expected number of channels (stereo)
EXPECTED_CHANNELS: int = TARGET_CHANNELS  # 2 (stereo)

# Minimum number of samples in output (~10ms at 44100Hz)
MIN_SAMPLES: int = 441

# Data corruption errors
ERROR_WAVEFORM_CONTAINS_NAN: str = "Waveform contains NaN values"
ERROR_WAVEFORM_CONTAINS_INF: str = "Waveform contains infinite values"

# Shape validation errors
ERROR_WAVEFORM_WRONG_SHAPE: str = (
    "Waveform has unexpected shape: {shape}, expected [2, N]"
)
ERROR_WAVEFORM_TOO_SHORT: str = (
    "Waveform too short: {samples} samples (min {min_samples})"
)

# Channel validation errors
ERROR_CHANNEL_MISMATCH: str = (
    "Channel count mismatch: got {actual}, expected {expected}"
)

# Sample rate validation errors
ERROR_SAMPLE_RATE_MISMATCH: str = (
    "Sample rate mismatch: got {actual}Hz, expected {expected}Hz"
)

# Output quality errors
ERROR_OUTPUT_SILENT: str = (
    "Output is silent (RMS={rms:.2e}), separation likely failed"
)

# File I/O errors
ERROR_FILE_NOT_CREATED: str = "Output file was not created: {path}"
ERROR_FILE_NOT_READABLE: str = "Output file exists but cannot be read: {path}"


# Minimum RMS level (below this is silent)
MIN_RMS_LEVEL: float = 1e-7