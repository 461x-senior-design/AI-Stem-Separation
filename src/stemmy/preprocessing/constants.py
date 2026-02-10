# constants.py
# Audio processing constants for preprocessing pipeline.
# These values must match the model's training configuration.

#########################
# Change by Ryan:
# Reason:
# This module is not an authoritative "global constants" file.
# Re-export shared defaults from stemmy.constants so preprocessing modules
# stay aligned with training/inference while keeping a single source of truth.
# What it does:
# Re-exports the shared constants so existing preprocessing imports keep working
# without duplicating or drifting from stemmy.constants.
from stemmy.constants import (  # noqa: F401
    HOP_LENGTH,
    N_FFT,
    N_FREQ_BINS,
    STFT_CENTER,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    WINDOW,
)
#########################
