# Lab 3 TLDR: librosa Fundamentals - STFT and Spectrograms

## Overview
**Duration:** 3-5 minutes  
**Focus:** Transforming waveforms into spectrograms using Short-Time Fourier Transform (STFT)

## Key Learnings
- STFT converts time-domain audio to frequency-time domain
- Spectrograms: 2D representations (frequency × time)
- Magnitude for processing, phase for reconstruction
- Round-trip: Audio ↔ Spectrogram ↔ Audio (lossless)

## Core Code Concepts
```python
import librosa

# STFT transformation
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)    # What U-Net processes
phase = np.angle(stft)      # For reconstruction

# Round-trip reconstruction
reconstructed = librosa.istft(stft, hop_length=512, length=len(audio))
```

## Team Connections
- **Cameron's POC:** Core STFT/ISTFT operations throughout
- **cervanj2's Architecture:** STFT in preprocessing, ISTFT in post-processing
- **Complete Pipeline:** Audio → STFT → U-Net → ISTFT → Separated audio

## Success Criteria
- Compute STFT and create spectrograms
- Understand magnitude vs phase components
- Perform round-trip reconstruction with ISTFT
- Explain Cameron's STFT-based separation workflow

## Pipeline Position
NumPy array → Spectrogram (2D) → Ready for PyTorch tensor conversion
