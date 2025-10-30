# Lab 3: librosa STFT and Spectrograms - 1-Minute Presentation Guide

## Overview
Demonstrate STFT transformation and the reversible nature of spectrogram processing. Show how waveforms become 2D data for neural networks. Total presentation time: 1 minute.

## Setup Beforehand (5-10 minutes prep)
- **Environment Requirements:**
  - librosa, numpy, matplotlib installed
  - Audio file ready for processing
- **Files to Prepare:**
  - `lab3_stft_basics.py` with functions: `create_spectrogram()`, `visualize_spectrogram()`, `test_round_trip()`
  - Sample audio file in working directory
- **Hardware Check:**
  - Display for spectrogram visualization
  - Sufficient memory for STFT computation
- **Presentation Materials:**
  - Slide 3: STFT code and visualization
  - Slide 4: ISTFT reconstruction
  - Backup: STFT parameters explanation

## Demo Execution Script (Exactly 1 Minute)

### Timing: 0:00 - 0:15 (15 seconds): Introduction & STFT Transformation
**Code to Run:**
```python
import librosa
import numpy as np

# Load audio
audio, sr = librosa.load('sample_audio.wav', sr=22050)
print(f"Audio loaded: {audio.shape} samples")

# STFT transformation
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)
phase = np.angle(stft)

print(f"STFT shape: {stft.shape} (complex)")
print(f"Magnitude shape: {magnitude.shape}")
print(f"Phase shape: {phase.shape}")
```

**Narration:**
"Lab 3: STFT transforms 1D audio into 2D spectrograms! This is the core of Cameron's POC and our U-Net pipeline."

**Expected Output:**
```
Audio loaded: (661500,) samples
STFT shape: (1025, 1293) (complex)
Magnitude shape: (1025, 1293)
Phase shape: (1025, 1293)
```

### Timing: 0:15 - 0:35 (20 seconds): Visualize Spectrogram
**Code to Run:**
```python
import librosa.display
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude spectrogram
librosa.display.specshow(
    librosa.amplitude_to_db(magnitude, ref=np.max),
    sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=axes[0]
)
axes[0].set_title('Magnitude Spectrogram (dB)')
fig.colorbar(axes[0].images[0], ax=axes[0], format='%+2.0f dB')

# Phase spectrogram
librosa.display.specshow(
    phase, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=axes[1], cmap='twilight'
)
axes[1].set_title('Phase Spectrogram')
fig.colorbar(axes[1].images[0], ax=axes[1])

plt.tight_layout()
plt.show()
```

**Narration:**
"Look at this! Magnitude shows loudness, phase shows timing. Bright areas are vocal frequencies - this is image-like data for U-Net."

**Key Observation:** Point out frequency patterns and color differences between magnitude/phase.

### Timing: 0:35 - 0:55 (20 seconds): Round-Trip Reconstruction
**Code to Run:**
```python
# Reconstruct audio with ISTFT
reconstructed = librosa.istft(stft, hop_length=512, length=len(audio))

# Check quality
difference = np.abs(audio - reconstructed).max()
mean_diff = np.abs(audio - reconstructed).mean()

print(f"Original length: {len(audio)}")
print(f"Reconstructed length: {len(reconstructed)}")
print(f"Max difference: {difference:.2e}")
print(f"Mean difference: {mean_diff:.2e}")
print(f"Quality: {'Perfect!' if difference < 1e-5 else 'Very Good'}")
```

**Narration:**
"ISTFT reverses STFT perfectly! This lossless transformation enables our complete audio separation pipeline."

**Expected Output:**
```
Original length: 661500
Reconstructed length: 661500
Max difference: 1.23e-13
Mean difference: 1.45e-14
Quality: Perfect!
```

### Timing: 0:55 - 1:00 (5 seconds): Wrap-up
**Narration:**
"STFT/ISTFT is reversible - the foundation for U-Net audio processing!"

## Key Points to Emphasize
- STFT as time-frequency transformation
- Spectrograms as 2D representations for neural networks
- Perfect reconstruction with ISTFT
- Connection to Cameron's manual analysis

## Troubleshooting
- If STFT slow: Reduce n_fft or use shorter audio
- If visualization issues: Check matplotlib backend
- If reconstruction imperfect: Verify hop_length consistency

## Success Criteria
- [ ] STFT computed with correct shapes
- [ ] Spectrogram visualized with magnitude and phase
- [ ] Round-trip reconstruction shows near-zero error
- [ ] Transformation concepts explained clearly
