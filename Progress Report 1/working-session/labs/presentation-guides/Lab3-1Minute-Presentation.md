# Lab 3: librosa STFT and Spectrograms - 1-Minute Presentation Guide

## How We Present This in Exactly One Minute!
**Let's nail this 1-minute STFT demonstration!** We'll transform audio waveforms into spectrograms with live coding, show the perfect round-trip reconstruction, and reveal how this creates the 2D data that U-Net processes. Technical magic in 60 seconds!

## What the 1-Minute Presentation Will Be

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
"Lab 3: STFT transforms 1D audio into 2D spectrograms! This is the core of the POC and our U-Net pipeline."

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
- Connection to the POC manual analysis

## Troubleshooting
- If STFT slow: Reduce n_fft or use shorter audio
- If visualization issues: Check matplotlib backend
- If reconstruction imperfect: Verify hop_length consistency

## Success Criteria
- [ ] STFT computed with correct shapes
- [ ] Spectrogram visualized with magnitude and phase
- [ ] Round-trip reconstruction shows near-zero error
- [ ] Transformation concepts explained clearly

## Setup Beforehand (25-30 minutes total prep time)
Each team member will assemble their own code and working repository for this lab.

### Environment Setup (7-10 minutes)
- **Install Dependencies:** `pip install librosa numpy matplotlib`
- **librosa Verification:** Test STFT/ISTFT operations with sample data
- **Visualization Setup:** Configure matplotlib for spectrogram display with proper colorbars

### Code Assembly and Repository Setup (12-15 minutes)
- **Create Working Directory:** Set up Lab 3 repository with git initialization
- **Implement STFT Functions:**
  - Create `lab3_stft_basics.py` with functions: `create_spectrogram()`, `visualize_spectrogram()`, `test_round_trip()`
  - Include magnitude/phase separation and reconstruction logic
  - Add parameter exploration (n_fft, hop_length variations)
- **Independent Testing:** Test round-trip reconstruction accuracy
- **Performance Optimization:** Experiment with different STFT parameters
- **Version Control:** Commit tested code with meaningful commit messages

### Audio Assets and Test Data Preparation (5-7 minutes)
- **Audio File Selection:** Choose 30-second audio file for consistent demos
- **Multiple Test Files:** Prepare different audio types (speech, music, mixed)
- **Pre-computed Spectrograms:** Generate spectrograms for faster demo loading
- **Quality Verification:** Test reconstruction quality across different audio types

### Hardware and Visualization Verification (4-5 minutes)
- **STFT Performance:** Test computation time on available hardware
- **Memory Requirements:** Verify sufficient RAM for spectrogram arrays
- **Display Quality:** Ensure spectrogram plots render with proper frequency/time axes
- **Color Scheme Testing:** Verify dB scaling and colorbar functionality

### Presentation Materials Organization (1-2 minutes)
- **Slide Preparation:** Access Slide 3 (STFT demo) and Slide 4 (ISTFT reconstruction)
- **Demo Timing Practice:** Run complete sequence to confirm 1-minute execution
- **Backup Slides:** Prepare STFT parameter explanations
