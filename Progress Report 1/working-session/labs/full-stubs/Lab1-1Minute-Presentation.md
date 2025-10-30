# Lab 1: Data Integration and Preparation - 1-Minute Presentation Guide

## Overview
Present the core concepts of loading audio into Python and understanding it as numerical data. Focus on live code execution and visualization. Total presentation time: 1 minute.

## Setup Beforehand (5-10 minutes prep)
- **Environment Requirements:**
  - Python 3.x with librosa, numpy, matplotlib installed
  - Jupyter notebook or Python IDE with code execution
  - Audio file ready (e.g., `sample_audio.wav` - any WAV/MP3 works)
- **Files to Prepare:**
  - `lab1_basic.py` with functions: `load_audio_basic()`, `visualize_waveform()`, `spectrogram_preview()`
  - Audio file in working directory
- **Hardware Check:**
  - Display capable of showing matplotlib plots
  - Audio playback optional (for verification)
- **Presentation Materials:**
  - Slide 3: Code example (audio loading and visualization)
  - Slide 4: Spectrogram preview code
  - Backup: Slide 2 (audio as data explanation) if time allows

## Demo Execution Script (Exactly 1 Minute)

### Timing: 0:00 - 0:15 (15 seconds): Introduction & Load Audio
**Code to Run:**
```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
audio, sr = librosa.load('sample_audio.wav', sr=22050)

# Display basic information
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")
```

**Narration (speak while typing/running):**
"Welcome to Lab 1! We start by loading audio into Python using librosa. This transforms any audio file into a NumPy array of numbers."

**Expected Output:**
```
Audio shape: (661500,)
Sample rate: 22050 Hz
Duration: 30.00 seconds
```

### Timing: 0:15 - 0:35 (20 seconds): Visualize Waveform
**Code to Run:**
```python
# Visualize first 1000 samples
plt.figure(figsize=(12, 4))
plt.plot(audio[:1000])
plt.title("Audio Waveform (first 1000 samples)")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.show()
```

**Narration:**
"See this waveform? Each point represents the air pressure at a specific moment. The oscillations show the sound wave pattern."

**Key Observation:** Point out the waveform oscillations and amplitude range.

### Timing: 0:35 - 0:50 (15 seconds): Spectrogram Preview
**Code to Run:**
```python
# Preview spectrogram transformation
spectrogram = librosa.stft(audio)
magnitude = np.abs(spectrogram)

print(f"Waveform shape: {audio.shape}")
print(f"Spectrogram shape: {magnitude.shape}")
print(f"Frequency bins: {magnitude.shape[0]}")
print(f"Time frames: {magnitude.shape[1]}")
```

**Narration:**
"Here's a preview of Lab 3: STFT transforms our 1D waveform into a 2D spectrogram. This is what neural networks process!"

**Expected Output:**
```
Waveform shape: (661500,)
Spectrogram shape: (1025, 646)
Frequency bins: 1025
Time frames: 646
```

### Timing: 0:50 - 1:00 (10 seconds): Wrap-up
**Narration:**
"That's Lab 1! Audio is just numerical data. This foundation enables everything from NumPy processing to U-Net separation."

## Key Points to Emphasize
- Audio as numerical arrays (not "magic")
- Sample rate and duration concepts
- Waveform visualization
- Preview of spectrogram transformation
- Connection to Cameron's POC starting point

## Troubleshooting
- If audio file missing: Use `librosa.example('nutcracker')` as fallback
- If plots don't show: Ensure matplotlib backend is configured
- If slow loading: Use shorter audio file or `duration` parameter

## Success Criteria
- [ ] Audio loaded successfully
- [ ] Basic info printed (shape, rate, duration)
- [ ] Waveform plot displayed
- [ ] Spectrogram preview shows transformation
- [ ] Concepts explained clearly in 1 minute
