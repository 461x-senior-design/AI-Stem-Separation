# Lab 1: Data Integration and Preparation - 1-Minute Presentation Guide

## How We Present This in Exactly One Minute!
**Let's crush this 1-minute presentation!** We're going to live-code our way through audio loading and visualization, showing how audio becomes numerical data for machine learning. It's fast, it's live, and it demonstrates the foundation of our entire audio pipeline!

## What the 1-Minute Presentation Will Be

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
- Connection to the POC starting point

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

## Setup Beforehand (15-20 minutes total prep time)
Each team member will assemble their own code and working repository for this lab.

### Environment Setup (5-7 minutes)
- **Install Dependencies:** `pip install librosa numpy matplotlib`
- **Verify Installation:** Test imports in Python environment
- **IDE Setup:** Prepare Jupyter notebook or Python IDE with code execution capabilities

### Code Assembly and Repository Setup (7-10 minutes)
- **Create Working Directory:** Set up dedicated folder for Lab 1
- **Assemble Code Files:**
  - Create `lab1_basic.py` with functions: `load_audio_basic()`, `visualize_waveform()`, `spectrogram_preview()`
  - Copy/paste from provided code examples or write from scratch
  - Add proper imports and docstrings
- **Test Code Independently:** Run functions with sample data to ensure they work
- **Version Control:** Initialize git repo and commit working code

### Files and Assets Preparation (3-5 minutes)
- **Audio File Acquisition:** Obtain or download `sample_audio.wav` (30-second clip recommended)
- **Alternative Audio Sources:** Prepare fallback options (librosa examples, personal audio files)
- **Test File Loading:** Verify audio file loads correctly in your environment

### Hardware and Display Verification (2-3 minutes)
- **Display Setup:** Confirm matplotlib plots render properly
- **Audio Playback:** Optional - test audio playback for verification
- **Performance Check:** Time loading of sample audio file

### Presentation Materials Organization (1-2 minutes)
- **Slide Preparation:** Ensure access to Slide 3 (code example) and Slide 4 (spectrogram preview)
- **Backup Materials:** Prepare Slide 2 if time allows for audio concepts explanation
- **Timing Practice:** Run through demo script once to verify 1-minute timing
