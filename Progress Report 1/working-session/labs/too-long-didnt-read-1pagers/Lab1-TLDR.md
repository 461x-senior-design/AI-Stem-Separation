# Lab 1 TLDR: Data Integration and Preparation

## Overview
**Duration:** 3-5 minutes  
**Focus:** Loading audio files into Python and understanding them as numerical data

## Key Learnings
- Audio files are arrays of numbers (samples)
- Sample rate = samples per second (22,050 Hz standard)
- Audio manipulation uses NumPy arrays
- Spectrograms preview (frequency content over time)

## Core Code Concepts
```python
import librosa
import numpy as np

# Load audio
audio, sr = librosa.load('sample_audio.wav', sr=22050)
print(f"Shape: {audio.shape}, Sample rate: {sr}")

# Visualize waveform
plt.plot(audio[:1000])
plt.title("Audio Waveform")
```

## Team Connections
- **Cameron's POC:** Starts with `librosa.load('Intergalactic.mp3')`
- **cervanj2's Architecture:** Foundation of Preprocessing Layer
- **Yovannoa's Classifier:** Dataset loading principles

## Success Criteria
- Load audio files with librosa
- Understand audio as numerical data
- Visualize basic waveforms
- Explain spectrogram benefits (preview for Lab 3)

## Pipeline Position
Audio file → NumPy array (waveform) → Ready for STFT processing
