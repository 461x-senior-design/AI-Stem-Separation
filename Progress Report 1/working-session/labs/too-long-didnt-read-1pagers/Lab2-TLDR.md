# Lab 2 TLDR: NumPy for Audio Processing

## Overview
**Duration:** 3-5 minutes  
**Focus:** Efficient numerical operations on audio data using NumPy

## Key Learnings
- NumPy operations: 10-100x faster than Python loops
- Array slicing for audio segments
- Cameron's 18-slice analysis approach
- NumPy ↔ PyTorch tensor conversion

## Core Code Concepts
```python
import numpy as np

# Vectorized operations (fast!)
audio_quiet = audio * 0.5  # Volume adjustment
audio_inverted = -audio    # Phase inversion

# Array slicing (Cameron's method)
num_slices = 18
slice_size = time_frames // num_slices
for i in range(num_slices):
    start = i * slice_size
    end = start + slice_size
    slice_data = magnitude[:, start:end]  # All freq, specific time
```

## Team Connections
- **Cameron's POC:** 765,000 measurements via NumPy operations
- **cervanj2's Architecture:** NumPy in preprocessing & post-processing
- **Ryan & Yovannoa:** NumPy ↔ PyTorch conversion in tutorials

## Success Criteria
- Perform element-wise operations (volume, phase)
- Use array slicing to extract segments
- Understand Cameron's 18-slice approach
- Convert between NumPy arrays and PyTorch tensors

## Pipeline Position
NumPy array (waveform) → NumPy array (processed) → Ready for spectrogram creation
