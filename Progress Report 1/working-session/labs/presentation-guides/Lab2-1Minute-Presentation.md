# Lab 2: NumPy for Audio Processing - 1-Minute Presentation Guide

## How We Present This in Exactly One Minute!
**Let's dominate this 1-minute NumPy demo!** We'll show how NumPy makes audio processing lightning-fast with vectorized operations, create audio signals, and demonstrate the 18-slice analysis that powers our measurements. Live coding at its finest!

## What the 1-Minute Presentation Will Be

### Timing: 0:00 - 0:15 (15 seconds): Introduction & Generate Signal
**Code to Run:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple audio signal (440 Hz sine wave)
sample_rate = 22050
duration = 1.0  # seconds
frequency = 440  # Hz (A note)

t = np.linspace(0, duration, int(sample_rate * duration))
audio_signal = np.sin(2 * np.pi * frequency * t)

print(f"Created {len(audio_signal)} samples")
print(f"Signal range: [{audio_signal.min():.3f}, {audio_signal.max():.3f}]")
```

**Narration:**
"Lab 2 shows NumPy's power! We create 22,050 samples of a 440 Hz sine wave in one operation."

**Expected Output:**
```
Created 22050 samples
Signal range: [-1.000, 1.000]
```

### Timing: 0:15 - 0:35 (20 seconds): Volume Operations
**Code to Run:**
```python
# Volume adjustments
quiet_signal = audio_signal * 0.5  # Reduce volume
loud_signal = audio_signal * 2.0   # Increase volume

print(f"Original range: [{audio_signal.min():.3f}, {audio_signal.max():.3f}]")
print(f"Quiet range: [{quiet_signal.min():.3f}, {quiet_signal.max():.3f}]")
print(f"Loud range: [{loud_signal.min():.3f}, {loud_signal.max():.3f}]")

# Visualize
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(audio_signal[:500])
plt.title("Original Signal")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(quiet_signal[:500])
plt.title("Quiet Signal (×0.5)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(loud_signal[:500])
plt.title("Loud Signal (×2.0)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
```

**Narration:**
"Volume changes are simple multiplication! NumPy processes the entire array simultaneously - 100x faster than loops."

**Key Observation:** Show the scaled amplitudes and identical shapes.

### Timing: 0:35 - 0:55 (20 seconds): Cameron's 18-Slice Analysis
**Code to Run:**
```python
import librosa

# Load audio and create spectrogram
audio, sr = librosa.load('sample_audio.wav', sr=22050)
stft = librosa.stft(audio)
magnitude = np.abs(stft)

print(f"Full spectrogram: {magnitude.shape}")

# Cameron's 18-slice approach
num_slices = 18
time_frames = magnitude.shape[1]
slice_size = time_frames // num_slices

print(f"Dividing into {num_slices} slices of {slice_size} frames each")

# Show first 3 slices
for i in range(3):
    start = i * slice_size
    end = start + slice_size
    slice_data = magnitude[:, start:end]
    slice_mean = slice_data.mean()
    slice_std = slice_data.std()
    print(f"Slice {i+1}: mean={slice_mean:.4f}, std={slice_std:.4f}")
```

**Narration:**
"The POC analyzes spectrograms in 18 slices using NumPy array slicing. This enables the 765,000 measurements!"

**Expected Output:**
```
Full spectrogram: (1025, 646)
Dividing into 18 slices of 35 frames each
Slice 1: mean=0.0234, std=0.0412
Slice 2: mean=0.0198, std=0.0356
Slice 3: mean=0.0251, std=0.0398
```

### Timing: 0:55 - 1:00 (5 seconds): Wrap-up
**Narration:**
"NumPy makes audio processing fast and efficient - essential for the POC!"

## Key Points to Emphasize
- Vectorized operations speed (10-100x faster)
- Broadcasting for element-wise operations
- Array slicing for segmenting data
- Connection to the POC measurement approach

## Troubleshooting
- If audio file missing: Use sine wave generation as fallback
- If memory issues: Reduce array sizes or use smaller audio
- If slow operations: Ensure NumPy is installed correctly

## Success Criteria
- [ ] Sine wave generated with correct properties
- [ ] Volume operations demonstrated with plots
- [ ] 18-slice analysis shows slicing and statistics
- [ ] Concepts explained within 1 minute timeframe

## Setup Beforehand (20-25 minutes total prep time)
Each team member will assemble their own code and working repository for this lab.

### Environment Setup (5-7 minutes)
- **Install Dependencies:** `pip install numpy matplotlib librosa`
- **Verify NumPy Installation:** Test array operations and vectorization
- **IDE Preparation:** Set up Python environment with matplotlib backend configured

### Code Assembly and Repository Setup (10-12 minutes)
- **Create Working Directory:** Establish dedicated Lab 2 folder with git initialization
- **Assemble Core Functions:**
  - Create `lab2_numpy_basics.py` with functions: `create_sine_wave()`, `volume_operations()`, `slice_analysis()`
  - Implement vectorized operations and array slicing examples
  - Add comprehensive docstrings and error handling
- **Independent Testing:** Test each function with sample inputs
- **Performance Benchmarking:** Compare NumPy operations vs Python loops
- **Commit Working Code:** Use git to version control your implementation

### Data and Assets Preparation (4-5 minutes)
- **Audio File Preparation:** Obtain audio file from Lab 1 or download new sample
- **Spectrogram Generation:** Pre-compute spectrogram data for 18-slice demo
- **Test Data Creation:** Generate synthetic audio signals for sine wave demos

### Hardware and Performance Verification (3-4 minutes)
- **NumPy Performance:** Test array operations on available hardware (CPU/GPU)
- **Visualization Setup:** Ensure matplotlib plots display correctly with multiple subplots
- **Memory Check:** Verify sufficient RAM for spectrogram computations

### Presentation Materials Organization (1-2 minutes)
- **Slide Access:** Prepare Slide 3 (signal generation) and Slide 4 (18-slice analysis)
- **Demo Script Practice:** Run complete demo sequence once for timing
- **Backup Content:** Prepare broadcasting examples if needed
