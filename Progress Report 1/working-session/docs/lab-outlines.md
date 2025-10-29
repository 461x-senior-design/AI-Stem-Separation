# Five Mini-Labs: Audio Stem Separation with U-Net

## Overview
So here is the idea- we all did some really good work this week and we agreed to all include ourself in this weekend presentation (monday) - I think a way we can all both take ownership of our reserach and also learn new integrations a hte same would be to sort of offer 5 different little "labs"-- or tiny demos that we can each describe or show, or demo, -- in the progress report video. These 5 "mini" labs/demos break down the Astro tutorial into digestible 3-5 minute demos- and they each blend in what everyone Reserahced.  each focusing on a core component while building toward the complete U-Net vocal separation system.

Thinks like timing of each lab (if too short long etc) are minor things we can tweak as need be. I feel like presenting this, or something like this will put us in good position in terms of our preparation phase of the project and will also result in a really good submission for the progress repot.

Please review the following. If everyone likes this idea, maybe we can all pick from one of the five following below and then each present one Monday.

Let me know your feelings on this, if this doesn't hit the mark for everyone, please add your notes on how you feel and we'll try and get there.
---


# Lab Summary and Integration

## Progressive Build
Each lab builds on the previous:
1. **Lab 1:** Get audio data into Python
2. **Lab 2:** Understand how to manipulate it with NumPy
3. **Lab 3:** Transform it to spectrograms with librosa
4. **Lab 4:** Process it with PyTorch on GPU
5. **Lab 5:** Apply U-Net for vocal separation

## Team Integration
Every team member's work is represented:
- **Cameron:**  Pipeline and POC validation
- **cervanj2:** Architecture provides the roadmap
- **Ryan:** PyTorch basics foundation
- **Yovannoa:** Complete neural network training
- **Haedon:** Deep Design principles throughout 

## Alignment with Progress Report
- 5 separate slide shows (one per lab)
- 3-5 minute demos each (short, bitsized, easy to condense if necessary)
- Builds toward full demo in coming weeks
- Gets us ready for encoder/decoder processes at end of semester 
---

# Lab 1: Data Integration and Preparation

## Objective
Learn how to load audio files, convert them to spectrograms, and prepare data for neural network processing.

## Team Member Contribution
- **Cameron:** His POC starts with loading "Intergalactic" and creating spectrograms
- **Yovannoa:** Training a Classifier section demonstrates dataset loading and transformation
- **cervanj2:** Architecture identifies preprocessing as the first technical layer

## Learning Outcomes
- Load audio files using Python
- Understand audio as numerical data
- Convert audio to spectrograms (time-frequency representation)
- Visualize audio data

## "Hello World" Example
```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
audio, sr = librosa.load('sample_audio.wav', sr=22050)

# Display basic info
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f} seconds")

# Visualize waveform
plt.figure(figsize=(10, 4))
plt.plot(audio[:1000])  # First 1000 samples
plt.title("Audio Waveform (first 1000 samples)")
plt.show()
```

## Building Toward Astro
- This is step 1 of Cameron's pipeline
- Sets up data that will feed into U-Net
- Establishes audio → numerical data concept

## 3-5 Minute Demo Flow
1. **0:00-1:00** - Explain what audio data looks like numerically
2. **1:00-2:30** - Load audio file, show waveform
3. **2:30-4:00** - Introduce concept of spectrograms (preview next lab)
4. **4:00-5:00** - Connect to Cameron's POC and cervanj2's architecture

---

# Lab 2: NumPy for Audio Processing

## Objective
Understand how NumPy enables efficient numerical operations on audio data, forming the foundation for all audio processing.

## Team Member Contribution
- **Cameron:** POC performs ~765,000 measurements using NumPy arrays
- **cervanj2:** Preprocessing and post-processing layers both use NumPy
- **Ryan & Yovannoa:** Both tutorials show NumPy ↔ PyTorch tensor interoperability

## Learning Outcomes
- Create and manipulate NumPy arrays
- Perform element-wise operations on audio data
- Understand array shapes and dimensions
- Bridge between NumPy and PyTorch tensors

## "Hello World" Example
```python
import numpy as np

# Create a simple "audio" signal (sine wave)
sample_rate = 22050
duration = 1.0  # seconds
frequency = 440  # A4 note

# Generate time points
t = np.linspace(0, duration, int(sample_rate * duration))

# Generate sine wave
audio_signal = np.sin(2 * np.pi * frequency * t)

print(f"Signal shape: {audio_signal.shape}")
print(f"Signal dtype: {audio_signal.dtype}")
print(f"Signal range: [{audio_signal.min():.3f}, {audio_signal.max():.3f}]")

# Simple operation: volume adjustment
quiet_signal = audio_signal * 0.5
loud_signal = audio_signal * 2.0

print(f"Quiet signal range: [{quiet_signal.min():.3f}, {quiet_signal.max():.3f}]")
print(f"Loud signal range: [{loud_signal.min():.3f}, {loud_signal.max():.3f}]")
```

## Building Toward Astro
- Cameron's 18-slice analysis uses NumPy array slicing
- All STFT operations return NumPy arrays
- Understanding arrays is crucial for spectrogram manipulation

## 3-5 Minute Demo Flow
1. **0:00-1:00** - Why NumPy? (speed, vectorization, array operations)
2. **1:00-2:30** - Create and manipulate simple audio arrays
3. **2:30-4:00** - Show array slicing (like Cameron's 18 slices)
4. **4:00-5:00** - Preview how NumPy connects to librosa and PyTorch

---

# Lab 3: librosa Fundamentals

## Objective
Master librosa's audio processing capabilities, focusing on STFT (Short-Time Fourier Transform) for converting audio to spectrograms.

## Team Member Contribution
- **Cameron:** Uses librosa.stft() and librosa.istft() throughout POC
- **cervanj2:** Preprocessing layer explicitly calls out librosa and STFT
- **Connection to PyTorch:** Spectrograms become the input to U-Net

## Learning Outcomes
- Load audio with librosa
- Compute STFT to create spectrograms
- Understand magnitude and phase
- Reconstruct audio with ISTFT

## "Hello World" Example
```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio
audio, sr = librosa.load('sample_audio.wav', sr=22050)

# Compute STFT (this is the KEY transformation)
stft = librosa.stft(audio)
magnitude = np.abs(stft)
phase = np.angle(stft)

print(f"Audio shape: {audio.shape}")
print(f"STFT shape: {stft.shape}")  # (freq_bins, time_frames)
print(f"Magnitude shape: {magnitude.shape}")

# Visualize spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(
    librosa.amplitude_to_db(magnitude, ref=np.max),
    sr=sr,
    x_axis='time',
    y_axis='hz'
)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Reconstruct audio
reconstructed = librosa.istft(stft)
print(f"Reconstructed audio shape: {reconstructed.shape}")
```

## Building Toward Astro
- This is THE core transformation in Cameron's POC
- STFT converts audio → spectrogram for neural network input
- ISTFT converts spectrogram → audio for final output
- Matches cervanj2's preprocessing/post-processing layers

## 3-5 Minute Demo Flow
1. **0:00-1:00** - Why spectrograms? Time-frequency representation
2. **1:00-2:30** - Compute STFT, show magnitude spectrogram
3. **2:30-4:00** - Demonstrate ISTFT reconstruction (round-trip)
4. **4:00-5:00** - Explain how U-Net will process these spectrograms

---

# Lab 4: PyTorch Essentials

## Objective
Learn PyTorch fundamentals for deep learning: tensors, GPU acceleration, and basic neural network operations.

## Team Member Contribution
- **Ryan:** Completed "Learn the Basics" - tensors, DataLoader, autograd, training loop
- **Yovannoa:** Completed "60 Minute Blitz" - comprehensive neural network training
- **cervanj2:** Inference layer uses PyTorch and U-Net

## Learning Outcomes
- Create PyTorch tensors from NumPy arrays
- Move tensors to GPU for acceleration
- Understand automatic differentiation
- Perform basic neural network operations

## "Hello World" Example
```python
import torch
import numpy as np

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create tensor from NumPy array (simulating audio spectrogram)
audio_data = np.random.randn(256, 256)  # 256 freq bins × 256 time frames
tensor = torch.from_numpy(audio_data).float()

print(f"Tensor shape: {tensor.shape}")
print(f"Tensor device: {tensor.device}")

# Move to GPU
tensor_gpu = tensor.to(device)
print(f"GPU tensor device: {tensor_gpu.device}")

# Simple operation (element-wise)
tensor_processed = torch.relu(tensor_gpu)  # Common activation function
print(f"Processed tensor shape: {tensor_processed.shape}")

# Move back to CPU for saving/visualization
result = tensor_processed.cpu().numpy()
print(f"Result shape: {result.shape}")
```

## Building Toward Astro
- Spectrograms from librosa → PyTorch tensors
- U-Net is a PyTorch neural network
- GPU acceleration crucial for training
- Builds on Ryan and Yovannoa's PyTorch knowledge

## 3-5 Minute Demo Flow
1. **0:00-1:00** - What is PyTorch? Why for audio ML?
2. **1:00-2:30** - NumPy arrays → PyTorch tensors
3. **2:30-4:00** - GPU acceleration demonstration
4. **4:00-5:00** - Preview U-Net as a PyTorch model

---

# Lab 5: U-Net Architecture for Audio Separation

## Objective
Understand U-Net's encoder-decoder architecture and how it processes audio spectrograms for stem separation.

## Team Member Contribution
- **Cameron:** POC validates concept before U-Net training investment
- **cervanj2:** Places U-Net at core of inference layer
- **Ryan & Yovannoa:** PyTorch knowledge provides the implementation framework
- **Haedon:** Design philosophy applies to clean U-Net implementation

## Learning Outcomes
- Understand encoder-decoder architecture
- Learn how U-Net processes spectrograms
- Connect all previous labs into complete pipeline
- Understand training vs. inference

## "Hello World" Example
```python
import torch
import torch.nn as nn

# Simplified U-Net building block
class SimpleUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Create a simple block
block = SimpleUNetBlock(in_channels=1, out_channels=16)

# Simulate a spectrogram input (batch_size=1, channels=1, freq=256, time=256)
spectrogram = torch.randn(1, 1, 256, 256)

# Forward pass
output = block(spectrogram)
print(f"Input shape: {spectrogram.shape}")
print(f"Output shape: {output.shape}")
```

## Building Toward Next Phase 
- **Complete pipeline:**
  1. Load audio (Lab 1)
  2. NumPy operations (Lab 2)
  3. STFT with librosa (Lab 3)
  4. Convert to PyTorch tensor (Lab 4)
  5. **Process with U-Net** (Lab 5)
  6. ISTFT back to audio (Lab 3)
  7. Save separated stems (Lab 1)

## 3-5 Minute Demo Flow
1. **0:00-1:00** - U-Net architecture overview (encoder-decoder)
2. **1:00-2:30** - Explore forked Pytorch-UNet repository structure
3. **2:30-4:00** - Show how U-Net processes spectrograms
4. **4:00-5:00** - Connect all 5 labs into complete Astro pipeline

---

# Lab Summary and Integration

## Progressive Build
Each lab builds on the previous:
1. **Lab 1:** Get audio data into Python
2. **Lab 2:** Understand how to manipulate it with NumPy
3. **Lab 3:** Transform it to spectrograms with librosa
4. **Lab 4:** Process it with PyTorch on GPU
5. **Lab 5:** Apply U-Net for vocal separation

## Team Integration
Every team member's work is represented:
- **Cameron:** Overall pipeline and POC validation
- **cervanj2:** Architecture provides the roadmap
- **Ryan:** PyTorch basics foundation
- **Yovannoa:** Complete neural network training
- **Haedon:** Design principles throughout

## Alignment with Progress Report
- 5 separate slide shows (one per lab)
- 3-5 minute demos each
- Builds toward full Astro demo
- Prepares team for weeks 6-7-8 (actual U-Net training)
