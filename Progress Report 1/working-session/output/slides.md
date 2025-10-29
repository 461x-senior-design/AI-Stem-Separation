---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    background: #FFFFFF;
    color: #1F2937;
    font-family: 'Inter', 'Calibri', sans-serif;
    padding: 48px;
  }
  h1 {
    color: #1F2937;
    font-size: 40px;
    line-height: 1.1;
  }
  h2 {
    color: #2563EB;
    font-size: 28px;
    line-height: 1.2;
  }
  h3 {
    color: #1F2937;
    font-size: 24px;
  }
  code {
    background: #F3F4F6;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 18px;
  }
  pre {
    background: #F3F4F6;
    padding: 16px;
    border-radius: 8px;
    font-size: 16px;
  }
---

<!-- _class: lead -->
# Week 5 Progress Report
## 5 Mini-Labs: Building the Audio Separation Pipeline

**Team Audio Source Separation**
**Progress Report Presentation**

---

## Today's Goal: Prepare the Team for U-Net Training

**Our Mission:**
Break down the Astro tutorial into 5 digestible labs that prepare us for weeks 6-7-8 U-Net training

**What You'll Learn:**
- Lab 1: Data Integration & Audio Loading
- Lab 2: NumPy for Audio Processing
- Lab 3: librosa STFT Transformation
- Lab 4: PyTorch Essentials
- Lab 5: U-Net Architecture

**Connection to Team Work:**
Every lab maps to team member contributions from Solo Week 4 assignments

---

## The Complete Pipeline: Labs 1-5

```
Audio File (.wav/.mp3)
    ↓ [Lab 1: librosa.load()]
NumPy Array (waveform) - 661,500 samples
    ↓ [Lab 2: NumPy operations]
Processed NumPy Array
    ↓ [Lab 3: librosa.stft()]
Spectrogram (1,025 × 646) - 662,150 measurements
    ↓ [Lab 4: torch.from_numpy()]
PyTorch Tensor on GPU
    ↓ [Lab 5: U-Net processing]
Separated Vocals & Music
```

**Key Insight:** Each lab builds on the previous, creating the complete vocal separation pipeline

---

<!-- _class: lead -->
# Lab 1: Data Integration
## Getting Audio into Python

**Team Connections:**
- Cameron's POC: Starts with loading "Intergalactic"
- Yovannoa: Dataset loading principles from CIFAR-10
- cervanj2: Foundation of Preprocessing Layer

---

## Lab 1: Audio is Just Numbers

**Audio files store numerical samples representing air pressure at each moment**

**Key Concepts:**
- Sample rate = samples per second (22,050 Hz standard)
- Each sample is a floating-point number (-1.0 to 1.0)
- 30 seconds of audio = 661,500 samples at 22,050 Hz
- We can manipulate audio like any other data array

**Demo Code:**
```python
import librosa
audio, sr = librosa.load('sample_audio.wav', sr=22050)
print(f"Audio shape: {audio.shape}")  # (661500,)
print(f"Duration: {len(audio)/sr:.2f} seconds")  # 30.00
```

**Evidence:** Cameron's POC first step loads audio exactly this way

---

## Lab 1: Preview - Waveform to Spectrogram

**Waveforms show amplitude over time (1D)**
**Spectrograms show frequency content over time (2D)**

**The Transformation:**
```python
# Waveform: 1D array
waveform_shape = (661500,)  # Just time

# Spectrogram: 2D image-like array
spectrogram_shape = (1025, 646)  # Frequency × Time
```

**Why This Matters:**
- Neural networks work better with 2D spectrograms than 1D waveforms
- U-Net processes spectrograms as images
- Different sounds occupy different frequency regions

**Result:** 661,500 samples → 662,150 measurements (1,025 freq bins × 646 time frames)

---

<!-- _class: lead -->
# Lab 2: NumPy for Audio
## The Numerical Foundation

**Team Connections:**
- Cameron's POC: ~765,000 measurements using NumPy
- cervanj2: Both preprocessing & post-processing layers
- Ryan & Yovannoa: NumPy ↔ PyTorch conversion

---

## Lab 2: NumPy Powers Cameron's 765,000 Measurements

**Why NumPy is essential for audio processing:**

**Speed:** Vectorized operations are 10-100x faster than Python loops
```python
# Slow: Loop through 661,500 samples
for i in range(len(audio)):
    audio[i] = audio[i] * 0.5

# Fast: One NumPy operation
audio = audio * 0.5  # 100x faster!
```

**Cameron's Use Case:**
- 18-slice multi-scale analysis
- ~765,000 spectrogram measurements analyzed
- All using NumPy array slicing and operations

---

## Lab 2: Array Slicing - Cameron's 18-Slice Approach

**The POC divides spectrograms into 18 slices for multi-scale analysis**

**Demo Code:**
```python
# Full spectrogram: (1025, 646)
magnitude = np.abs(librosa.stft(audio))

# Divide into 18 time slices
num_slices = 18
slice_size = magnitude.shape[1] // num_slices  # 35 frames per slice

for i in range(num_slices):
    start = i * slice_size
    end = start + slice_size
    slice_data = magnitude[:, start:end]  # All freqs, specific time range

    # Analyze this slice
    print(f"Slice {i+1}: mean={slice_data.mean():.3f}")
```

**Key Syntax:** `magnitude[:, start:end]` - all frequencies, specific time frames

---

<!-- _class: lead -->
# Lab 3: librosa STFT
## From Waveforms to Spectrograms

**Team Connections:**
- Cameron's POC: Uses STFT throughout
- cervanj2: STFT in preprocessing, ISTFT in post-processing
- Complete round-trip: Audio → Spectrogram → Audio

---

## Lab 3: STFT is the Bridge Between Audio and Neural Networks

**Why spectrograms instead of waveforms?**

**The Problem:**
- Waveforms mix all frequencies into single amplitude values
- Neural networks struggle to separate overlapping sounds

**The Solution:**
- Spectrograms show frequency content over time (2D representation)
- Different instruments occupy different frequency ranges
- Vocals: 85-255 Hz fundamental + harmonics
- U-Net can learn to identify and separate frequency patterns

**Cameron's Insight:** Manual 18-slice spectrogram analysis achieved 70-80% separation

---

## Lab 3: The STFT Transformation

**Creating spectrograms with librosa.stft()**

```python
# Step 1: Load audio (Lab 1)
audio, sr = librosa.load('sample_audio.wav', sr=22050)

# Step 2: Apply STFT transformation
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)    # How loud each frequency is
phase = np.angle(stft)      # Phase info for reconstruction

print(f"Audio shape: {audio.shape}")         # (661500,)
print(f"Spectrogram shape: {magnitude.shape}")  # (1025, 646)
```

**The Magic:**
- 1D waveform (661,500 samples) → 2D spectrogram (1,025 × 646)
- STFT is reversible with ISTFT - perfect reconstruction!
- The POC keeps phase unchanged for high-quality separation

---

## Lab 3: The Round Trip - Perfect Reconstruction

**STFT is lossless - we can perfectly reconstruct audio**

```python
# Forward: Audio → Spectrogram
stft = librosa.stft(audio)

# Backward: Spectrogram → Audio
reconstructed = librosa.istft(stft, length=len(audio))

# Check quality
difference = np.abs(audio - reconstructed).max()
print(f"Max difference: {difference:.10f}")  # ~0.0000000012
```

**Why This Matters for Our Pipeline:**
1. Audio → STFT → Spectrogram (preprocessing)
2. U-Net processes spectrogram
3. Spectrogram → ISTFT → Separated audio (post-processing)

**cervanj2's Architecture:** STFT in preprocessing layer, ISTFT in post-processing layer

---

<!-- _class: lead -->
# Lab 4: PyTorch Essentials
## From NumPy Arrays to GPU Tensors

**Team Connections:**
- Ryan: Learned tensors, DataLoader, GPU usage
- Yovannoa: Trained neural networks with PyTorch
- cervanj2: PyTorch powers the Inference Layer (U-Net)

---

## Lab 4: PyTorch Enables GPU-Accelerated Deep Learning

**Why PyTorch for audio ML?**

**GPU Acceleration:** 10-100x faster than CPU for neural networks
**Automatic Differentiation:** Calculates gradients automatically for training
**Research-Friendly:** Easy to experiment and modify U-Net
**Ecosystem:** Our Pytorch-UNet repo uses PyTorch

**The Learning Path:**
- Ryan: Learned PyTorch basics with Fashion-MNIST (images)
- Yovannoa: Trained classifier on CIFAR-10 (54% accuracy, images)
- Our Project: Apply same concepts to audio spectrograms

**Key Insight:** Spectrograms are 2D images - image processing concepts transfer directly!

---

## Lab 4: Converting Spectrograms to PyTorch Tensors

**Complete pipeline: Audio → NumPy → PyTorch → GPU**

```python
import librosa, numpy as np, torch

# Labs 1-3: Get spectrogram
audio, sr = librosa.load('sample.wav', sr=22050)
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)  # NumPy array: (1025, 646)

# Lab 4: Convert to PyTorch tensor
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims

# Move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)

print(f"Shape: {tensor.shape}")  # torch.Size([1, 1, 1025, 646])
print(f"Device: {tensor.device}")  # cuda:0 or cpu
```

**U-Net Input Format:** (batch_size, channels, frequency, time)

---

<!-- _class: lead -->
# Lab 5: U-Net Architecture
## Automating Cameron's Manual Analysis

**Team Connections:**
- Cameron: POC validates concept (70-80% quality)
- cervanj2: U-Net is core of Inference Layer
- Ryan & Yovannoa: PyTorch knowledge enables implementation
- All Labs: Complete integration

---

## Lab 5: U-Net Learns What Cameron Does Manually

**What is U-Net?**
- Encoder-decoder neural network architecture
- Originally for medical image segmentation (2015)
- Perfect for spectrograms - they're 2D images!

**The Architecture:**
```
Input Spectrogram (1025 × 646)
        ↓
    Encoder (downsample, learn features)
        ↓
    Bottleneck (compressed representation)
        ↓
    Decoder (upsample, reconstruct)
        ↓
Output Separation Mask (1025 × 646)
```

**Cameron's POC vs. U-Net:**
- Cameron: Manual 18-slice analysis → 70-80% quality
- U-Net: Automatic learning → potential 85-95% quality

---

## Lab 5: The Complete Pipeline - All Labs Integrated

**From audio file to separated stems in 7 steps:**

```python
# Step 1: Load audio (Lab 1)
audio, sr = librosa.load('mixed_song.wav', sr=22050)

# Step 2: NumPy preprocessing (Lab 2)
audio = audio / np.abs(audio).max()

# Step 3: Create spectrogram (Lab 3)
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude, phase = np.abs(stft), np.angle(stft)

# Step 4: Convert to PyTorch (Lab 4)
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

# Step 5: U-Net processing (Lab 5)
vocal_mask = unet_model(tensor)  # Learns separation
music_mask = 1 - vocal_mask

# Step 6: Apply masks & reconstruct (Lab 3 reverse)
vocals = librosa.istft(magnitude * vocal_mask * np.exp(1j * phase))
music = librosa.istft(magnitude * music_mask * np.exp(1j * phase))

# Step 7: Save separated stems
# sf.write('vocals.wav', vocals, sr)
```

---

## Every Team Member Contributed to This Pipeline

**Cameron's POC:**
- Validates the concept works (70-80% quality)
- Manual analysis shows what U-Net should learn
- First step: loading audio with librosa

**cervanj2's Architecture:**
- Preprocessing Layer: Labs 1-3 (load, NumPy, STFT)
- Inference Layer: Lab 5 (U-Net)
- Post-processing Layer: Labs 3-4 reversed (ISTFT, save)

**Ryan's Tutorial:**
- GPU acceleration ✓
- Tensor operations ✓
- Simple network structure ✓

**Yovannoa's Tutorial:**
- Autograd for training ✓
- Training loop structure ✓
- Full classifier pipeline ✓

**Haedon's Philosophy:**
- Deep modules: U-Net hides complexity
- Clean interfaces: librosa.load(), unet.forward()

---

## Next Steps: Weeks 6-7-8

**What We've Accomplished:**
- ✅ Understood complete pipeline (5 labs)
- ✅ Connected all team member work
- ✅ Validated concept with Cameron's POC
- ✅ Forked Pytorch-UNet repository

**What's Next:**
1. **Week 6:** Prepare audio training datasets
2. **Week 7:** Train U-Net on paired vocal/music data
3. **Week 8:** Evaluate, optimize, and deploy

**Training Requirements:**
- Paired data: mixed audio + isolated stems
- Datasets: MUSDB18 or similar
- Hardware: GPU (Ryan's RTX 2070 or cloud GPU)
- Time: Hours to days depending on dataset size

**Expected Results:** Well-trained U-Net can achieve 85-95% separation quality

---

<!-- _class: lead -->
# Summary: 5 Labs, 1 Complete Pipeline

**Lab 1:** Load audio files with librosa → NumPy arrays
**Lab 2:** Process with NumPy → 765,000 measurements
**Lab 3:** Transform with STFT → 2D spectrograms
**Lab 4:** Convert to PyTorch → GPU-accelerated tensors
**Lab 5:** Process with U-Net → Separated vocals and music

**Every lab builds on the previous**
**Every team member is represented**
**Ready for weeks 6-7-8 U-Net training**

---

<!-- _class: lead -->
# Questions?

**Presentation Materials:**
- 5 detailed lab documents (7 pages each)
- Demo scripts with timing (3-5 min per lab)
- Code files ready to run
- Q&A prep for common questions

**Contact:** Available for clarifications

---

<!-- _class: lead -->
# Appendix: Technical Details

Available for deep-dive questions

---

## Appendix: STFT Parameters Explained

**Understanding the transformation parameters:**

```python
stft = librosa.stft(
    audio,
    n_fft=2048,      # Window size: frequency resolution
    hop_length=512,  # Step size: time resolution
    window='hann'    # Window function: reduces artifacts
)
```

**n_fft (2048):**
- Determines frequency bins: 2048/2 + 1 = 1,025 bins
- Larger = better frequency resolution, worse time resolution
- Cameron uses default 2048

**hop_length (512):**
- How far window shifts each frame
- Smaller = better time resolution, more frames
- 75% overlap (2048 - 512 = 1,536)

**Trade-off:** Frequency resolution ↔ Time resolution

---

## Appendix: U-Net Architecture Details

**Encoder-Decoder Structure:**

**Encoder (Downsampling):**
- Multiple convolutional layers
- Learns features at different scales
- MaxPooling reduces spatial dimensions

**Bottleneck:**
- Compressed representation
- Highest-level features

**Decoder (Upsampling):**
- Transposed convolutions
- Reconstructs spatial dimensions
- Skip connections from encoder preserve detail

**Skip Connections:**
- Copy features from encoder to decoder
- Preserve fine-grained details
- Critical for high-quality audio separation

---

## Appendix: Training Data Requirements

**What U-Net needs to learn:**

**Paired Training Data:**
- Input: Mixed audio spectrograms (vocals + instruments)
- Target: Isolated vocal spectrograms (ground truth)

**Popular Datasets:**
- MUSDB18: 150 songs, 10 hours, multiple stems
- DSD100: 100 songs, development set
- Custom: Record/source isolated stems

**Training Process:**
1. Load paired data
2. Forward pass: U-Net predicts separation
3. Calculate loss: How different from ground truth?
4. Backward pass: Update model weights
5. Repeat for many epochs

**Time:** Hours to days depending on dataset size and GPU

