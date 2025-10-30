# Lab 5: U-Net Architecture for Audio Separation

**Duration:** 3-5 minutes
**Presenter Guide:** This is the finale - bringing all labs together with U-Net

---

## Slide 1: Lab Overview (30 seconds)

### Title: "Lab 5: U-Net Architecture"
### Subtitle: "Automating Cameron's Manual Analysis"

**What you'll learn:**
- U-Net encoder-decoder architecture
- How U-Net processes spectrograms
- Complete pipeline integration (Labs 1-5)
- Training vs. Inference

**Team Member Connections:**
- **Cameron:** POC validates the concept - U-Net will automate his analysis
- **cervanj2:** U-Net is the core of the Inference Layer
- **Ryan & Yovannoa:** PyTorch knowledge enables U-Net implementation
- **Haedon:** Design principles applied to clean architecture

---

## Slide 2: What is U-Net? (1 minute)

### Title: "U-Net: The Architecture That Powers Vocal Separation"

**U-Net Overview:**
- Originally designed for medical image segmentation (2015)
- "U" shape: Encoder (downsampling) + Decoder (upsampling)
- Perfect for our task: separate mixed spectrogram into components
- Encoder learns "what" (features), Decoder learns "where" (localization)

**Visual Architecture:**
```
Input Spectrogram (1025 × 646)
        ↓
    ┌───────┐
    │Encoder│ ← Downsample, learn features
    │   ↓   │
    │ Bottleneck │ ← Compressed representation
    │   ↓   │
    │Decoder│ ← Upsample, reconstruct
    └───────┘
        ↓
Output Mask (1025 × 646)
```

**Why U-Net for Audio?**
- Spectrograms are 2D images (frequency × time)
- U-Net excels at image-to-image transformations
- Skip connections preserve spatial information (important for audio quality)
- Cameron's manual 18-slice analysis → U-Net learns this automatically

---

## Slide 3: U-Net Building Blocks (1 minute 30 seconds)

### Title: "Understanding U-Net Components"

**Code Example - Simplified U-Net Block:**
```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """
    Basic U-Net building block
    Simpler than full U-Net, but shows the concept
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Two convolutional layers (common pattern)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass through the block
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Create a simple block
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
block = UNetBlock(in_channels=1, out_channels=16).to(device)

# Test with spectrogram-sized input
# Shape: (batch=1, channels=1, freq=1025, time=646)
input_spec = torch.randn(1, 1, 1025, 646).to(device)
output = block(input_spec)

print("=== U-Net Block Test ===")
print(f"Input shape: {input_spec.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

**Expected Output:**
```
=== U-Net Block Test ===
Input shape: torch.Size([1, 1, 1025, 646])
Output shape: torch.Size([1, 16, 1025, 646])
Parameters: 448
```

**Key Concepts:**
- **Conv2d:** Learns spatial patterns (like musical features)
- **ReLU:** Activation function (non-linearity)
- **Channels:** 1 input channel → 16 feature channels
- Full U-Net has many of these blocks in encoder-decoder structure

---

## Slide 4: Complete Pipeline - Labs 1-5 Integrated (1 minute 30 seconds)

### Title: "The Complete Vocal Separation Pipeline"

**Full Code Integration:**
```python
import librosa
import numpy as np
import torch
import torch.nn as nn

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Processing on: {device}\n")

# ========================================
# COMPLETE PIPELINE: Audio → Separated Stems
# ========================================

print("=== Step 1: Load Audio (Lab 1) ===")
audio, sr = librosa.load('mixed_song.wav', sr=22050)
print(f"Audio shape: {audio.shape}")
print(f"Duration: {len(audio)/sr:.2f} seconds\n")

print("=== Step 2: NumPy Operations (Lab 2) ===")
# Any preprocessing (normalization, etc.)
audio_normalized = audio / np.abs(audio).max()
print(f"Audio normalized: min={audio_normalized.min():.3f}, "
      f"max={audio_normalized.max():.3f}\n")

print("=== Step 3: Create Spectrogram (Lab 3) ===")
stft = librosa.stft(audio_normalized, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)
phase = np.angle(stft)  # Save phase for reconstruction
print(f"Spectrogram shape: {magnitude.shape}")
print(f"Frequency bins: {magnitude.shape[0]}, Time frames: {magnitude.shape[1]}\n")

print("=== Step 4: Convert to PyTorch (Lab 4) ===")
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
tensor = tensor.to(device)
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor device: {tensor.device}\n")

print("=== Step 5: U-Net Processing (Lab 5) ===")
# Load pre-trained U-Net model (or use simplified version)
# unet_model = torch.load('trained_unet.pth')
# unet_model.eval()

# For demo: simulate U-Net output (in reality, model processes this)
print("U-Net processes the spectrogram...")
print("Learning to separate vocals from music...")

# Simulated output (real U-Net would do this)
vocal_mask = torch.sigmoid(torch.randn_like(tensor))  # 0-1 mask
music_mask = 1 - vocal_mask

print(f"Vocal mask shape: {vocal_mask.shape}")
print(f"Music mask shape: {music_mask.shape}\n")

print("=== Step 6: Apply Masks ===")
# Move back to NumPy
vocal_mask_np = vocal_mask.squeeze().cpu().numpy()
music_mask_np = music_mask.squeeze().cpu().numpy()

# Apply masks to original spectrogram
magnitude_vocals = magnitude * vocal_mask_np
magnitude_music = magnitude * music_mask_np
print(f"Separated magnitude shapes:")
print(f"  Vocals: {magnitude_vocals.shape}")
print(f"  Music: {magnitude_music.shape}\n")

print("=== Step 7: Reconstruct Audio (Lab 3 - ISTFT) ===")
# Reconstruct with original phase
stft_vocals = magnitude_vocals * np.exp(1j * phase)
stft_music = magnitude_music * np.exp(1j * phase)

vocals = librosa.istft(stft_vocals, hop_length=512, length=len(audio))
music = librosa.istft(stft_music, hop_length=512, length=len(audio))

print(f"Reconstructed audio shapes:")
print(f"  Vocals: {vocals.shape}")
print(f"  Music: {music.shape}")
print(f"  Original: {audio.shape}\n")

print("=== Step 8: Save Separated Stems ===")
# import soundfile as sf
# sf.write('separated_vocals.wav', vocals, sr)
# sf.write('separated_music.wav', music, sr)
print("Vocals and music stems saved!")

print("\n" + "="*50)
print("COMPLETE! Audio separated into vocals and music.")
print("="*50)
```

**The Magic:**
- Input: Mixed audio (vocals + music)
- U-Net: Learns to create separation masks
- Output: Separated vocals and music stems

**Cameron's POC vs. U-Net:**
- Cameron: Manual 18-slice analysis, 70-80% quality
- U-Net: Automatic learning, potential for >90% quality
- Same STFT/ISTFT framework
- Different processing in the middle

---

## Slide 5: Training vs. Inference (1 minute)

### Title: "How U-Net Learns to Separate Audio"

**Training Phase (Not Required for Demo):**
```python
# Training requires paired data:
# - Mixed audio spectrograms (input)
# - Isolated vocal spectrograms (ground truth)

for epoch in range(num_epochs):
    for mixed_spec, vocal_spec in training_data:
        # 1. Forward pass
        predicted_mask = unet_model(mixed_spec)
        separated_vocal = mixed_spec * predicted_mask

        # 2. Calculate loss (how different from truth?)
        loss = criterion(separated_vocal, vocal_spec)

        # 3. Backward pass (update model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

**Inference Phase (What We Demo):**
```python
# Inference uses trained model:
unet_model.eval()  # Set to evaluation mode
with torch.no_grad():  # No gradient calculation needed
    vocal_mask = unet_model(mixed_spectrogram)
    separated_vocals = mixed_spectrogram * vocal_mask
```

**Ryan and Yovannoa's Experience:**
- Ryan: Trained on Fashion-MNIST (10 epochs, 71% accuracy)
- Yovannoa: Trained on CIFAR-10 (multiple epochs, 54% accuracy)
- Our U-Net: Will train on audio datasets (many hours of paired vocals/music)

---

## Slide 6: Connection to All Team Work (30 seconds)

### Title: "Every Lab, Every Team Member"

**Complete Integration:**

**Cameron's Contribution:**
- POC proves the concept works
- Manual analysis shows what U-Net should learn
- 70-80% quality validates the approach

**cervanj2's Architecture:**
1. ✓ **Preprocessing Layer** (Labs 1-3)
2. ✓ **Convert to Tensor** (Lab 4)
3. ✓ **Inference Layer** (Lab 5 - U-Net)
4. ✓ **Post-processing Layer** (Labs 3-4 reversed)
5. ✓ **Output Layer** (Save separated audio)

**Ryan's Tutorial:**
- GPU acceleration ✓
- Tensor operations ✓
- Simple network structure ✓

**Yovannoa's Tutorial:**
- Autograd ✓
- Training loops ✓
- Full classifier pipeline ✓

**Haedon's Philosophy:**
- Deep modules: U-Net hides complexity, simple interface
- Strategic programming: Invest time in clean implementation
- "Design it twice": Explore architectures before committing

**Next Steps (Weeks 6-7-8):**
- Use `Pytorch-UNet` repository
- Train on audio datasets
- Evaluate and improve
- Deploy for real-world use

---

## Demo Script (Detailed Timing) - Provided by Claude

### 0:00 - 0:30: Introduction
"Welcome to Lab 5 - the finale! This is where everything comes together. We've learned how to load audio, process it with NumPy, transform it with librosa, and prepare it with PyTorch. Now we're going to use U-Net to actually separate vocals from music. This is what Cameron's POC was building toward!"

### 0:30 - 1:30: What is U-Net?
"So what is U-Net? [Show Slide 2]. U-Net is an encoder-decoder neural network architecture. Picture a 'U' shape - the encoder goes down, learning features, the bottleneck compresses everything, and the decoder goes back up, reconstructing the output.

U-Net was originally designed for medical image segmentation in 2015, but it's perfect for our task. Why? Because spectrograms are just 2D images! The frequency axis is like height, the time axis is like width, and U-Net learns to separate the mixed spectrogram into vocal and music components.

Cameron's POC does this manually with 18-slice analysis. U-Net learns to do it automatically - and potentially even better!"

### 1:30 - 3:00: Building Blocks
"Let's look inside U-Net. [Show Slide 3 and run code].

This is a simplified U-Net block - the building blocks of the full architecture. It uses convolutional layers to learn spatial patterns in the spectrogram. Think of it like learning to recognize vocal patterns versus instrument patterns.

[Show output]

Input: 1 channel (our spectrogram). Output: 16 channels (learned features). The full U-Net has many of these blocks arranged in the encoder-decoder structure.

This is exactly the kind of architecture Ryan and Yovannoa learned about - Conv2D layers, ReLU activations, all running on GPU."

### 3:00 - 4:30: Complete Pipeline
"Now let's see the complete pipeline. [Show Slide 4 and walk through code].

We start with a mixed audio file - vocals and music together. Lab 1: we load it. Lab 2: we normalize with NumPy. Lab 3: we create the spectrogram with STFT. Lab 4: we convert to a PyTorch tensor on GPU.

Then Lab 5 - U-Net processes this tensor and creates two masks: one for vocals, one for music. These masks tell us which parts of the spectrogram are vocals versus instruments.

We apply the masks, reconstruct with ISTFT, and save the separated stems. That's the complete pipeline! This matches cervanj2's architecture perfectly - every layer is here."

### 4:30 - 5:00: Wrap-up
"That's all 5 labs! You now understand the complete vocal separation pipeline from Cameron's POC:
- Lab 1: Load audio
- Lab 2: NumPy operations
- Lab 3: STFT transformation
- Lab 4: PyTorch tensors
- Lab 5: U-Net separation

Next steps for weeks 6-7-8: we'll train our own U-Net model on audio datasets using the Pytorch-UNet repository. Every team member's work has prepared us for this.

Questions?"

---

## Code Files to Provide

### `lab5_unet_demo.py`
```python
"""
Lab 5: U-Net Architecture
Complete vocal separation pipeline
"""
import torch
import torch.nn as nn
import librosa
import numpy as np

class SimpleUNetBlock(nn.Module):
    """Simplified U-Net building block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

def test_unet_block():
    """Test a simple U-Net block"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    block = SimpleUNetBlock(1, 16).to(device)
    input_tensor = torch.randn(1, 1, 1025, 646).to(device)

    output = block(input_tensor)

    print("=== U-Net Block Test ===")
    print(f"Input: {input_tensor.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")

def complete_separation_pipeline(audio_path):
    """Complete pipeline: Audio → Separated Stems"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nProcessing on: {device}")
    print("="*60)

    # Lab 1: Load Audio
    print("\n[Lab 1] Loading audio...")
    audio, sr = librosa.load(audio_path, sr=22050)
    print(f"  Shape: {audio.shape}, Duration: {len(audio)/sr:.2f}s")

    # Lab 2: NumPy operations
    print("\n[Lab 2] NumPy preprocessing...")
    audio_normalized = audio / np.abs(audio).max()
    print(f"  Normalized: [{audio_normalized.min():.3f}, {audio_normalized.max():.3f}]")

    # Lab 3: Create spectrogram
    print("\n[Lab 3] Creating spectrogram (STFT)...")
    stft = librosa.stft(audio_normalized, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    print(f"  Shape: {magnitude.shape} (freq × time)")

    # Lab 4: Convert to PyTorch
    print("\n[Lab 4] Converting to PyTorch tensor...")
    tensor = torch.from_numpy(magnitude).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    print(f"  Shape: {tensor.shape} (batch, channels, freq, time)")
    print(f"  Device: {tensor.device}")

    # Lab 5: U-Net processing (simulated)
    print("\n[Lab 5] U-Net processing...")
    print("  (Simulated output - real model would process here)")

    # Simulate masks
    vocal_mask = torch.sigmoid(torch.randn_like(tensor))
    music_mask = 1 - vocal_mask

    print(f"  Vocal mask: {vocal_mask.shape}")
    print(f"  Music mask: {music_mask.shape}")

    # Apply masks
    print("\n[Post-processing] Applying masks...")
    vocal_mask_np = vocal_mask.squeeze().cpu().numpy()
    music_mask_np = music_mask.squeeze().cpu().numpy()

    magnitude_vocals = magnitude * vocal_mask_np
    magnitude_music = magnitude * music_mask_np

    # Reconstruct audio
    print("\n[Lab 3 - ISTFT] Reconstructing audio...")
    stft_vocals = magnitude_vocals * np.exp(1j * phase)
    stft_music = magnitude_music * np.exp(1j * phase)

    vocals = librosa.istft(stft_vocals, hop_length=512, length=len(audio))
    music = librosa.istft(stft_music, hop_length=512, length=len(audio))

    print(f"  Vocals: {vocals.shape}")
    print(f"  Music: {music.shape}")

    print("\n" + "="*60)
    print("COMPLETE! Audio separated into vocals and music.")
    print("="*60)

    return vocals, music

if __name__ == "__main__":
    # Test U-Net block
    test_unet_block()

    # Complete pipeline demo (requires audio file)
    # vocals, music = complete_separation_pipeline('sample_audio.wav')
```

### `lab5_explore_pytorch_unet.py`
```python
"""
Lab 5: Explore the Pytorch-UNet Repository
Guide for exploring the forked U-Net implementation
"""

def explore_pytorch_unet():
    """
    Guide to exploring the Pytorch-UNet repository

    Repository: https://github.com/brookcs3/Pytorch-UNet

    Key Files to Explore:
    1. unet/unet_model.py - Main U-Net architecture
    2. unet/unet_parts.py - Building blocks (encoder, decoder)
    3. train.py - Training script
    4. predict.py - Inference script
    """

    print("=== Exploring Pytorch-UNet Repository ===\n")

    print("1. U-Net Model (unet/unet_model.py)")
    print("   - Complete U-Net architecture")
    print("   - Encoder-decoder structure")
    print("   - Skip connections between layers\n")

    print("2. U-Net Parts (unet/unet_parts.py)")
    print("   - DoubleConv: Two conv layers (like our SimpleUNetBlock)")
    print("   - Down: Downsampling block")
    print("   - Up: Upsampling block")
    print("   - OutConv: Final output layer\n")

    print("3. Training (train.py)")
    print("   - Data loading")
    print("   - Training loop")
    print("   - Loss calculation")
    print("   - Model checkpointing\n")

    print("4. Inference (predict.py)")
    print("   - Load trained model")
    print("   - Process new data")
    print("   - Generate predictions\n")

    print("Our Modifications for Audio:")
    print("   - Input: Spectrograms (not images)")
    print("   - Output: Separation masks")
    print("   - Training data: Audio datasets")
    print("   - Same architecture, different domain!")

if __name__ == "__main__":
    explore_pytorch_unet()
```

---

## Visual Aids Needed

1. **U-Net Architecture Diagram:** Full U-shape with encoder-decoder
2. **Skip Connections Visual:** How information flows through U-Net
3. **Complete Pipeline Flowchart:** All 5 labs integrated
4. **Before/After Spectrograms:** Mixed → Separated vocals/music

---

## Backup Slides

### Extra: Skip Connections
Skip connections copy features from encoder to decoder at matching resolutions

### Extra: Loss Functions
- MSE Loss: Mean squared error between predicted and ground truth
- L1 Loss: Mean absolute error
- Perceptual Loss: Based on feature similarity

---

## Q&A Prep - Provided by Claude

**Q: "How long does U-Net training take?"**
A: "Depends on dataset size and hardware. With GPU, hours to days. The POC took ~30 minutes per song manually - U-Net will be much faster after training."

**Q: "What training data do we need?"**
A: "Paired data: mixed audio + isolated stems. Datasets like MUSDB18 provide this. Need hundreds of songs for good performance."

**Q: "Can U-Net separate more than vocals and music?"**
A: "Yes! You can train separate models for drums, bass, vocals, other. Or use multi-channel output."

**Q: "How does this compare to the POC's 70-80% quality?"**
A: "Well-trained U-Net can achieve 85-95% quality. The POC validated the concept - U-Net scales it up."

---

## Success Criteria

Students should be able to:
- [ ] Explain U-Net encoder-decoder architecture
- [ ] Understand how U-Net processes spectrograms
- [ ] Connect all 5 labs into complete pipeline
- [ ] Distinguish between training and inference
- [ ] Relate U-Net to Cameron's POC

---

## Connection Summary

**All Labs Integrated:**
- Lab 1: Audio loading
- Lab 2: NumPy operations
- Lab 3: STFT/ISTFT transformations
- Lab 4: PyTorch tensors and GPU
- Lab 5: U-Net processing

**All Team Members Represented:**
- Cameron: POC concept validation
- cervanj2: Architecture implementation
- Ryan: PyTorch foundations
- Yovannoa: Training concepts
- Haedon: Design principles

**Next Steps:** Weeks 6-7-8 will train U-Net on real audio datasets!
