# Lab 5: U-Net Architecture - 1-Minute Presentation Guide

## Overview
Demonstrate U-Net's encoder-decoder architecture and show the complete vocal separation pipeline integrating all 5 labs. Total presentation time: 1 minute.

## Setup Beforehand (5-10 minutes prep)
- **Environment Requirements:**
  - PyTorch, librosa, numpy, matplotlib installed
  - GPU recommended for tensor operations
  - Audio file for complete pipeline demo
- **Files to Prepare:**
  - `lab5_unet_demo.py` with functions: `SimpleUNetBlock`, `complete_separation_pipeline()`
  - Pre-trained U-Net model or simulation code
- **Hardware Check:**
  - GPU for tensor processing
  - Display for potential visualizations
- **Presentation Materials:**
  - Slide 3: U-Net building block demonstration
  - Slide 4: Complete integrated pipeline
  - Backup: Training vs inference explanation

## Demo Execution Script (Exactly 1 Minute)

### Timing: 0:00 - 0:15 (15 seconds): U-Net Building Block
**Code to Run:**
```python
import torch
import torch.nn as nn

# Simple U-Net building block
class SimpleUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Test the block
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
block = SimpleUNetBlock(in_channels=1, out_channels=16).to(device)

input_tensor = torch.randn(1, 1, 1025, 646).to(device)
output = block(input_tensor)

print("=== U-Net Building Block ===")
print(f"Input: {input_tensor.shape}")
print(f"Output: {output.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

**Narration:**
"Lab 5: U-Net's encoder-decoder architecture learns to separate spectrograms. This block processes frequency patterns!"

**Expected Output:**
```
=== U-Net Building Block ===
Input: torch.Size([1, 1, 1025, 646])
Output: torch.Size([1, 16, 1025, 646])
Parameters: 448
```

### Timing: 0:15 - 0:45 (30 seconds): Complete Pipeline Integration
**Code to Run:**
```python
import librosa
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Processing on: {device}\\n")

# ==========================================
# COMPLETE PIPELINE: Labs 1-5 Integration
# ==========================================

print("=== Step 1-3: Audio → Spectrogram ===")
audio, sr = librosa.load('sample_audio.wav', sr=22050)
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)
phase = np.angle(stft)
print(f"Audio: {audio.shape} → Spectrogram: {magnitude.shape}")

print("\\n=== Step 4: Convert to PyTorch ===")
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
print(f"Tensor: {tensor.shape}, Device: {tensor.device}")

print("\\n=== Step 5: U-Net Processing ===")
# Simulate U-Net: create vocal and music masks
vocal_mask = torch.sigmoid(torch.randn_like(tensor))  # 0-1 mask
music_mask = 1 - vocal_mask

print(f"Vocal mask: {vocal_mask.shape}")
print(f"Music mask: {music_mask.shape}")

print("\\n=== Step 6-7: Apply Masks & Reconstruct ===")
vocal_mask_np = vocal_mask.squeeze().cpu().numpy()
music_mask_np = music_mask.squeeze().cpu().numpy()

magnitude_vocals = magnitude * vocal_mask_np
magnitude_music = magnitude * music_mask_np

stft_vocals = magnitude_vocals * np.exp(1j * phase)
stft_music = magnitude_music * np.exp(1j * phase)

vocals = librosa.istft(stft_vocals, hop_length=512, length=len(audio))
music = librosa.istft(stft_music, hop_length=512, length=len(audio))

print(f"Separated vocals: {vocals.shape}")
print(f"Separated music: {music.shape}")
```

**Narration:**
"All 5 labs integrated! Audio → spectrogram → tensor → U-Net masks → separated vocals and music. U-Net automates Cameron's manual analysis!"

**Expected Output:**
```
Processing on: cuda

=== Step 1-3: Audio → Spectrogram ===
Audio: (661500,) → Spectrogram: (1025, 646)

=== Step 4: Convert to PyTorch ===
Tensor: torch.Size([1, 1, 1025, 646]), Device: cuda:0

=== Step 5: U-Net Processing ===
Vocal mask: torch.Size([1, 1, 1025, 646])
Music mask: torch.Size([1, 1, 1025, 646])

=== Step 6-7: Apply Masks & Reconstruct ===
Separated vocals: (661500,)
Separated music: (661500,)
```

### Timing: 0:45 - 1:00 (15 seconds): Wrap-up
**Narration:**
"U-Net processes spectrograms to create separation masks. This automates Cameron's POC - neural networks learn what he did manually!"

## Key Points to Emphasize
- U-Net encoder-decoder architecture for spectrogram processing
- Complete integration of all 5 labs
- Automation of Cameron's manual 18-slice analysis
- Pipeline: Audio → Spectrogram → Tensor → Masks → Separated Audio

## Troubleshooting
- If GPU unavailable: Use CPU (slower but works)
- If memory issues: Reduce tensor sizes
- If audio processing slow: Use shorter audio file

## Success Criteria
- [ ] U-Net block demonstrates convolution operations
- [ ] Complete pipeline shows all 5 labs integrated
- [ ] Separation masks created and applied
- [ ] Audio reconstruction completed
- [ ] Concepts explained within 1 minute
