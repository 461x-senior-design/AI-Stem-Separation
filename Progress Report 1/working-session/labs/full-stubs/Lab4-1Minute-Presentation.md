# Lab 4: PyTorch Essentials - 1-Minute Presentation Guide

## Overview
Show PyTorch tensor creation, GPU acceleration, and the complete pipeline from NumPy spectrograms to GPU-ready tensors for U-Net. Total presentation time: 1 minute.

## Setup Beforehand (5-10 minutes prep)
- **Environment Requirements:**
  - PyTorch installed (with CUDA if GPU available)
  - NumPy, librosa available
  - Audio file for complete pipeline demo
- **Files to Prepare:**
  - `lab4_pytorch_basics.py` with functions: `check_pytorch_setup()`, `numpy_to_pytorch_demo()`, `audio_to_tensor_pipeline()`
  - Ensure PyTorch can access GPU
- **Hardware Check:**
  - GPU availability (recommended for demo)
  - Sufficient memory for tensor operations
- **Presentation Materials:**
  - Slide 3: Tensor creation and GPU transfer
  - Slide 4: Complete pipeline integration
  - Backup: PyTorch operations examples

## Demo Execution Script (Exactly 1 Minute)

### Timing: 0:00 - 0:15 (15 seconds): PyTorch Setup Check
**Code to Run:**
```python
import torch
import numpy as np

print("=== PyTorch Setup Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\\nSelected device: {device}")
```

**Narration:**
"Lab 4: PyTorch enables GPU acceleration! This provides the speed needed for U-Net neural networks."

**Expected Output (with GPU):**
```
=== PyTorch Setup Check ===
PyTorch version: 2.0.0
CUDA available: True
CUDA version: 12.1
GPU device: NVIDIA GeForce RTX 2070 Super Max-Q

Selected device: cuda
```

### Timing: 0:15 - 0:35 (20 seconds): NumPy to PyTorch Conversion
**Code to Run:**
```python
# Simulate spectrogram data (1025 freq × 646 time)
spectrogram_np = np.random.randn(1025, 646).astype(np.float32)
print(f"NumPy array: shape={spectrogram_np.shape}, dtype={spectrogram_np.dtype}")

# Convert to PyTorch tensor
tensor_cpu = torch.from_numpy(spectrogram_np)
print(f"PyTorch tensor (CPU): {tensor_cpu.shape}, {tensor_cpu.dtype}")

# Move to GPU
tensor_gpu = tensor_cpu.to(device)
print(f"PyTorch tensor (GPU): {tensor_gpu.shape}, device={tensor_gpu.device}")

# Basic neural operation
processed = torch.relu(tensor_gpu) * 0.5
print(f"After processing: {processed.shape}, device={processed.device}")

# Back to NumPy
result_np = processed.cpu().numpy()
print(f"Back to NumPy: {result_np.shape}")
```

**Narration:**
"Seamless conversion: NumPy spectrograms become PyTorch tensors on GPU. Neural operations happen instantly!"

**Expected Output:**
```
NumPy array: shape=(1025, 646), dtype=float32
PyTorch tensor (CPU): torch.Size([1025, 646]), torch.float32
PyTorch tensor (GPU): torch.Size([1025, 646]), device=cuda:0
After processing: torch.Size([1025, 646]), device=cuda:0
Back to NumPy: (1025, 646)
```

### Timing: 0:35 - 0:55 (20 seconds): Complete Pipeline Integration
**Code to Run:**
```python
import librosa

# Step 1-3: Audio → Spectrogram (Labs 1-3)
audio, sr = librosa.load('sample_audio.wav', sr=22050)
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)

print("=== Complete Pipeline: Audio → GPU Tensor ===")
print(f"1. Audio loaded: {audio.shape}")
print(f"2. Spectrogram: {magnitude.shape}")

# Step 4: Convert to PyTorch (Lab 4)
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
tensor = tensor.to(device)

print(f"3. PyTorch tensor: {tensor.shape}")
print(f"   Batch: {tensor.shape[0]}, Channels: {tensor.shape[1]}")
print(f"   Frequency: {tensor.shape[2]}, Time: {tensor.shape[3]}")
print(f"   Device: {tensor.device}")
print("\\nReady for U-Net processing!")
```

**Narration:**
"Complete integration: Audio file → NumPy spectrogram → PyTorch tensor on GPU. This is what U-Net processes!"

**Expected Output:**
```
=== Complete Pipeline: Audio → GPU Tensor ===
1. Audio loaded: (661500,)
2. Spectrogram: (1025, 646)
3. PyTorch tensor: torch.Size([1, 1, 1025, 646])
   Batch: 1, Channels: 1
   Frequency: 1025, Time: 646
   Device: cuda:0

Ready for U-Net processing!
```

### Timing: 0:55 - 1:00 (5 seconds): Wrap-up
**Narration:**
"PyTorch bridges audio processing to neural networks - GPU acceleration makes U-Net fast!"

## Key Points to Emphasize
- GPU acceleration benefits for neural networks
- Seamless NumPy ↔ PyTorch conversion
- 4D tensor shape requirements (batch, channels, height, width)
- Integration with Ryan and Yovannoa's PyTorch learning

## Troubleshooting
- If no GPU: Code works on CPU (just slower)
- If CUDA errors: Check PyTorch installation
- If memory issues: Use smaller tensors or shorter audio

## Success Criteria
- [ ] PyTorch setup verified with GPU detection
- [ ] NumPy array successfully converted to GPU tensor
- [ ] Complete pipeline shows all transformations
- [ ] GPU acceleration demonstrated within 1 minute
