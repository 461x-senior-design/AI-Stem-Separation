# Lab 4 TLDR: PyTorch Essentials

## Overview
**Duration:** 3-5 minutes  
**Focus:** Converting spectrograms to GPU-accelerated PyTorch tensors

## Key Learnings
- PyTorch tensors = NumPy arrays with GPU acceleration
- Automatic differentiation for training neural networks
- Tensor shape requirements for U-Net: (batch, channels, freq, time)
- Seamless NumPy ↔ PyTorch conversion

## Core Code Concepts
```python
import torch

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NumPy to PyTorch
tensor = torch.from_numpy(spectrogram).float()
tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
tensor = tensor.to(device)  # Move to GPU

# Operations on GPU
processed = torch.relu(tensor)  # Neural network operations
```

## Team Connections
- **Ryan's Tutorial:** GPU access, tensor basics, simple networks
- **Yovannoa's Tutorial:** Autograd, training loops, complete workflow
- **cervanj2's Architecture:** PyTorch powers the Inference Layer (U-Net)

## Success Criteria
- Check for GPU availability and setup
- Convert NumPy arrays to PyTorch tensors
- Move tensors between CPU and GPU
- Understand 4D tensor shape requirements for U-Net

## Pipeline Position
Spectrogram (NumPy) → PyTorch tensor (GPU) → Ready for U-Net processing
