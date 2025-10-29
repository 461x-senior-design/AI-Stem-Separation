# Lab 5 TLDR: U-Net Architecture for Audio Separation

## Overview
**Duration:** 3-5 minutes  
**Focus:** Using U-Net neural network to separate vocals from music

## Key Learnings
- U-Net: Encoder-decoder architecture for image-to-image tasks
- Spectrograms as 2D images processed by convolutional layers
- Training vs inference phases
- Complete pipeline integration (Labs 1-5)

## Core Code Concepts
```python
# Complete pipeline
audio → librosa.load() → NumPy ops → STFT → PyTorch tensor → U-Net → ISTFT → separated audio

# U-Net processing
unet_model.eval()
with torch.no_grad():
    vocal_mask = unet_model(spectrogram_tensor)
    separated_vocals = spectrogram * vocal_mask
```

## Team Connections
- **Cameron's POC:** Manual analysis → U-Net automation
- **cervanj2's Architecture:** U-Net is the Inference Layer
- **Ryan & Yovannoa:** PyTorch concepts enable U-Net implementation
- **Haedon:** Clean architecture design principles

## Success Criteria
- Explain U-Net encoder-decoder structure
- Understand spectrogram processing as image task
- Connect all 5 labs into complete pipeline
- Distinguish training vs inference workflows

## Pipeline Position
PyTorch tensor → U-Net processing → Separation masks → Reconstructed audio stems
