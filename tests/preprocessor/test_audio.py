# Test script - run from project root
import numpy as np
from pathlib import Path
from src.preprocessing.audio import load_audio

# Test with a real audio file (replace with your path)
base_path = Path(__file__).parent
audio_path = base_path / "samples" / "plinky_key.wav"
waveform, sr = load_audio(audio_path)

print(f"Sample rate: {sr}")
print(f"Shape: {waveform.shape}")
print(f"Duration: {waveform.shape[-1] / sr:.2f} seconds")
print(f"Dtype: {waveform.dtype}")
print(f"Range: [{waveform.min():.3f}, {waveform.max():.3f}]")

# Verify stereo
if waveform.ndim == 2:
    print(f"✓ Stereo: {waveform.shape[0]} channels")
else:
    print(f"Mono: shape {waveform.shape}")
