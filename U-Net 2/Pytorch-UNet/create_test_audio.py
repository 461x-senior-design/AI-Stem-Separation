"""
Create dummy audio files for testing the sanity check
"""
import numpy as np
import soundfile as sf

print("Creating test audio files...")

# Create 5 seconds of audio at 22050 Hz
sr = 22050
duration = 5.0
samples = int(sr * duration)

# Create a simple sine wave as "vocal" (500 Hz)
t = np.linspace(0, duration, samples)
vocal = 0.3 * np.sin(2 * np.pi * 500 * t)

# Create a mix with vocal + bass (100 Hz) + drums (noise)
bass = 0.2 * np.sin(2 * np.pi * 100 * t)
drums = 0.1 * np.random.randn(samples)
mixture = vocal + bass + drums

# Normalize
vocal = vocal / np.max(np.abs(vocal))
mixture = mixture / np.max(np.abs(mixture))

# Save
sf.write('Intergalactic_Acapella.wav', vocal, sr)
sf.write('Intergalactic_Stereo.wav', mixture, sr)

print("✓ Created:")
print("  Intergalactic_Acapella.wav (test vocal)")
print("  Intergalactic_Stereo.wav (test mixture)")
print("\nThese are dummy files for testing only.")
print("For real results, use actual music files.")
print("\nNow run: python prepare_audio_files.py")
