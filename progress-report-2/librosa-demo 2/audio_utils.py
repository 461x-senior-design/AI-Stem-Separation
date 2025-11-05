"""
Audio Input and Visualization Utilities

Functions for loading audio and creating visualizations.
Based on the original audio_utils.py but enhanced for demo workflow.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile  # Enable FLAC support through librosa
import subprocess
import sys

# Create output directories
os.makedirs("figures", exist_ok=True)

def check_ffmpeg():
    """Check if ffmpeg is installed (required for FLAC support)"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def load_audio(path, sr, mono):
    """
    Load audio file with librosa.

    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono (True/False)

    Returns:
        audio: Audio array
        sr: Sample rate
    """
    # Check if FLAC file and ffmpeg is needed
    if path.lower().endswith('.flac'):
        if not check_ffmpeg():
            print("ERROR: FLAC file detected but ffmpeg is not installed.")
            print("Install ffmpeg to load FLAC files:")
            print("  macOS: brew install ffmpeg")
            print("  Linux: sudo apt-get install ffmpeg")
            sys.exit(1)

    audio, sr = librosa.load(path, sr=sr, mono=mono)
    channels = 1 if audio.ndim == 1 else audio.shape[0]
    print(f"Loaded {path} - {len(audio)/sr:.2f}s, {sr}Hz, {channels}ch")
    return audio, sr

def plot_waveform(audio, sr, duration_s, start_s, plot_width, plot_height, y_min, y_max,
                 output_file, title_suffix):
    """
    Plot a waveform segment and save to file.

    Args:
        audio: Audio array
        sr: Sample rate
        duration_s: Duration to show in seconds
        start_s: Start time in seconds
        output_file: Output file path
        title_suffix: Suffix for plot title
    """
    # Extract segment
    start_idx = int(start_s * sr)
    end_idx = int((start_s + duration_s) * sr)
    segment = audio[start_idx:end_idx]

    # Create plot
    plt.figure(figsize=(plot_width, plot_height))
    plt.plot(segment)
    plt.title(f"Waveform{title_suffix} ({duration_s}s segment)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plotted {len(segment)} samples → {duration_s:.3f} seconds")
    print(f"Saved waveform to {output_file}")

def show_spectrogram(audio, sr, n_fft, hop_length, output_file, title_suffix):
    """
    Create and save spectrogram visualization.

    Args:
        audio: Audio array
        sr: Sample rate
        output_file: Output file path
        title_suffix: Suffix for plot title
    """
    # Compute STFT
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Create plot
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        sr=sr,
        x_axis="time",
        y_axis="hz"
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram{title_suffix}")
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Created spectrogram: {S.shape[0]} freq bins × {S.shape[1]} time frames")
    print(f"Saved spectrogram to {output_file}")