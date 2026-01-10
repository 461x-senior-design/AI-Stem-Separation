#!/usr/bin/env python3
"""
Audio File Preparation Script
==============================
This script will:
1. Find your audio files in the pre/process/100-window/ directory
2. Convert FLAC to WAV temporarily if needed (using ffmpeg)
3. Standardize format (mono, 22050Hz, normalized, 4.7 seconds)
4. Save to pre/rtg/100-window/ directory

Place your files in:
- pre/process/100-window/ (files named *_100-full.wav/.flac and *_100-stem.wav/.flac)
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import subprocess
import os

print("="*70)
print("AUDIO FILE PREPARATION")
print("="*70)

def check_ffmpeg():
    """Check if ffmpeg is installed (required for FLAC support)"""
    # Try different possible ffmpeg locations
    ffmpeg_paths = [
        "ffmpeg",  # System PATH
        "/opt/homebrew/bin/ffmpeg",  # Homebrew on Apple Silicon
        "/usr/local/bin/ffmpeg",  # Homebrew on Intel Mac
        "/usr/bin/ffmpeg",  # System install
    ]
    
    for ffmpeg_path in ffmpeg_paths:
        try:
            subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
            return ffmpeg_path  # Return the working path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    return None  # No ffmpeg found

# Directories - work from current directory
BASE_DIR = Path.cwd()
PROCESS_100_DIR = BASE_DIR / "pre" / "process" / "100-window"
RTG_100_DIR = BASE_DIR / "pre" / "rtg" / "100-window"

# Create directories if they don't exist
PROCESS_100_DIR.mkdir(parents=True, exist_ok=True)
RTG_100_DIR.mkdir(parents=True, exist_ok=True)

# Config
TARGET_SR = 22050  # Sample rate
TARGET_DURATION = 4.7  # seconds

def find_files(directory):
    """Find audio files with specific suffix patterns"""
    if not directory.exists():
        return None, None
        
    # Search for both .wav and .flac files
    wav_files = list(directory.glob("*.wav"))
    flac_files = list(directory.glob("*.flac"))
    files = wav_files + flac_files
    
    full_file = None
    stem_file = None
    
    for f in files:
        name = f.stem.lower()  # Case-insensitive matching
        if name.endswith("_100-full"):
            full_file = f
        elif name.endswith("_100-stem"):
            stem_file = f
    
    return full_file, stem_file

def prepare_file(source_path, target_path, file_type):
    """Process and standardize audio file"""
    
    print(f"\n[Processing {file_type}]")
    print(f"  Source: {source_path.name}")
    
    # FLAC files need conversion to temp WAV first
    temp_file = None
    load_path = source_path
    
    if source_path.suffix.lower() == '.flac':
        ffmpeg_path = check_ffmpeg()
        if not ffmpeg_path:
            print(f"  ✗ ERROR: FLAC file detected but ffmpeg is not installed.")
            print("  Install ffmpeg to load FLAC files:")
            print("    macOS: brew install ffmpeg")
            print("    Linux: sudo apt-get install ffmpeg")
            return False
        
        # Create temp WAV file
        temp_file = source_path.parent / f".temp_{source_path.stem}.wav"
        print(f"  Converting FLAC to temporary WAV (using {ffmpeg_path})...")
        
        cmd = [ffmpeg_path, '-i', str(source_path), '-y', str(temp_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ✗ Conversion failed: {result.stderr}")
            return False
        
        load_path = temp_file
        print(f"  ✓ FLAC converted successfully")
    
    # Load audio
    try:
        # Load with target duration
        audio, sr = librosa.load(str(load_path), sr=None, mono=False, duration=TARGET_DURATION)
        print(f"  ✓ Loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        if temp_file and temp_file.exists():
            temp_file.unlink()
        return False
    
    # Check format
    if audio.ndim == 1:
        duration = len(audio) / sr
        channels = 1
    else:
        duration = len(audio[0]) / sr
        channels = audio.shape[0]
    
    print(f"  Current format:")
    print(f"    Sample rate: {sr} Hz")
    print(f"    Channels: {channels}")
    print(f"    Duration: {duration:.2f} seconds")
    
    # Prepare audio
    print(f"\n  Converting...")
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = librosa.to_mono(audio)
        print(f"    ✓ Converted to mono")
    
    # Resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        print(f"    ✓ Resampled to {TARGET_SR} Hz")
    
    # Ensure exact duration (trim or pad)
    target_samples = int(TARGET_DURATION * TARGET_SR)
    current_samples = len(audio)
    
    if current_samples > target_samples:
        audio = audio[:target_samples]
        print(f"    ✓ Trimmed to {TARGET_DURATION} seconds")
    elif current_samples < target_samples:
        audio = np.pad(audio, (0, target_samples - current_samples))
        print(f"    ✓ Padded to {TARGET_DURATION} seconds")
    else:
        print(f"    ✓ Duration is correct ({TARGET_DURATION} seconds)")
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        print(f"    ✓ Normalized")
    
    # Save
    try:
        sf.write(str(target_path), audio, TARGET_SR)
        print(f"\n  ✓ Saved to: {target_path}")
        print(f"    File size: {os.path.getsize(target_path)} bytes")
    except Exception as e:
        print(f"\n  ✗ Error saving file: {e}")
        if temp_file and temp_file.exists():
            temp_file.unlink()
        return False
    
    # Clean up temp file if it exists
    if temp_file and temp_file.exists():
        temp_file.unlink()
        print(f"  ✓ Cleaned up temporary file")
    
    return True

# ============================================
# MAIN EXECUTION
# ============================================

print(f"\nLooking for audio files in: {PROCESS_100_DIR}/\n")

# Find files
full_100, stem_100 = find_files(PROCESS_100_DIR)

if full_100:
    print(f"  ✓ Found full mix: {full_100.name}")
else:
    print(f"  ✗ No file ending with '_100-full.wav' or '_100-full.flac' found")

if stem_100:
    print(f"  ✓ Found stem: {stem_100.name}")
else:
    print(f"  ✗ No file ending with '_100-stem.wav' or '_100-stem.flac' found")

# Process if we have both files
if full_100 and stem_100:
    print("\n" + "="*70)
    print("PROCESSING FILES")
    print("="*70)
    
    # Process files
    vocal_ok = prepare_file(stem_100, RTG_100_DIR / "isolated_vocal.wav", "Isolated Vocal")
    mixture_ok = prepare_file(full_100, RTG_100_DIR / "stereo_mixture.wav", "Full Mixture")
    
    if vocal_ok and mixture_ok:
        print("\n" + "="*70)
        print("✓ SUCCESS - Files prepared!")
        print("="*70)
        print("\nCreated standardized files:")
        print(f"  {RTG_100_DIR / 'isolated_vocal.wav'} (4.7s, mono, 22050Hz)")
        print(f"  {RTG_100_DIR / 'stereo_mixture.wav'} (4.7s, mono, 22050Hz)")
        print("\nReady for visualization! Run: python main.py")
    else:
        print("\n" + "="*70)
        print("✗ PROCESSING FAILED")
        print("="*70)
        sys.exit(1)
else:
    print("\n" + "="*70)
    print("✗ FILES NOT FOUND")
    print("="*70)
    print(f"\nPlease add your audio files to: {PROCESS_100_DIR}/")
    print("\nRequired naming pattern:")
    print("  - Any file ending with '_100-full.wav' or '_100-full.flac'")
    print("  - Any file ending with '_100-stem.wav' or '_100-stem.flac'")
    print("\nExample filenames:")
    print("  - intergalactic_100-full.flac (full mix with instruments + vocals)")
    print("  - intergalactic_100-stem.flac (isolated vocals)")
    sys.exit(1)
