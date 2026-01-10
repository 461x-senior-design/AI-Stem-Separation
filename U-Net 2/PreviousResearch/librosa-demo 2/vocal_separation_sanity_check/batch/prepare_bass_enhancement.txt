"""
Bass Enhancement Audio Preparation
===================================
Converts your audio files to the right format for low-end enhancement.

Place your files (any format: mp3, wav, flac, etc) in this directory:
- reference_track.* (song with GOOD bass)
- target_track.* (your track to enhance)

This script will convert them to proper WAV format at 44.1kHz.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

print("="*70)
print("BASS ENHANCEMENT - AUDIO PREPARATION")
print("="*70)

# Config
TARGET_DIR = Path(__file__).parent
TARGET_SR = 44100  # Standard 44.1kHz

def find_audio_file(base_name):
    """Find audio file with any extension"""
    extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    
    for ext in extensions:
        path = TARGET_DIR / f"{base_name}{ext}"
        if path.exists():
            return path
    
    return None

def load_audio_info(source_path):
    """Load audio and get its info"""
    
    print(f"\n[Loading {source_path.name}]")
    
    try:
        audio, sr = librosa.load(str(source_path), sr=None, mono=False)
        print(f"  ✓ Loaded successfully")
        
        duration = len(audio.flatten()) / sr if audio.ndim == 1 else len(audio[0]) / sr
        channels = 1 if audio.ndim == 1 else audio.shape[0]
        
        print(f"    Sample rate: {sr} Hz")
        print(f"    Channels: {channels}")
        print(f"    Duration: {duration:.2f} seconds")
        
        return audio, sr, True
        
    except Exception as e:
        print(f"  ✗ Error loading: {e}")
        return None, None, False

def prepare_bass_file(audio, original_sr, target_name):
    """Prepare audio file for bass enhancement"""
    
    print(f"\n[Processing {target_name}]")
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = librosa.to_mono(audio)
        print(f"  ✓ Converted to mono")
    
    # Resample to 44.1kHz
    if original_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=TARGET_SR)
        print(f"  ✓ Resampled from {original_sr} Hz to {TARGET_SR} Hz")
    else:
        print(f"  ✓ Already at {TARGET_SR} Hz")
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    print(f"  ✓ Normalized")
    
    # Save
    target_path = TARGET_DIR / f"{target_name}.wav"
    sf.write(str(target_path), audio, TARGET_SR)
    print(f"  ✓ Saved to: {target_path.name}")
    
    return True

# ============================================
# MAIN
# ============================================

print("\nLooking for audio files...")

# Find files
ref_source = find_audio_file("reference_track")
if ref_source:
    print(f"  Found reference: {ref_source.name}")
else:
    print(f"  ✗ No reference track found")

target_source = find_audio_file("target_track")
if target_source:
    print(f"  Found target: {target_source.name}")
else:
    print(f"  ✗ No target track found")

if not ref_source or not target_source:
    print("\n" + "="*70)
    print("✗ NEED BOTH FILES")
    print("="*70)
    print("\nPlease add BOTH audio files to this directory:")
    print(f"  {TARGET_DIR}/")
    print("\nName them:")
    print("  reference_track.* (song with GOOD bass - any format)")
    print("  target_track.* (your track to enhance - any format)")
    print("\nExamples:")
    print("  reference_track.mp3")
    print("  target_track.wav")
    sys.exit(1)

# Load both files
print("\n" + "="*70)
print("LOADING FILES")
print("="*70)

ref_audio, ref_sr, ref_ok = load_audio_info(ref_source)
target_audio, target_sr, target_ok = load_audio_info(target_source)

if not (ref_ok and target_ok):
    print("\n✗ Failed to load files")
    sys.exit(1)

# Process both files to 44.1kHz
print("\n" + "="*70)
print("PROCESSING FILES")
print("="*70)
print(f"\nConverting both files to {TARGET_SR} Hz (44.1kHz)")

ref_success = prepare_bass_file(ref_audio, ref_sr, "reference_track")
target_success = prepare_bass_file(target_audio, target_sr, "target_track")

# Summary
print("\n" + "="*70)
if ref_success and target_success:
    print("✓ SUCCESS - Files ready for bass enhancement!")
    print("="*70)
    print("\nPrepared files:")
    print(f"  reference_track.wav")
    print(f"  target_track.wav")
    print("\nBoth files are now:")
    print(f"  - Mono")
    print(f"  - {TARGET_SR} Hz (44.1kHz)")
    print(f"  - Normalized")
    print("\nRun:")
    print("  python3 enhance_low_end.py")
else:
    print("✗ FAILED - Please fix issues above")
    print("="*70)
