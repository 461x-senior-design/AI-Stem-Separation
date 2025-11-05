"""
Audio File Preparation Script
==============================
This script will:
1. Find your audio files in the process/ directories
2. Convert them to the right format
3. Move them to rtg/ (ready-to-go) directories

Place your files in:
- process/100-window/ for short version (files named *_100-full.wav and *_100-stem.wav)
- process/no-limit/ for full-length version (files named *_nl-full.wav and *_nl-stem.wav)
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

print("="*70)
print("AUDIO FILE PREPARATION")
print("="*70)

# Directories
BASE_DIR = Path(__file__).parent
PROCESS_100_DIR = BASE_DIR / "process" / "100-window"
PROCESS_NL_DIR = BASE_DIR / "process" / "no-limit"
RTG_100_DIR = BASE_DIR / "rtg" / "100-window"
RTG_NL_DIR = BASE_DIR / "rtg" / "no-limit"

# Create rtg directories if they don't exist
RTG_100_DIR.mkdir(parents=True, exist_ok=True)
RTG_NL_DIR.mkdir(parents=True, exist_ok=True)

# Config
TARGET_SR = 22050  # Sample rate

def find_files(directory, suffix_full, suffix_stem):
    """Find audio files with specific suffixes"""
    files = list(directory.glob("*.wav"))
    
    full_file = None
    stem_file = None
    
    for f in files:
        if f.name.endswith(suffix_full):
            full_file = f
        elif f.name.endswith(suffix_stem):
            stem_file = f
    
    return full_file, stem_file

def prepare_file(source_path, target_path, target_duration, file_type):
    """Check audio file and prepare it"""
    
    print(f"\n[Processing {file_type}]")
    print(f"  Source: {source_path.name}")
    
    # Load audio
    try:
        audio, sr = librosa.load(str(source_path), sr=None, mono=False, duration=target_duration)
        print(f"  ✓ Loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return False
    
    # Check format
    duration = len(audio.flatten()) / sr if audio.ndim == 1 else len(audio[0]) / sr
    channels = 1 if audio.ndim == 1 else audio.shape[0]
    
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
    
    # Trim or pad to target duration if specified
    if target_duration is not None:
        target_samples = int(target_duration * TARGET_SR)
        current_samples = len(audio)
        
        if current_samples > target_samples:
            audio = audio[:target_samples]
            print(f"    ✓ Trimmed to {target_duration} seconds")
        elif current_samples < target_samples:
            audio = np.pad(audio, (0, target_samples - current_samples))
            print(f"    ✓ Padded to {target_duration} seconds")
        else:
            print(f"    ✓ Duration is correct ({target_duration} seconds)")
    else:
        print(f"    ✓ Keeping full duration ({len(audio) / TARGET_SR:.2f} seconds)")
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    print(f"    ✓ Normalized")
    
    # Save
    sf.write(str(target_path), audio, TARGET_SR)
    print(f"\n  ✓ Saved to: {target_path}")
    
    return True

# ============================================
# MAIN EXECUTION
# ============================================

print("\nLooking for audio files...\n")

# Check 100-window directory
print("Checking process/100-window/")
full_100, stem_100 = find_files(PROCESS_100_DIR, "_100-full.wav", "_100-stem.wav")

if full_100:
    print(f"  Found full mix: {full_100.name}")
if stem_100:
    print(f"  Found stem: {stem_100.name}")

# Check no-limit directory
print("\nChecking process/no-limit/")
full_nl, stem_nl = find_files(PROCESS_NL_DIR, "_nl-full.wav", "_nl-stem.wav")

if full_nl:
    print(f"  Found full mix: {full_nl.name}")
if stem_nl:
    print(f"  Found stem: {stem_nl.name}")

# Determine which to process
if full_100 and stem_100:
    print("\n" + "="*70)
    print("PROCESSING: 100-WINDOW VERSION (4.7 seconds)")
    print("="*70)
    
    # Process with 4.7 second duration
    vocal_ok = prepare_file(stem_100, RTG_100_DIR / "isolated_vocal.wav", 4.7, "Isolated Vocal (100-window)")
    mixture_ok = prepare_file(full_100, RTG_100_DIR / "stereo_mixture.wav", 4.7, "Full Mixture (100-window)")
    
    if vocal_ok and mixture_ok:
        print("\n" + "="*70)
        print("✓ SUCCESS - 100-window files ready!")
        print("="*70)
        print("\nPrepared files in rtg/100-window/:")
        print(f"  {RTG_100_DIR / 'isolated_vocal.wav'}")
        print(f"  {RTG_100_DIR / 'stereo_mixture.wav'}")
        print("\nRun:")
        print("  python sanity_check_complete.py")

elif full_nl and stem_nl:
    print("\n" + "="*70)
    print("PROCESSING: NO-LIMIT VERSION (full song)")
    print("="*70)
    
    # Process without duration limit
    vocal_ok = prepare_file(stem_nl, RTG_NL_DIR / "isolated_vocal.wav", None, "Isolated Vocal (no-limit)")
    mixture_ok = prepare_file(full_nl, RTG_NL_DIR / "stereo_mixture.wav", None, "Full Mixture (no-limit)")
    
    if vocal_ok and mixture_ok:
        print("\n" + "="*70)
        print("✓ SUCCESS - No-limit files ready!")
        print("="*70)
        print("\nPrepared files in rtg/no-limit/:")
        print(f"  {RTG_NL_DIR / 'isolated_vocal.wav'}")
        print(f"  {RTG_NL_DIR / 'stereo_mixture.wav'}")
        print("\nRun:")
        print("  python sanity_check_full_length.py")

else:
    print("\n" + "="*70)
    print("✗ NO FILES FOUND")
    print("="*70)
    print("\nPlease add your audio files to one of these directories:")
    print("\nFor 100-window version (4.7 seconds):")
    print(f"  {PROCESS_100_DIR}/")
    print("  Name your files:")
    print("    yourfile_100-full.wav   (full mixture)")
    print("    yourfile_100-stem.wav   (isolated vocal)")
    print("\nFor no-limit version (full song):")
    print(f"  {PROCESS_NL_DIR}/")
    print("  Name your files:")
    print("    yourfile_nl-full.wav    (full mixture)")
    print("    yourfile_nl-stem.wav    (isolated vocal)")
    print("\nExamples:")
    print("  song_100-full.wav, song_100-stem.wav")
    print("  track_nl-full.wav, track_nl-stem.wav")
    sys.exit(1)
