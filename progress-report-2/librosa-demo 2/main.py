#!/usr/bin/env python3
"""
Phase 1: Audio Input & Visualization Demo

Loads prepared audio files and creates interactive visualizations
showing time-domain waveforms and frequency-domain spectrograms.
"""

import argparse
import sys
import os
import time
from pathlib import Path
import subprocess

from audio_utils import load_audio, plot_waveform, show_spectrogram

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Audio Input & Visualization"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate (Hz). Default: 22050"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Waveform duration to show (seconds). Default: 1.0"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.5,
        help="Start time for waveform (seconds). Default: 0.5"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause between steps for explanation"
    )
    args = parser.parse_args()

    print("\nPHASE 1: AUDIO INPUT & VISUALIZATION")
    time.sleep(0.2)
    print("="*50)
    time.sleep(0.2)
    print("Goal: Load audio pairs & visualize the separation challenge")
    time.sleep(0.2)
    print("We'll compare: MIXTURE vs ACAPELLA")
    time.sleep(0.2)
    print("="*50)
    time.sleep(0.3)

    # Check for prepared files in rtg/100-window/
    mixture_file = Path("pre/rtg/100-window/stereo_mixture.wav")
    acapella_file = Path("pre/rtg/100-window/isolated_vocal.wav")

    if not mixture_file.exists() or not acapella_file.exists():
        print("\nAUDIO PREPARATION PHASE")
        time.sleep(0.2)
        print("-"*30)
        time.sleep(0.2)

        # Check for files with _100-full and _100-stem pattern in process directory
        process_dir = Path("pre/process/100-window")
        process_dir.mkdir(parents=True, exist_ok=True)

        # Search for both .wav and .flac files
        wav_files = list(process_dir.glob("*.wav"))
        flac_files = list(process_dir.glob("*.flac"))
        all_files = wav_files + flac_files

        intergalactic_full = None
        intergalactic_stem = None

        for f in all_files:
            if f.name.endswith("_100-full.wav") or f.name.endswith("_100-full.flac"):
                intergalactic_full = f
            elif f.name.endswith("_100-stem.wav") or f.name.endswith("_100-stem.flac"):
                intergalactic_stem = f

        print("Checking for demo audio files in pre/process/100-window/...")
        time.sleep(0.2)
        if intergalactic_full:
            print(f"  [✓] {intergalactic_full.name}")
        else:
            print(f"  [✗] *_100-full.wav/.flac")
        time.sleep(0.15)
        if intergalactic_stem:
            print(f"  [✓] {intergalactic_stem.name}")
        else:
            print(f"  [✗] *_100-stem.wav/.flac")
        time.sleep(0.2)

        if not intergalactic_full or not intergalactic_stem:
            print("\nMissing files! Please add files to pre/process/100-window/:")
            print("  - *_100-full.wav or *_100-full.flac (full mixture)")
            print("  - *_100-stem.wav or *_100-stem.flac (isolated vocal)")
            sys.exit(1)

        print("\nFiles found. Ready to process.")

        if args.interactive:
            input("\nPress Enter to run audio standardization...")

        # Run prepare_audio_files.py
        print("\nRunning audio standardization")
        time.sleep(0.2)
        print("\033[92m  $ python prepare_audio_files.py\033[0m")
        if args.interactive:
            input("Press Enter to continue...")
        time.sleep(0.2)
        print("  Converting to mono, resampling, normalizing...")
        time.sleep(0.2)

        # Run the prepare script
        result = subprocess.run([
            sys.executable,
            "prepare_audio_files.py"
        ], capture_output=True, text=True, cwd=".")

        # Show actual output from the script
        if result.stdout:
            print("\nPrepare script output:")
            print(result.stdout)
        
        if result.returncode != 0:
            print("\nAudio preparation failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            sys.exit(1)

        print("\nPreparation phase complete.")
        time.sleep(0.3)
        if args.interactive:
            input("\nPress Enter to start visualization phase...")

    # Verify files exist now
    if not mixture_file.exists() or not acapella_file.exists():
        print("ERROR: Prepared files not found")
        print(f"  Expected: {mixture_file}")
        print(f"  Expected: {acapella_file}")
        print("\nThe preparation script may have failed. Check the output above.")
        print("If using FLAC files, make sure ffmpeg is installed:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        sys.exit(1)

    # Step 1: Load Both Audio Files
    print("\nLOADING AUDIO FILES")
    time.sleep(0.2)
    print("-"*20)
    time.sleep(0.2)

    print("Loading prepared audio pair for analysis...")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to load mixture track...")

    print("  Loading stereo_mixture.wav (instruments + vocals)")
    time.sleep(0.15)
    print(f"\033[92m  $ audio, sr = load_audio('{mixture_file}', sr={args.sr}, mono=True)\033[0m")
    if args.interactive:
        input("Press Enter to continue...")
    time.sleep(0.2)
    mixture_audio, sr = load_audio(str(mixture_file), args.sr, True)
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to load acapella track...")

    print("  Loading isolated_vocal.wav (vocals only)")
    time.sleep(0.15)
    print(f"\033[92m  $ acapella_audio, sr = load_audio('{acapella_file}', sr={args.sr}, mono=True)\033[0m")
    if args.interactive:
        input("Press Enter to continue...")
    time.sleep(0.2)
    acapella_audio, _ = load_audio(str(acapella_file), args.sr, True)
    time.sleep(0.2)

    print("  Both files loaded successfully.")
    time.sleep(0.15)
    print("  Standardized: mono, 22050 Hz, 4.71s duration")
    time.sleep(0.3)

    if args.interactive:
        input("\nPress Enter to begin visualization...")

    # Step 2: Plot Waveforms for Both Files
    print("\nWAVEFORM VISUALIZATION")
    time.sleep(0.2)
    print("-"*25)
    time.sleep(0.2)

    print("Creating time-domain visualizations...")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter for waveform explanation...")

    print("  Shows: amplitude vs time")
    time.sleep(0.15)
    print(f"  Duration: {args.duration}s segment (starting at {args.start}s)")
    time.sleep(0.15)
    print("  MIXTURE: instruments + vocals (complex waveform)")
    time.sleep(0.15)
    print("  ACAPELLA: vocals only (focused waveform)")
    time.sleep(0.15)
    print("  Goal: Compare the two audio sources.")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to generate MIXTURE waveform...")

    print("\n  Processing mixture track...")
    time.sleep(0.15)
    print(f"\033[92m  $ plot_waveform(audio, sr={sr}, duration={args.duration}, start={args.start})\033[0m")
    time.sleep(0.2)
    plot_waveform(mixture_audio, sr, args.duration, args.start, 10, 3, -1.1, 1.1,
                  "figures/waveform.png", "")
    print("  Mixture waveform created.")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to generate ACAPELLA waveform...")

    print("  Processing acapella track...")
    time.sleep(0.15)
    print(f"\033[92m  $ plot_waveform(acapella_audio, sr={sr}, duration={args.duration}, start={args.start})\033[0m")
    time.sleep(0.2)
    plot_waveform(acapella_audio, sr, args.duration, args.start, 10, 3, -1.1, 1.1,
                  "figures/acapella_waveform.png", " (Acapella)")
    print("  Acapella waveform created.")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to continue to spectrograms...")

    # Step 3: Plot Spectrograms for Both Files
    print("\nSPECTROGRAM VISUALIZATION")
    time.sleep(0.2)
    print("-"*27)
    time.sleep(0.2)

    print("Creating frequency-domain visualizations...")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter for spectrogram explanation...")

    print("  Method: Short-Time Fourier Transform (STFT)")
    time.sleep(0.15)
    print("  Shows: frequency content over time")
    time.sleep(0.15)
    print("  Scale: brightness = loudness, vertical = frequency")
    time.sleep(0.15)
    print("  MIXTURE: shows all instruments + vocals")
    time.sleep(0.15)
    print("  ACAPELLA: shows vocal frequencies only")
    time.sleep(0.15)
    print("  Reveals what we need to separate.")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to create MIXTURE spectrogram...")

    print("\n  Processing mixture track...")
    time.sleep(0.15)
    print(f"\033[92m  $ show_spectrogram(audio, sr={sr}, n_fft=2048, hop_length=512)\033[0m")
    time.sleep(0.2)
    show_spectrogram(mixture_audio, sr, 2048, 512, "figures/mixture_spectrogram.png", " (Mixture)")
    print("  Mixture spectrogram created.")
    time.sleep(0.2)

    if args.interactive:
        input("\nPress Enter to create ACAPELLA spectrogram...")

    print("  Processing acapella track...")
    time.sleep(0.15)
    print(f"\033[92m  $ show_spectrogram(acapella_audio, sr={sr}, n_fft=2048, hop_length=512)\033[0m")
    time.sleep(0.2)
    show_spectrogram(acapella_audio, sr, 2048, 512, "figures/acapella_spectrogram.png", " (Acapella)")
    print("  Acapella spectrogram created.")
    time.sleep(0.3)

    # Display images
    if args.interactive:
        input("\nPress Enter to display all visualizations...")

    print("\nDisplaying all visualizations...")
    time.sleep(0.2)
    print("  Opening 4 images: waveforms and spectrograms for both audio files")
    time.sleep(0.2)
    try:
        subprocess.run(["feh", "--title", "Mixture Waveform", "figures/waveform.png"], check=False)
        subprocess.run(["feh", "--title", "Acapella Waveform", "figures/acapella_waveform.png"], check=False)
        subprocess.run(["feh", "--title", "Mixture Spectrogram", "figures/mixture_spectrogram.png"], check=False)
        subprocess.run(["feh", "--title", "Acapella Spectrogram", "figures/acapella_spectrogram.png"], check=False)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Note: feh image viewer not available.")
        time.sleep(0.15)
        print("View images manually in figures/ directory.")
        time.sleep(0.2)

    print("\n" + "="*50)
    time.sleep(0.2)
    print("PHASE 1 COMPLETE")
    time.sleep(0.2)
    print("="*50)
    time.sleep(0.2)
    print("\nThis shows the input side - getting audio pairs into Python.")
    time.sleep(0.2)
    print("Compare mixture vs acapella to see the separation challenge.")
    time.sleep(0.2)
    print("The spectrograms are what we'll use in Phase 2 for separation.")
    time.sleep(0.2)

if __name__ == "__main__":
    main()