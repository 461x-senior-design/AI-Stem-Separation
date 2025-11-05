#!/usr/bin/env python3
"""
Clean up demo directory to pre-demo state.

This script removes all generated files from running the demo,
allowing you to rerun the demo from a clean state.

Usage:
    python clean.py
"""

import os
import shutil
from pathlib import Path

def clean_demo_directory():
    """Clean up all generated files while preserving original demo files."""

    print("Cleaning librosa-demo directory...")
    print("Removes: figures/, output/, pre/rtg/ files")
    print("Preserves: pre/process/ files (user audio inputs)")
    print("="*50)

    # Files to keep (original demo files)
    keep_files = {
        "main.py",
        "audio_utils.py",
        "phase2_demo.py",
        "audio_output_utils.py",
        "README.md",
        "clean.py",
        "Intergalactic_Full_100-full.wav",
        "Intergalactic_Acapella_100-stem.wav"
    }

    # Directories to clean
    clean_dirs = ["figures", "output", "pre/rtg"]

    # Clean specific directories
    for dir_name in clean_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Removing directory: {dir_name}/")
            shutil.rmtree(dir_path)

    # Clean loose files in root directory (old mixture.wav, acapella.wav, etc.)
    current_dir = Path(".")
    for item in current_dir.iterdir():
        if item.is_file() and item.name not in keep_files:
            if item.name.endswith('.wav') or item.name.endswith('.png'):
                print(f"Removing file: {item.name}")
                item.unlink()

    # Recreate necessary directories
    Path("figures").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    Path("pre/rtg/100-window").mkdir(parents=True, exist_ok=True)

    print("\nDirectory cleaned.")
    print("\nKept files:")
    for file in sorted(keep_files):
        if Path(file).exists():
            print(f"  {file}")

    print("\nCleaned locations:")
    print("  figures/ (visualization outputs)")
    print("  output/ (separated audio)")
    print("  pre/rtg/100-window/ (processed files)")

    print("\nPreserved:")
    print("  pre/process/100-window/ (user audio inputs)")

    print("\nRecreated directories:")
    print("  figures/")
    print("  output/")
    print("  pre/rtg/100-window/")

    print("\nReady to rerun demo.")
    print("Run: python main.py --interactive")

if __name__ == "__main__":
    clean_demo_directory()
