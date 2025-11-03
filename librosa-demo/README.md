# AI Stem Separation Demo

This project demonstrates a complete three-phase workflow for vocal separation, from visualization through spectral fingerprinting-based separation to simulated U-Net architecture.

## Phases

1.  **Phase 1: Audio Input & Visualization (`main.py`)**
    *   Loads paired audio files (full mixture + isolated vocal)
    *   Standardizes format: mono, 22050Hz, 4.7s duration
    *   Generates waveform and spectrogram visualizations for comparison
    *   Prepares audio for Phase 2 processing

2.  **Phase 2: Vocal Separation via Spectral Fingerprinting (`audio_processing.py`)**
    *   Creates 18 multi-scale "slices" of each spectrogram using convolutional filters
    *   Compresses each slice through encoder layers to bottleneck representation
    *   Extracts 425 metrics per time window (400-point frequency profile + 25 derived features)
    *   Optimizes EQ curves to match mixture fingerprint to vocal fingerprint
    *   Reconstructs separated vocal audio (achieves 60-80% quality proof-of-concept)
    *   **Goal:** Prove that spectral fingerprinting contains sufficient information for separation

3.  **Phase 3: Simulated 4-Stem U-Net Architecture (`phase3_demo.py`)**
    *   Demonstrates the planned production U-Net model architecture
    *   Simulates 4-stem separation: Vocals, Drums, Bass, Other
    *   Visualizes encoder → bottleneck → decoder data flow with skip connections
    *   Shows expected output quality from a fully trained model
    *   **Goal:** Illustrate the next step after fingerprinting validation

---

## Setup

### Cross-Platform Compatibility

✅ **All Python scripts are fully cross-platform compatible** (Windows, macOS, Linux)
- Uses `pathlib.Path` for all file paths
- Uses `subprocess` for cross-platform command execution
- No hard-coded path separators or platform-specific code

### Installation

1.  **Install `uv`** (optional, but recommended):

    ```bash
    pip install uv
    ```

2.  **Install Python Dependencies:**

    ```bash
    # Using uv (recommended)
    uv pip install -r requirements.txt

    # OR using regular pip
    pip install -r requirements.txt
    ```

3.  **Install System Tools:**

    **Required:**
    - `ffmpeg` - For processing `.flac` files

    **Optional:**
    - `feh` - For automatic image display (Linux/macOS only, not available on Windows)

    **Platform-specific installation:**

    *   **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg feh
        ```

    *   **Linux (using apt-get):**
        ```bash
        sudo apt-get update && sudo apt-get install -y ffmpeg feh
        ```

    *   **Windows:**
        ```powershell
        # Install ffmpeg via Chocolatey
        choco install ffmpeg

        # OR download manually from: https://ffmpeg.org/download.html
        # Note: feh is not available on Windows - images will be saved but not auto-displayed
        ```

---

## Workflow

**Step 1: Place Audio Files**

Before running the scripts, place your audio files in the `pre/process/100-window/` directory. The scripts look for files with the following naming convention:

*   `*_100-full.flac` or `*_100-full.wav` (the full mixture)
*   `*_100-stem.flac` or `*_100-stem.wav` (the isolated vocal)

**Step 2: Phase 1 - Visualization (`main.py`)**

Run `main.py` to standardize the audio and visualize the input files. This is a useful first step to ensure your audio is being loaded correctly.

```bash
# From the librosa-demo directory
python main.py --interactive
```

This will:

*   Create standardized `.wav` files in `pre/rtg/100-window/`.
*   Generate four images in the `figures/` directory for you to inspect.

**Step 3: Phase 2 - Vocal Separation (`audio_processing.py`)**

After preparing the audio in Phase 1, run `audio_processing.py` to perform the vocal separation.

```bash
# From the librosa-demo directory
python audio_processing.py
```

This script will:

*   Load the standardized audio from `pre/rtg/100-window/`.
*   Apply 18-slice spectral fingerprinting to both mixture and vocal.
*   Optimize EQ curves through gradient descent (100 iterations).
*   Reconstruct the separated vocal track.
*   Save results to the `output/` directory:
    *   `extracted_vocal.wav` - The separated vocal
    *   `1_original_mixture.wav` - Reference mixture
    *   `2_target_vocal.wav` - Reference isolated vocal
    *   `optimization_loss.png` - Training loss curve
    *   `spectrograms.png` - Visual comparison

**Step 4: Phase 3 - U-Net Architecture Demo (`phase3_demo.py`)**

Run the simulated 4-stem U-Net architecture demonstration.

```bash
# From the librosa-demo directory
python phase3_demo.py
```

This script will:

*   Use the output from Phase 2 as input.
*   Simulate 4-stem separation (Vocals, Drums, Bass, Other) using signal processing.
*   Visualize the U-Net architecture data flow.
*   Save results to the `output_phase3/` directory:
    *   `stems/vocals.wav`, `stems/drums.wav`, `stems/bass.wav`, `stems/other.wav`
    *   `4_stem_separation_spectrograms.png` - Visual comparison of all stems

---

## Notes

*   **Output Files:**
    *   Phase 2: `output/extracted_vocal.wav` - The separated vocal track
    *   Phase 3: `output_phase3/stems/` - Four separated stems (vocals, drums, bass, other)
*   **Image Viewing:**
    *   On **Linux/macOS** with `feh` installed: Images open automatically
    *   On **Windows** or without `feh`: Images save to `figures/` and `output/` directories for manual viewing
*   **Audio File Format:** Accepts both `.wav` and `.flac` files (requires ffmpeg for FLAC support)

---

## Non-Interactive Workflow

To run the entire workflow from start to finish without any interactive prompts, you can run all three scripts in sequence. This is ideal for automated processing.

```bash
# From the librosa-demo directory

# Run all three phases sequentially
python main.py && python audio_processing.py && python phase3_demo.py
```

**Individual phases:**

```bash
# Phase 1: Visualization and preparation
python main.py

# Phase 2: Vocal separation via spectral fingerprinting
python audio_processing.py

# Phase 3: Simulated U-Net 4-stem architecture
python phase3_demo.py
```
