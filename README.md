# AI-Stem-Separation

AI Stem Separation tool built using PyTorch and U-Net for our Senior Design Project.

## Project Structure

This repository contains two demonstration workflows:

### 1. Progress Report #1: Basic Audio Visualization (`progress-report-1/`)

Minimal demo showing foundational audio processing capabilities:
- Loads `.wav` or `.flac` audio files
- Generates waveform and spectrogram visualizations
- Uses `feh` for image display

**Quick Start:**
```bash
cd progress-report-1
python main.py <audio_file> [--sr 22050] [--duration 0.10] [--start 0.0]
```

See [`progress-report-1/README.md`](progress-report-1/README.md) for details.

---

### 2. Complete 3-Phase Vocal Separation Demo (`librosa-demo/`)

Full workflow demonstrating spectral fingerprinting-based vocal separation across three phases:

#### **Phase 1: Audio Input & Visualization** (`main.py`)
- Loads paired audio (full mixture + isolated vocal)
- Standardizes format (mono, 22050Hz, 4.7s duration)
- Creates comparison visualizations (waveforms and spectrograms)

#### **Phase 2: Vocal Separation via Spectral Fingerprinting** (`audio_processing.py`)
- Applies 18-slice multi-scale spectral fingerprinting
- Compresses spectrograms through encoder layers to bottleneck
- Extracts 425 metrics per time window (400-point frequency profile + 25 derived features)
- Optimizes EQ curves to match mixture fingerprint to vocal fingerprint
- Reconstructs separated vocal audio (60-80% quality proof-of-concept)

#### **Phase 3: Simulated 4-Stem U-Net Architecture** (`phase3_demo.py`)
- Demonstrates planned U-Net architecture for production system
- Simulates 4-stem separation (Vocals, Drums, Bass, Other)
- Visualizes encoder → bottleneck → decoder data flow
- Shows expected output quality from trained model

**Quick Start:**
```bash
cd librosa-demo

# Install dependencies
uv pip install -r requirements.txt
brew install ffmpeg feh  # macOS

# Place audio files in pre/process/100-window/:
#   - *_100-full.wav/.flac (mixture)
#   - *_100-stem.wav/.flac (isolated vocal)

# Run complete workflow
python main.py && python audio_processing.py && python phase3_demo.py
```

See [`librosa-demo/README.md`](librosa-demo/README.md) for detailed documentation.

---

## Dependencies

### Python Packages

```bash
# Using uv (recommended)
uv pip install librosa matplotlib soundfile numpy scipy

# OR using pip
pip install librosa matplotlib soundfile numpy scipy
```

### System Tools

**Required:**
- `ffmpeg` - FLAC file processing (all platforms)

**Optional:**
- `feh` - Image viewer (Linux/macOS only, not available on Windows)

**Platform-specific installation:**

**macOS:**
```bash
brew install ffmpeg feh
```

**Linux:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg feh
```

**Windows:**
```powershell
# Install ffmpeg via Chocolatey
choco install ffmpeg

# OR download from: https://ffmpeg.org/download.html
# Note: Images will be saved but not auto-displayed on Windows
```

### Cross-Platform Compatibility

✅ All Python scripts use `pathlib.Path` for cross-platform file paths
✅ Works on Windows, macOS, and Linux
✅ Optional image auto-display with `feh` (Linux/macOS) or manual viewing (Windows)

---

## Key Concepts

**Spectral Fingerprinting:** Each time window (~46ms) is analyzed through 18 different convolutional filters applied to the spectrogram, producing 425 metrics per window. This creates a unique "fingerprint" distinguishing vocal from instrumental content.

**Optimization Approach:** Rather than immediately training a neural network, Phase 2 proves the concept by optimizing EQ curves to transform the mixture's fingerprint to match the vocal's fingerprint. This validates that the fingerprint contains sufficient information for separation.

**Next Steps:** Train a U-Net model on the validated fingerprinting approach to achieve 95%+ separation quality at 1800× faster inference speed.
