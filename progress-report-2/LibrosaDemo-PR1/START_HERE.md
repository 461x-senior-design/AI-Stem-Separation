# 🎯 START HERE - Team 2 Guide

## What You're About to See

This demo showcases **Conv2D operations** - one of the core building blocks of the U-Net architecture. But instead of applying convolutions over time (like traditional audio processing), we're applying them **over frequency domains** using LibROSA's spectral transformations.

**Why this matters:** This is the foundation that makes encoder-decoder architectures work. Once you understand how Conv2D can extract patterns from spectrograms, the rest of the U-Net is really just creative choices about:
- How many layers to stack
- What filter sizes to use
- How to connect encoder to decoder

The hard part? Understanding **how spectral features encode musical patterns**. Once you get that, the rest flows naturally.

---

## 🔧 Setup (One-Time)

### Install UV (if you don't have it)

```bash
pip install uv
```

**Why UV?** It just works. No virtual environments to activate, no containers to remember to enter. Just run `uv pip install` anywhere and you're good to go.

### Install Dependencies

```bash
# From the librosa-demo 2 directory
uv pip install -r requirements.txt
```

### Install System Tools

**Linux:**
```bash
sudo apt-get update && sudo apt-get install -y feh
# sudo apt-get install -y ffmpeg  # Uncomment if you want FLAC support
```

**macOS:**
```bash
# brew install feh  # Linux image viewer - not needed on macOS
# brew install ffmpeg  # Uncomment if you want FLAC support
```

**Note for macOS users:** `feh` is a Linux image viewer. On macOS, just replace `feh` with `open` in the code wherever you see it (the `open` command is built-in).

**Windows:**
```bash
# Note: feh not available on Windows - images save to disk instead
# choco install ffmpeg  # Uncomment if you want FLAC support
```

**Note:** FFmpeg is only needed if you want to load `.flac` files. If you're using `.wav` files, you can skip it.

---

## 🚀 Running the Demo

### Step 1: Prepare Your Audio Files

Place paired audio in `pre/process/100-window/`:
- `yourfile_100-full.wav` or `.flac` (full mixture with everything)
- `yourfile_100-stem.wav` or `.flac` (isolated vocal)

The files **must** be time-aligned (same start point, same duration).

### Step 2: Run Phase 1 - Visualization

```bash
python main.py
```

**What this does:**
- Loads and standardizes your audio (mono, 22050Hz, 4.7s)
- Generates waveform and spectrogram visualizations
- Saves processed audio to `pre/rtg/100-window/`

**Output:** Visual comparison showing what the audio looks like before processing.

### Step 3: Run Phase 2 - Conv2D Spectral Fingerprinting

```bash
python audio_processing.py
```

**What this does:**
- Applies **18 different Conv2D filters** to spectrograms
- Each filter detects different patterns:
  - Horizontal lines (sustained notes)
  - Vertical lines (transients/attacks)
  - Diagonal patterns (pitch sweeps)
  - Harmonic stacks (vocal formants)
  - Edge detection (timbre boundaries)
- Compresses each filtered view through encoder layers
- Extracts 425 metrics per time window
- Optimizes to separate vocal from mixture

**Output:** `output/extracted_vocal.wav` + visualizations

### Step 4: Run Phase 3 - Simulated U-Net Architecture

```bash
python phase3_demo.py
```

**What this does:**
- Demonstrates the **encoder → bottleneck → decoder** flow
- Simulates 4-stem separation (Vocals, Drums, Bass, Other)
- Shows how the full U-Net would process the separated vocal

**Output:** `output_phase3/stems/` containing all 4 separated stems

---

## 🧠 Key Concepts

### Conv2D Over Frequency (Not Time)

Traditional audio processing applies filters over **time** (e.g., reverb, delay).

We're applying Conv2D over **frequency bins** in spectrograms:
- Each pixel = energy at a specific frequency and time
- Conv2D filters detect **patterns across frequencies**
- Example: A 3×3 kernel can detect harmonic triads (fundamental + 3rd + 5th)

**This is what makes U-Net work for audio separation.**

### The 18 Slices

Think of these as 18 different "views" of the same spectrogram:

| Slice | Pattern Detected |
|-------|------------------|
| 0 | Raw magnitude (baseline) |
| 1-8 | Basic patterns (horizontal/vertical/diagonal/harmonics) |
| 9-15 | Oriented edge detectors (22.5°, 45°, 67.5°, 90°, etc.) |
| 16-18 | Laplacian and pooled versions |

Each slice captures something unique about the vocal. Combined, they create a "fingerprint" that only matches the actual vocal.

### Why This Sets Up Encoder-Decoder Work

**Phase 2 does manually what U-Net will learn automatically:**

1. **Encoder:** Compress spectrogram through Conv2D layers → extract features
2. **Bottleneck:** Represent entire audio in compact feature space
3. **Decoder:** Reconstruct vocal spectrogram from features

Right now, we're hand-crafting the filters (slice_0 through slice_17). The U-Net will **learn** what filters to use by training on thousands of examples.

**This is the hardest conceptual leap.** Once you see how Conv2D + spectral features = pattern detection, the rest is just architecture choices.

---

## 📊 Expected Results

### Phase 2 Quality
- **60-80% vocal separation** (proof-of-concept quality)
- You'll hear the vocal clearly, but with some artifacts
- Drums/bass should be much quieter
- Some phasing/clicking is normal

### Phase 3 Quality
- **Simulated 4-stem separation** (not trained, just signal processing)
- Shows what a trained U-Net would produce
- Each stem demonstrates the potential of the full pipeline

---

## 🎓 What You're Learning

1. **Conv2D isn't just for images** - it works on any 2D data (spectrograms are 2D)
2. **Spectral processing > time-domain processing** - frequency patterns encode musical structure
3. **Feature engineering → neural learning** - we hand-craft features, U-Net learns them
4. **Encoder-decoder intuition** - compress, extract, reconstruct

**After this demo, encoder-decoder architecture will make intuitive sense.**

The creative part comes later:
- How many encoder layers? (3? 4? 5?)
- What kernel sizes? (3×3? 5×5? 7×7?)
- How many filters per layer? (64? 128? 256?)
- Where to add skip connections?

But the **core concept** - using Conv2D to detect spectral patterns - that's what this demo teaches.

---

## 🐛 Troubleshooting

**"NO FILES FOUND"**
- Check file naming: Must end with `_100-full.wav` and `_100-stem.wav`
- Check directory: Files go in `pre/process/100-window/`

**"Takes forever"**
- Normal! Phase 2 processes ~765,000 data points
- Use the 4.7s snippet version (not full-length)
- Reduce iterations in config if needed

**"Results sound terrible"**
- Make sure files are time-aligned (use Audacity Time Shift Tool)
- Verify "full" = mixture, "stem" = isolated vocal (not swapped)
- Try different audio - some sources separate better than others

**Out of memory**
- Reduce `n_fft` from 2048 to 1024 in the config
- Close other applications
- Use shorter audio clips

---

## 🎯 Success Criteria

You've understood the demo when you can explain:

1. **Why we use spectrograms instead of raw waveforms**
   - Answer: Conv2D needs 2D data, spectrograms show frequency+time structure

2. **What Conv2D filters are detecting**
   - Answer: Musical patterns (harmonics, transients, formants, timbre)

3. **How this relates to encoder-decoder**
   - Answer: Encoder compresses spectrogram → extracts features, Decoder reconstructs target

4. **Why this is "the hard part"**
   - Answer: Understanding spectral feature encoding is conceptually difficult; architecture choices are creative but easier

---

## 📁 Output Files

After running all three phases:

```
librosa-demo 2/
├── figures/
│   ├── waveform.png              # Phase 1 visualization
│   └── spectrogram.png           # Phase 1 visualization
├── output/
│   ├── extracted_vocal.wav       # Phase 2 result
│   ├── optimization_loss.png     # Phase 2 learning curve
│   └── spectrograms.png          # Phase 2 comparison
└── output_phase3/
    ├── stems/
    │   ├── vocals.wav            # Phase 3 separated vocal
    │   ├── drums.wav             # Phase 3 separated drums
    │   ├── bass.wav              # Phase 3 separated bass
    │   └── other.wav             # Phase 3 separated other
    └── 4_stem_separation_spectrograms.png
```

---

## 🚀 Next Steps

After completing this demo:

1. **Review the code** - Look at `audio_processing.py` to see the 18 Conv2D filters
2. **Experiment** - Try different audio sources (rock, pop, classical)
3. **Study encoder layers** - See how spectrograms compress from 1025×216 → bottleneck
4. **Design your U-Net** - Decide on architecture choices for your implementation

**The foundation is set. Now it's time to build the full U-Net.**

---

## ❓ Questions?

- Check the main `README.md` for detailed technical documentation
- Review `CLAUDE.md` for project context and architecture overview
- See `vocal_separation_sanity_check/README.md` for the original POC explanation

**Ready? Let's run the demo!** 🎵
