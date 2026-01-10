# 🎯 START HERE - Vocal Separation Sanity Check

## What Is This?

This is a **proof-of-concept experiment** to answer one critical question before investing weeks in U-Net training:

**"Can we separate vocals from a mix using spectral fingerprinting alone, without any neural network?"**

If the answer is yes (even with 70-80% quality), then training a U-Net to do the same thing automatically should achieve 95%+ quality and run 1800× faster.

**This is validation research.** We're testing the core hypothesis manually before automating it with deep learning.

---

## 🧠 The Core Idea

### Manual Feature Engineering (What This Does)

1. Take an isolated vocal + full mixture (both containing the same vocal)
2. Create **18 different Conv2D-filtered views** of each spectrogram
3. Compress each view through **4 encoder layers** to a bottleneck
4. Extract **425 metrics per time window** (~765,000 total data points)
5. Use gradient descent to **optimize EQ curves** until the mixture's fingerprint matches the vocal's
6. Apply those EQ curves to separate the vocal

**Time:** A few minutes per song
**Quality:** 70-80% separation (proof-of-concept)

### What U-Net Will Do (After Training)

1. Learn the optimal Conv2D filters automatically (instead of our hand-crafted 18 slices)
2. Learn the encoder architecture automatically (instead of our fixed 4 layers)
3. Predict separation masks directly (instead of optimizing EQ curves)

**Time:** ~10ms per song
**Quality:** 95%+ separation (production-ready)

---

## 🎓 Why This Matters

### Risk Mitigation

Training a U-Net is expensive:
- **Time:** Days to weeks of GPU training
- **Data:** Need thousands of (vocal, mixture) pairs
- **Compute:** Significant cloud costs or local GPU usage

**Before investing that, we need to know if the approach even works.**

This sanity check proves:
- ✅ Spectral fingerprinting contains enough information to separate vocals
- ✅ The encoder-decoder architecture makes sense for this task
- ✅ Multi-scale Conv2D filters can detect vocal-specific patterns
- ✅ Optimization can find the right separation parameters

If this fails, we know the core idea is flawed. If it succeeds, U-Net training is a safe bet.

### Learning Value

Even if you never train the U-Net, this experiment teaches:
- How Conv2D filters extract patterns from spectrograms
- How encoder layers compress information to bottlenecks
- How optimization finds solutions in high-dimensional spaces
- How audio separation works at a fundamental level

**You'll understand encoder-decoder architectures intuitively after running this.**

---

## 🔧 Setup

### Install UV (if needed)

```bash
pip install uv
```

**Why UV?** Fast, reliable, no virtual environments to manage. Just works.

### Install Dependencies

```bash
# From the vocal_separation_sanity_check directory
uv pip install numpy librosa soundfile scipy matplotlib
```

**Note:** No ffmpeg needed unless you want FLAC support. WAV files work out of the box.

---

## 🚀 Running the Experiment

### Option 1: Quick Test (Recommended for First Run)

**Processes ~4.7 seconds of audio (100 time windows)**

#### Step 1: Prepare Audio Files

Place paired audio in `process/100-window/`:
```
process/100-window/
├── mysong_100-full.wav    (full mixture with vocals, drums, bass, etc.)
└── mysong_100-stem.wav    (isolated vocal only, same timing)
```

**Critical:** Files must be **time-aligned** (same start point, same duration). Use Audacity's Time Shift Tool if needed.

#### Step 2: Prepare Files

```bash
python prepare_audio_files.py
```

**What this does:** Converts to mono, standardizes sample rate, moves to `rtg/100-window/`

#### Step 3: Run Sanity Check

```bash
python sanity_check_complete.py
```

**What this does:**
- Creates 18 spectral fingerprints (slice_0 through slice_17)
- Compresses each through 4 encoder layers
- Extracts 425 metrics × 100 windows × 18 slices = 765,000 data points
- Optimizes mixture to match vocal fingerprint (100 iterations)
- Reconstructs separated vocal audio

**Output:** `output/extracted_vocal.wav` + visualizations

**Time:** 2-5 minutes depending on your CPU

---

### Option 2: Full Song (After Quick Test Works)

**Processes entire song (all time windows)**

#### Step 1: Prepare Audio Files

Place paired audio in `process/no-limit/`:
```
process/no-limit/
├── track_nl-full.wav    (full mixture)
└── track_nl-stem.wav    (isolated vocal)
```

#### Step 2: Prepare Files

```bash
python prepare_audio_files.py
```

#### Step 3: Run Sanity Check

```bash
python sanity_check_full_length.py
```

**Output:** `output_full/extracted_vocal.wav` + visualizations

**Time:** 10-30 minutes depending on song length

---

## 📊 Understanding the Output

### Files Generated

```
output/
├── extracted_vocal.wav              # The separated vocal (your result!)
├── 1_original_mixture.wav           # Copy of input mixture (for comparison)
├── 2_target_vocal.wav               # Copy of target vocal (ground truth)
├── optimization_loss.png            # Shows learning curve (loss should decrease)
└── spectrograms.png                 # Visual comparison of spectrograms
```

### Quality Expectations

**What "70-80% quality" means:**

✅ **Should work:**
- Vocal is clearly audible and intelligible
- Drums are significantly quieter (~80% reduction)
- Bass is significantly quieter (~80% reduction)
- You can recognize it as the vocal from the song

❌ **Won't work:**
- Studio-quality isolation (some artifacts expected)
- Complete instrument removal (some bleed-through normal)
- Perfect timbre preservation (some phasing/clicking possible)
- Zero background noise

**If you can hear the vocal clearly with reduced background, the experiment succeeded.**

---

## 🧪 The 18 Spectral "Slices"

Each slice applies a different Conv2D filter to detect specific patterns:

| Slice | Pattern Detected | Why It Matters for Vocals |
|-------|------------------|---------------------------|
| 0 | Raw magnitude | Baseline (no filtering) |
| 1 | Horizontal lines | Sustained notes, vowels |
| 2 | Vertical lines | Transients, consonants |
| 3 | Diagonal (↗) | Pitch sweeps, vibrato |
| 4 | Diagonal (↘) | Downward inflections |
| 5 | Blob detection | Formant clusters |
| 6 | Harmonic stack | Fundamental + overtones |
| 7-8 | Edge detection | Timbre boundaries |
| 9-15 | Oriented edges | Directional patterns at 22.5°, 45°, 67.5°, 90°, etc. |
| 16 | Laplacian | High-frequency detail |
| 17-18 | Pooled versions | Multi-scale averaging |

**Each slice captures something unique about how vocals look in a spectrogram.** Combined, they create a fingerprint that only the actual vocal can match.

---

## 🎯 The Fingerprint (425 Metrics Per Window)

For each ~46ms time window, we extract:

- **400-point frequency profile:** Energy at 0Hz, 50Hz, 100Hz, ..., 20kHz
- **6 band energies:** Bass, low-mid, mid, high-mid, presence, high
- **6 spectral shape metrics:** Centroid, spread, rolloff, flatness, slope, crest
- **5 harmonic features:** Fundamental frequency, # harmonics, spacing, strength, etc.
- **4 formant measurements:** Vowel-specific frequency peaks
- **4 dynamics measurements:** Attack, decay, sustain, release characteristics

**Total:** 425 metrics × 100 windows × 18 slices = **765,000 unique measurements**

This fingerprint should be unique enough that only the actual vocal matches it.

---

## 🐛 Troubleshooting

### "NO FILES FOUND"
**Problem:** Script can't locate your audio files
**Solution:**
- Check file naming: Must end with `_100-full.wav` + `_100-stem.wav` (or `_nl-` versions)
- Check directory: Files must be in `process/100-window/` or `process/no-limit/`
- Case-sensitive on Linux/macOS

### "FileNotFoundError: rtg/100-window/"
**Problem:** Haven't run preparation script
**Solution:** Run `python prepare_audio_files.py` first

### Takes Forever
**Problem:** Full-length version processing entire song
**Solution:**
- Use 100-window version instead (processes only 4.7s)
- Reduce iterations in config (edit `sanity_check_complete.py`)
- Close other applications to free up CPU

### Out of Memory
**Problem:** High-resolution spectrograms use too much RAM
**Solution:**
- Reduce `n_fft` from 2048 to 1024 (edit config in script)
- Use shorter audio clips
- Close other applications

### Results Sound Terrible
**Problem:** Poor separation quality
**Solution:**
- **Check alignment:** Files must start at same time (use Audacity Time Shift)
- **Check files:** "full" = mixture, "stem" = isolated vocal (don't swap them)
- **Try different audio:** Some sources separate better (clear vocals work best)
- **Verify ground truth:** Make sure your "stem" file is actually the isolated vocal

---

## ✅ Success Criteria

### The experiment succeeded if:

1. **Vocal is audible** - You can hear and understand the separated vocal
2. **Instruments are quieter** - Drums/bass reduced by ~70-80%
3. **Loss decreases** - Check `optimization_loss.png` - should trend downward
4. **Spectrograms match** - Visual comparison shows similarity to target

### The experiment failed if:

1. **Vocal is inaudible** - Can't hear the vocal at all
2. **No instrument reduction** - Sounds same as original mixture
3. **Loss doesn't decrease** - Optimization isn't working
4. **Complete noise** - Output is just static/garbage

**If 70%+ separation works, the core hypothesis is validated.** U-Net training is the next step.

---

## 🧠 What This Teaches You

### Conceptual Understanding

After running this, you'll intuitively understand:

1. **Why spectrograms matter** - 2D representations enable Conv2D pattern detection
2. **What Conv2D filters detect** - Not just edges, but musical patterns (harmonics, formants)
3. **How encoders work** - Compression extracts essential features, discards noise
4. **Why bottlenecks exist** - Force model to learn compact representations
5. **What optimization does** - Search high-dimensional space for best parameters

**This is the hard part of understanding U-Net.** Once you get this, the rest is just architecture.

### Technical Skills

You'll gain hands-on experience with:
- LibROSA spectral transformations (STFT/ISTFT)
- NumPy array manipulation at scale
- SciPy optimization (gradient descent)
- Multi-scale feature engineering
- Audio signal processing fundamentals

---

## 🚀 Next Steps After Validation

### If This Works (70-80% Quality)

1. **Analyze results** - Which of the 18 slices were most useful? Check the code.
2. **Design U-Net** - Mirror this architecture (18 input channels, 4 encoder layers, etc.)
3. **Gather dataset** - Find 1000+ (vocal, mixture) pairs for training
4. **Train model** - Use PyTorch to automate what we did manually
5. **Compare** - Trained U-Net should achieve 95%+ quality in 10ms

### If This Doesn't Work (<50% Quality)

1. **Debug fingerprint** - Are 18 slices enough? Need more?
2. **Check encoder** - Are 4 layers sufficient? Too many?
3. **Verify metrics** - Are 425 measurements capturing the right features?
4. **Try different approach** - Maybe spectral fingerprinting isn't the right path

**This experiment tells you whether to proceed with U-Net training or pivot to a different approach.**

---

## 📁 File Descriptions

- **`sanity_check_complete.py`** - Full pipeline (100 windows, 4.7s)
- **`sanity_check_full_length.py`** - Full pipeline (entire song)
- **`sanity_check.py`** - Analysis only (no audio output, just fingerprints)
- **`prepare_audio_files.py`** - Converts and standardizes your audio
- **`test_setup.py`** - Verifies dependencies are installed correctly
- **`requirements.txt`** - List of Python packages needed

---

## 💡 Pro Tips

### Getting Best Results

1. **Use clear vocals** - Solo singing works better than dense harmonies
2. **Use well-mixed audio** - Professional recordings separate better than demos
3. **Use time-aligned files** - Alignment matters more than you think
4. **Start with 100-window** - Debug on short clips before processing full songs
5. **Check visualizations** - `spectrograms.png` shows if optimization is working

### Common Mistakes

1. **Swapped files** - "full" and "stem" reversed (vocal in full, mixture in stem)
2. **Misaligned files** - Vocal starts 0.5s earlier/later than mixture
3. **Wrong format** - Stereo files instead of mono (script expects mono)
4. **Bad ground truth** - "stem" file has background bleed (not truly isolated)

---

## ❓ Questions You Might Have

**Q: Why not just train the U-Net directly?**
A: Training takes days/weeks and costs money. This proves the idea works in minutes.

**Q: Why 18 slices specifically?**
A: Experimentation. You could try 12 or 24. 18 balances coverage with computation time.

**Q: Why 425 metrics per window?**
A: Comprehensive feature set. Could be reduced, but more data = better fingerprint.

**Q: Will this work on instruments besides vocals?**
A: Maybe! Try it on drums or bass. Might need different slices/metrics.

**Q: Can I use this for production?**
A: No, it's too slow (minutes vs milliseconds). But it validates training a real-time U-Net.

**Q: What if I don't have isolated vocals?**
A: Use public datasets like MUSDB18 (provides stems). Or use AI separation tools to create ground truth.

---

## 🎯 The Bottom Line

**This is a validation experiment, not a production tool.**

**Goal:** Prove that spectral fingerprinting can separate vocals (70-80% quality)
**If successful:** Proceed with U-Net training (target: 95%+ quality, 10ms inference)
**If unsuccessful:** Pivot to different approach before investing in training

**Ready to validate the hypothesis? Let's run the experiment!** 🎵
