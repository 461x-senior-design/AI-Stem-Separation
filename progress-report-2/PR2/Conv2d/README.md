# Conv2d Vocal Separation - Best Implementation

This folder contains the **production-ready Conv2D vocal separation** code, extracted from the research in `vocal_separation_sanity_check/`.

## 🎯 What's Here

This is the **optimized Conv2D spectral fingerprinting approach** that achieves vocal separation through:
- 18 different Conv2D filter views of spectrograms (horizontal, vertical, diagonal, harmonics, edge detectors, pooling)
- Strategy 3 dB-space optimization for numerical stability
- Direct 400-point interpolation (no encoder/decoder complexity)
- Gradient descent to match mixture fingerprint to vocal fingerprint

## 📁 Files

### Core Scripts

**`sanity_check_progressive_CURRENT_WORKING.py`**
- The verified working version with Strategy 3 dB-space optimization
- Tests progressive separation using 1 to N slices
- Default: 18 slices (but 8 is optimal - see below)
- Usage: `python sanity_check_progressive_CURRENT_WORKING.py --slices 8`

**`best.py`**
- **The recommended script** - advanced version with greedy algorithm to find optimal slice ordering
- Automatically tests both original progressive order AND greedy-optimized order
- **Critical finding**: 8 slices in original order gives the BEST results (loss=1.181991)
- Outputs detailed comparison plots and audio files showing progressive improvement
- Usage: `python best.py --slices 8`
- **What it does:**
  1. Runs greedy algorithm to find optimal slice ordering
  2. Tests progressive separation (1 to N slices) in original order
  3. Tests the greedy-discovered optimal ordering
  4. Generates comparison plots: loss progression, final loss comparison, spectrograms
  5. Identifies and saves the overall best result
- **Output directory:** `output_progressive_Nslices/` with subdirectories for each test

### Audio Data

**`rtg/100-window/`** - Processed, standardized audio ready for testing
- `stereo_mixture.wav` - Full mixture (instruments + vocals)
- `isolated_vocal.wav` - Isolated vocal track (ground truth)
- `stereo_instrumental.wav` - Instrumental track (mixture - vocal)

**`process/100-window/`** - Source audio files (various formats)
- Original paired audio files before standardization
- `*_100-full.wav` - Full mixture files
- `*_100-stem.wav` - Isolated vocal stems

### Dependencies

**`requirements.txt`**
```
numpy
librosa
soundfile
scipy
matplotlib
```

## 🚀 Quick Start

### Install Dependencies

```bash
# From the Conv2d directory
pip install -r requirements.txt
```

### Run the Best Version (8 Slices)

```bash
python best.py --slices 8
```

**What this does:**
1. Tests slices 1 through 8 in original order (optimal)
2. Tests greedy-optimized ordering of all 8 slices
3. Compares results and identifies best approach
4. Outputs to `output_progressive_8slices/`

### Run Progressive Testing (Any Number of Slices)

```bash
# Test with 5 slices (fast)
python sanity_check_progressive_CURRENT_WORKING.py --slices 5

# Test with 8 slices (optimal)
python sanity_check_progressive_CURRENT_WORKING.py --slices 8

# Test with all 18 slices (comprehensive but worse results)
python sanity_check_progressive_CURRENT_WORKING.py --slices 18
```

## 📊 Key Findings

### Optimal Configuration: 8 Slices

From extensive testing, **using only the first 8 slices gives the BEST results:**

```
Slice Order (1-8):
  1. slice_1_horizontal     - Sustained frequencies
  2. slice_2_vertical       - Onset detection
  3. slice_3_diagonal_up    - Pitch sweeps upward
  4. slice_4_diagonal_down  - Pitch sweeps downward
  5. slice_5_blob           - Blob detection
  6. slice_6_harmonic       - Harmonic stacks
  7. slice_7_highpass       - Edge detection
  8. slice_8_lowpass        - Smoothing

Results:
  Loss: 1.181991 (BEST)
  Quality: ~70-80% vocal separation
```

**Adding slices 9-18 actually HURTS performance:**
- 10 slices: loss=9.670357
- 18 slices: loss=7.955098
- Greedy reordering: loss=6.905200

## 🔬 Technical Details

### The 18 Conv2D Slices

Each slice is a different Conv2D filter applied to the magnitude spectrogram:

| Slice # | Name | Purpose |
|---------|------|---------|
| 0 | Raw | Baseline magnitude spectrogram |
| 1 | Horizontal | Sustained frequencies (vowels) |
| 2 | Vertical | Transients/onsets (consonants) |
| 3 | Diagonal Up | Pitch sweeps upward |
| 4 | Diagonal Down | Pitch sweeps downward |
| 5 | Blob | Blob detection (resonances) |
| 6 | Harmonic | Harmonic stacks (formants) |
| 7 | High-pass | Edge detection (timbre) |
| 8 | Low-pass | Smoothing (fundamentals) |
| 9-15 | Edge 22°-157° | Oriented edge detectors |
| 16 | Laplacian | All edges |
| 17 | MaxPool | Downsampled peaks |
| 18 | AvgPool | Downsampled averages |

### Strategy 3: dB-Space Optimization

The key to numerical stability:

```python
# Convert to dB space
working_mag_db = librosa.amplitude_to_db(working_mag, ref=1.0, amin=1e-8)

# Initialize EQ curves (additive in dB space)
eq_curves_db = [np.zeros(400) for _ in range(num_windows)]

# Optimization loop
for iteration in range(num_iterations):
    vocal_fp_db = librosa.amplitude_to_db(vocal_fp, ref=1.0, amin=1e-8)
    adjusted_fp_db = mixture_fp_db + eq_curves_db[win_idx]  # Additive!
    gradient = 2 * (adjusted_fp_db - vocal_fp_db)
    eq_curves_db[win_idx] -= learning_rate * gradient
    eq_curves_db[win_idx] = np.clip(eq_curves_db[win_idx], -20, 20)

# Convert back to linear
refined_mag = librosa.db_to_amplitude(refined_mag_db, ref=1.0)
```

### Configuration

```python
CONFIG = {
    'sr': 22050,           # Sample rate
    'duration': 4.7,       # Audio duration (seconds)
    'n_fft': 2048,         # FFT window size
    'hop_length': 1040,    # Hop length (adjusted for exactly 100 windows)
    'num_iterations': 100, # Gradient descent iterations per slice
    'learning_rate': 0.01, # Learning rate for EQ optimization
}
```

## 📈 Expected Output

After running `best.py --slices 8`:

```
output_progressive_8slices/
├── 1_slices/
│   ├── extracted_vocal.wav
│   ├── 1_original_mixture.wav
│   └── 2_target_vocal.wav
├── 2_slices/
│   └── extracted_vocal.wav
├── ...
├── 8_slices/
│   └── extracted_vocal.wav        ← BEST RESULT (Loss: 1.181991)
├── best_sequence_found/
│   └── extracted_vocal_best_sequence.wav
├── all_tests_loss_progression.png
├── final_loss_comparison.png
└── spectrogram_comparison.png
```

**Listen to:** `output_progressive_8slices/8_slices/extracted_vocal.wav`

## 🎧 Audio Quality

**What to expect:**
- ✅ Clear vocal separation (70-80% quality)
- ✅ Drums/bass significantly reduced
- ✅ Vocal intelligibility preserved
- ⚠️ Some artifacts/phasing (normal for non-ML approach)
- ⚠️ Not perfect - this is proof-of-concept quality

**This demonstrates that Conv2D spectral fingerprinting contains enough information for source separation, validating the approach for U-Net implementation.**

## 🔬 Research Notes

This code represents the culmination of research into:
1. NumPy → librosa migration for numerical stability
2. Testing 4 gain strategies (Strategy 3 dB-space won)
3. Progressive testing to find optimal slice count
4. Greedy algorithm for optimal slice ordering

**Key discovery**: Simple is better. The first 8 basic pattern detectors (horizontal, vertical, diagonal, blob, harmonic, high-pass, low-pass) contain all the essential information. Complex edge detectors and pooling operations add noise.

## 🚀 Next Steps

This Conv2D approach demonstrates the **feasibility** of spectral fingerprinting for source separation. The next phase is to:

1. **Replace hand-crafted filters with learned filters** - Use a U-Net to learn optimal Conv2D filters
2. **Train on large dataset** - Learn patterns across thousands of songs
3. **Scale to 4-stem separation** - Extend from vocals-only to (vocals, drums, bass, other)

The insights from this work inform U-Net architecture design:
- Focus on basic pattern detectors (first 8 slices)
- Use dB-space processing for stability
- Optimize for 100 time windows (hop_length=1040)
- 400-point frequency representation is sufficient

---

## 📖 References

See the original research in:
- `PreviousResearch/librosa-demo 2/vocal_separation_sanity_check/`
- Documentation: `START_HERE.md`, `README.md`, `COMPLETE_DOC.md`
