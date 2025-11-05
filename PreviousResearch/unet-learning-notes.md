# U-Net Audio Separation Learning Notes

## Nyquist Theorem & Sample Rate

**Rule:** Maximum representable frequency = sample_rate / 2

### What is a "Cycle"?

One complete repetition of a wave pattern:

```
     ╱╲        ╱╲        ╱╲
    ╱  ╲      ╱  ╲      ╱  ╲
───╱────╲────╱────╲────╱────╲───
        ╲  ╱      ╲  ╱      ╲  ╱
         ╲╱        ╲╱        ╲╱

  |<--1 cycle-->|
```

**Important:** Zero-crossings have direction!
```
     ╱╲
    ╱  ╲
───╱────╲───  ← crosses going DOWN (different!)
    ↑   ╲  ╱
    |    ╲╱
    └─ crosses going UP

  |<--1 cycle-->|
```

One cycle = cross up → peak → cross down → trough → cross up again

Think of a pendulum: crosses center going right, then crosses center going left = 1 complete swing.

**Frequency = cycles per second:**
- 440 Hz (note A) = 440 cycles/second
- 1000 Hz = 1000 cycles/second

### Why 2 Samples Per Cycle?

**Sample once per cycle (BAD):**
```
     ╱╲        ╱╲        ╱╲
    ╱  ╲      ╱  ╲      ╱  ╲
   •────╲────•────╲────•────╲───
        ╲  ╱      ╲  ╱      ╲  ╱
         ╲╱        ╲╱        ╲╱
```
Always sample same point → can't tell it's a wave!

**Sample twice per cycle (MINIMUM):**
```
     ╱╲        ╱╲        ╱╲
    ╱  ╲      ╱  ╲      ╱  ╲
   •────•────•────•────•────•───
        ╲  ╱      ╲  ╱      ╲  ╱
         ╲╱        ╲╱        ╲╱
```
Now you can tell it's oscillating!

**Why:** When digitizing audio, you need at least 2 samples per wave cycle to reconstruct it. If a wave completes 1000 cycles per second (1000 Hz), you need at least 2000 samples/second to capture it.

**Example from our demo:**
- Sample rate: 22,050 Hz
- Max frequency: 22,050 / 2 = 11,025 Hz
- Missing: Everything above 11,025 Hz (cymbals, hi-hats, "air")

**For full music separation:**
- Use 44,100 Hz (CD quality) → captures up to 22,050 Hz
- Use 48,000 Hz (pro audio) → captures up to 24,000 Hz

**Trade-off:** Higher sample rate = better frequency range but more data to process.

### Sample Rate vs Audio Content Frequency

**Key distinction:**
- **Sample rate** (22,050 Hz, 44,100 Hz) = how fast you're taking measurements
- **Audio frequency** (440 Hz, 10,000 Hz) = how fast the sound wave oscillates

**Example: 440 Hz bell recorded at 22,050 Hz**
- Bell vibrates: 440 cycles/second
- Microphone samples: 22,050 times/second
- Result: ~50 samples per bell cycle (plenty!)

### Why Real Sounds Need High Sample Rates

A "440 Hz bell" isn't just 440 Hz - it has:
- Fundamental: 440 Hz
- Harmonics: 880 Hz, 1320 Hz, 1760 Hz, 2200 Hz...
- Attack transient: 10,000+ Hz (the initial "ding")

Sample rate must be ≥ 2× the **highest frequency component**, not just the fundamental pitch.

**Pure sine wave exception:** A perfect 440 Hz sine (no harmonics) would sound nearly identical at 22,050 Hz vs 44,100 Hz.

### Why 44,100 Hz is Standard

**Human hearing:** ~20 Hz to ~20,000 Hz
**By Nyquist:** 20,000 Hz × 2 = 40,000 Hz minimum

**44,100 Hz captures up to 22,050 Hz** → covers all human hearing

**Why not higher?**
- Most humans lose high-frequency hearing with age (15-18 kHz typical for adults)
- Double-blind tests show people can't distinguish 44.1 kHz from 96 kHz in playback
- Higher rates (96k, 192k) mostly waste storage/processing for inaudible frequencies
- Exception: Some benefit during production for processing headroom, then downsample for delivery

**For ML/source separation:** 44.1 kHz is the sweet spot.

### What is Aliasing?

**Aliasing** = high frequencies get misrepresented as lower frequencies when you sample too slowly (violate Nyquist).

**Example: 10 Hz wave sampled at only 8 Hz**
```
Actual 10 Hz:  ╱╲    ╱╲    ╱╲    ╱╲
              ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
             ╱    ╲╱    ╲╱    ╲╱    ╲

Sample at 8Hz: •      •      •      •

Reconstructed:  •─────•─────•─────•   ← Looks like ~2 Hz (wrong!)
                 ╲   ╱ ╲   ╱
                  ╲ ╱   ╲ ╱
                   •     •
```

**Real audio example:**
- Recording at 22,050 Hz (max 11,025 Hz)
- Cymbal has 15,000 Hz component
- Gets aliased to ~7,050 Hz (22,050 - 15,000)
- Sounds like weird digital artifacts

### Anti-Aliasing Filter

**Solution:** Low-pass filter BEFORE sampling to remove frequencies above Nyquist limit.

**Process:**
1. Analog signal comes in (full frequency range)
2. Anti-aliasing filter cuts everything above Nyquist limit
   - At 44.1 kHz: remove above ~20 kHz
   - At 22.05 kHz: remove above ~10 kHz
3. Now safe to sample - no high frequencies left to alias

**The trade-off:**
- At 44.1 kHz: filter must be steep (pass 20 kHz, block 22.05 kHz = only 2 kHz transition!)
- Steep filters can cause phase shifts
- At 96 kHz: gentler filter possible (pass 20 kHz, block 48 kHz = 28 kHz transition)
- This is one legitimate argument for higher sample rates (though modern converters minimize the issue)

---

## Understanding the Sanity Check Demo

### STFT Basics
- **n_fft = 2048** → creates **1025 frequency bins** (2048/2 + 1)
- **Bin width** = sample_rate / n_fft
  - At 22,050 Hz: 22,050 / 2048 = ~10.77 Hz per bin
  - At 44,100 Hz: 44,100 / 2048 = ~21.5 Hz per bin
- Each bin captures a **range** of frequencies, not a single exact frequency
  - Bin 0: 0-10.77 Hz (at 22kHz) or 0-21.5 Hz (at 44kHz)
  - Bin 20: ~215-236 Hz
  - Bin 1024: Nyquist frequency (11,025 Hz or 22,050 Hz)

### Original Demo Structure (sanity_check_complete.py)

**What it creates:**
- 1025 frequency bins × 102 time windows = 104,550 STFT values
- 18 different Conv2D slices (horizontal, vertical, diagonal edges, etc.)
- 425 metrics extracted per window (400-point freq profile + 25 derived)
- Total fingerprint: 425 × 102 = 43,350 values per audio file

**The hidden simplification:**
All 18 slices get created and fingerprinted, but optimization only uses slice_0_raw:
```python
vocal_fp = vocal_fps['slice_0_raw'][win_idx]['freq_profile_400']  # for simplicity
```

This makes it essentially a **librosa + gradient descent demo**, NOT a real Conv2D demo.

### Why Conv2D Matters for Source Separation

Different Conv2D kernels capture different vocal/instrument characteristics:
- **Horizontal** → sustained frequencies (held notes)
- **Vertical** → onsets/attacks (consonants, drum hits)
- **Diagonal** → pitch glides, vibrato
- **Harmonic** → overtone structure (very vocal-specific!)
- **Edge detection** → formant transitions

Using all 18 provides richer information about what makes something "vocal-like" vs "instrument-like."

---

## Three Experimental Versions Created

We created three versions to explore different analysis approaches, all using 44.1kHz/24-bit audio.

### Version 1: `sanity_check_complete--original-but-plus-18conv2d.py`
**Location:** `output/18conv2d/`

**What's different:**
- Actually USES all 18 Conv2D slices in optimization
- Loops through all slices and combines their losses/gradients
- Each slice contributes to the EQ curve learning

**Structure:**
```python
for slice_name in vocal_fps.keys():  # All 18 slices
    vocal_fp = vocal_fps[slice_name][win_idx]['freq_profile_400']
    # Compute loss and gradient for this slice
    # Accumulate across all slices
```

**Purpose:** Test if multi-scale Conv2D features improve separation quality

**Fingerprint size:** 425 metrics × 100 windows × 18 slices (but combined during optimization)

---

### Version 2: `sanity_check_bin-by-bin-1025-by-100.py`
**Location:** `output/bin-by-bin-1025x100/`

**What's different:**
- Analyzes each frequency bin independently at each time window
- Creates a sparse "spectrum" with only one bin populated
- Extracts 425 metrics from each (bin, window) combination

**Structure:**
```python
for bin_idx in range(1025):
    for window_idx in range(100):
        sparse_window = zeros(1025)
        sparse_window[bin_idx] = magnitude[bin_idx, window_idx]
        metrics = extract_425_metrics(sparse_window)
```

**Purpose:** Maximum granularity - every frequency analyzed independently at every moment

**Fingerprint size:** 1025 bins × 100 windows × 425 metrics = **43,562,500 values**

**Key insight:** Most metrics will be degenerate (can't find harmonics in one bin), but energy and some temporal patterns might be informative.

---

### Version 3: `sanity_check_bin-by-bin-1025-by-1.py`
**Location:** `output/bin-by-bin-1025x1/`

**What's different:**
- Treats each frequency band as a time-series across the entire clip
- Samples each band 100 times temporally
- Analyzes temporal evolution within each frequency band

**Structure:**
```python
for bin_idx in range(1025):
    band_over_time = magnitude[bin_idx, :]  # All windows for this band
    for sample_idx in [0, 1, 2, ..., 99]:  # 100 temporal samples
        sparse_window = zeros(1025)
        sparse_window[bin_idx] = band_over_time[sample_idx]
        metrics = extract_425_metrics(sparse_window)
```

**Purpose:** Analyze temporal patterns within each frequency band independently

**Fingerprint size:** 1025 bands × 100 samples × 425 metrics = **43,562,500 values**

**Key insight:** Frequency-first approach - how does each band behave over time?

---

## Key Changes Made

### 1. Sample Rate: 22,050 Hz → 44,100 Hz
**Why:**
- Captures full human hearing range (0-22,050 Hz) instead of just 0-11,025 Hz
- Preserves cymbal, hi-hat, and "air" frequencies
- Bin width: 21.5 Hz instead of 10.77 Hz

### 2. Bit Depth: 16-bit → 24-bit
**Why:**
- Preserve original audio quality (144 dB dynamic range vs 96 dB)
- Avoid potential quantization artifacts
- No downside for analysis purposes

**Note:** librosa loads as 32-bit float internally for processing, then saves as 24-bit PCM.

### 3. Conv2D Usage
**Original:** Created 18 slices but only used slice_0_raw (no Conv2D benefit)
**Version 1:** Actually uses all 18 Conv2D slices
**Versions 2 & 3:** Removed Conv2D entirely (analyzing raw bins)

---

## Understanding Loss and Optimization

**Loss = measurement of "how wrong we are"**
- High loss = mixture looks very different from vocal target
- Low loss = mixture looks similar to vocal target

**Gradient descent process:**
1. Start with neutral EQ (all 1.0s)
2. Measure loss: how different is mixture from vocal?
3. Calculate gradient: which direction reduces loss?
4. Update EQ: `eq_curve -= learning_rate * gradient`
5. Repeat 100 times

**Why accumulate loss across bins/slices:**
When we have multiple fingerprints (1025 bins or 18 slices), we:
- Measure loss for each one
- Average them together
- Use combined loss to update EQ

This ensures the EQ curve considers ALL the detailed analysis, not just one piece.

**Example from bin-by-bin:**
```python
for bin_idx in range(1025):
    loss_for_this_bin = compare(mixture, vocal_target_for_bin)
    total_loss += loss_for_this_bin
average_loss = total_loss / 1025  # Combined measurement
```

---

## Audio Processing Chain

### Preparation (prepare_audio_files-44khz.py)
1. Load audio (any format, any sample rate, any bit depth)
2. Convert to mono if stereo
3. Resample to 44,100 Hz
4. Trim/pad to exactly 4.7 seconds
5. Normalize to ±1.0 range
6. Save as 24-bit PCM WAV

### Analysis (sanity_check_*.py)
1. Load prepared audio
2. Create STFT (1025 bins × ~100 windows)
3. Extract fingerprints (methodology varies by version)
4. Optimize EQ curves using gradient descent
5. Apply learned EQ to mixture
6. Reconstruct separated vocal
7. Save results in version-specific output folder

---
