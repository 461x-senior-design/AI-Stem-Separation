# Vocal Separation Sanity Check - Technical Deep Dive

Just documenting everything about this experiment in case I need to reference it later or someone wants to understand what's actually happening.

## What This Is

Trying to separate vocals from a mix without training a neural network. Just want to see if the core approach works.

The idea: create a super detailed "fingerprint" of what the vocal looks like (765,000 measurements), then adjust the mixture until its fingerprint matches. If that works even a little, then training a U-Net should work way better.

Expected outcome: maybe 70-80% quality separation. Enough to prove it works.

---

## Why 18 Slices?

Different conv filters catch different patterns that vocals have:

- **Horizontal lines** - vocals sustain frequencies (drums don't)
- **Harmonic stacks** - vocals are pitched (cymbals aren't)
- **Mid-range energy** - vocals live at 500-4000 Hz mostly
- **Formant peaks** - vowel sounds create specific patterns
- **Smooth energy** - vocals don't spike like percussion

One slice isn't enough because you can fake any single pattern. But faking all 18 patterns simultaneously across 100 time windows with 425 metrics each? That's only going to happen if you're actually the vocal.

## The Encoder Compression

Starting with 1025 frequency bins per time window, compress it down through 4 layers:

```
1025 bins (raw STFT)
  ↓ MaxPool 2x
512 bins (layer 1)
  ↓ MaxPool 2x  
256 bins (layer 2) ← pull band energies from here
  ↓ MaxPool 2x
128 bins (layer 3) ← detect harmonics here
  ↓ MaxPool 2x
64 bins (layer 4) ← spectral shape metrics
  ↓ MaxPool 2x
32 bins (bottleneck) ← most compressed
```

At the bottleneck it's not "2340 Hz has this much energy" anymore. It's more like "this sounds vocal-ish" or "this sounds drum-ish". Abstract features.

## The 400-Point Frequency Profile

This is the key part. Instead of just saying "mid energy = 8.2", we store the energy at every 50 Hz from 0 to 20 kHz. That's 400 points.

So:
- Profile[0] = energy at 0 Hz
- Profile[1] = energy at 50 Hz  
- Profile[2] = energy at 100 Hz
- ...
- Profile[36] = energy at 1800 Hz (vocal formant range)
- ...
- Profile[400] = energy at 20 kHz

Like a 400-band graphic EQ readout. Shows exactly where the energy is.

Why this matters: drums have energy at different frequencies than vocals. With 400 independent points, we can adjust each band separately to match the vocal's profile.

## The Pipeline

### Phase 1: Analysis

```
Load audio → STFT → magnitude spectrogram
  ↓
Create 18 slices (different conv filters)
  ↓
For each slice:
  For each time window:
    - Run through 4 encoder layers
    - Extract 425 metrics at bottleneck
  ↓
Result: 18 slices × 100 windows × 425 metrics = 765,000 data points
```

Do this for both the isolated vocal and the mixture.

### Phase 2: Optimization

```
Initialize 100 EQ curves (one per window, 400 points each)
  ↓
For 100 iterations:
  For each window:
    - Apply current EQ to mixture window
    - Compare to vocal window (MSE loss)
    - Compute gradient
    - Update EQ curve (gradient descent)
  ↓
Result: learned 400-point EQ curve for every window
```

Total parameters learned: 100 windows × 400 EQ points = 40,000

### Phase 3: Reconstruction

```
For each window:
  - Take mixture magnitude
  - Apply learned EQ curve
  - Store adjusted magnitude
  ↓
Combine adjusted magnitude with original phase
  ↓
Inverse STFT → audio waveform
  ↓
Normalize and save
```

---

## Implementation Details

### Functions

**create_18_slices(mag_spec)**
- Takes the raw magnitude spectrogram
- Applies 18 different conv2d filters + pooling operations
- Returns dict with 18 different views

**window_to_bottleneck(window, sr)**
- Takes one time window (1025 freq bins)
- Downsamples 5 times (1025 → 512 → 256 → 128 → 64 → 32)
- Extracts 425 metrics
- Returns dict with all the measurements

**optimize_eq_curves(vocal_fps, mix_fps, ...)**
- Runs gradient descent for N iterations
- Learns 400-point EQ curve for each window
- Returns list of learned curves

**reconstruct_vocal(mix_stft, eq_curves, ...)**
- Applies learned EQ to mixture
- Does inverse STFT
- Returns separated audio

### Performance

On my machine (CPU):
- Phase 1 (analysis): ~5 seconds for 4.7 sec audio
- Phase 2 (optimization): ~2-3 minutes for 100 iterations
- Phase 3 (reconstruction): ~1 second

Total: about 3-4 minutes for a short clip.

Full song would be proportionally longer. 3 minute song = probably 30-40 minutes.

---

## Expected Results

Not expecting perfection here. This is manual proof-of-concept.

### Good outcome:
- Vocal is audible and mostly clear
- Drums way quieter  
- Bass way quieter
- Some artifacts (clicking, phasing) but not too bad
- Overall sounds like a vocal

### Bad outcome:
- Can't hear the vocal
- Everything still sounds like the full mix
- Tons of artifacts
- Worse than just the original mix

Target is somewhere in the 60-80% quality range. If we hit that, it proves the concept. Then U-Net training should get to 95%+.

### Failure modes:

1. **Not enough slices** - fingerprint isn't unique enough
2. **Wrong metrics** - missing important features that distinguish vocals
3. **Bad optimization** - learning rate wrong, not enough iterations, etc
4. **Fundamental flaw** - the whole approach doesn't work

If it fails completely, at least we know before wasting time training.

---

## Bridging to U-Net

### What's the same:
- Input is mixture spectrogram
- Create multiple "views" (our 18 slices = their learned conv filters)
- Compress through encoder layers
- Extract features at bottleneck
- Decode back to full resolution
- Output is separated spectrogram

### What's different:
- **Our version:** hand-designed slices, manual optimization, per-song, takes minutes
- **U-Net version:** learned filters, trained once on thousands of songs, generalizes to any song, takes 10ms

But the core idea is the same. If our manual version works at all, U-Net should be able to learn it way better.

---

## Key Insights

### 1. It's not magic
Neural networks are just optimization. If you can manually solve a problem (even slowly), a network can probably learn to solve it faster and better.

### 2. Multi-scale matters
Vocals look different at different scales. Need to capture patterns at multiple resolutions. Single-scale analysis misses too much.

### 3. Unique fingerprints work
765,000 measurements is enough to uniquely identify a vocal. Nothing else can match all those points simultaneously.

### 4. Encoder-decoder architecture makes sense
Compress to abstract features (what), then expand back to concrete details (where). The bottleneck forces the model to learn meaningful representations.

### 5. Skip connections are important
Bottleneck loses fine details on purpose (that's the compression). Skip connections let the decoder recover those details during reconstruction.

---

## Next Steps

If this works:

1. Look at which slices were most useful - can we drop some?
2. Look at which metrics mattered - can we simplify?
3. Design U-Net architecture based on what worked here
4. Gather dataset (thousands of vocal + mixture pairs)
5. Train U-Net
6. Compare trained model to this baseline

If it doesn't work:

1. Figure out why (fingerprint? optimization? fundamental?)
2. Try different slices
3. Try different metrics  
4. Try different optimization approach
5. Re-evaluate if the whole concept is sound

---

## Files

- `sanity_check.py` - phases 1-2 only (no output audio)
- `sanity_check_complete.py` - full pipeline, short clip
- `sanity_check_full_length.py` - full pipeline, entire song
- `prepare_audio_files.py` - audio file prep helper
- `test_setup.py` - verify dependencies
- `requirements.txt` - package versions
- `README.md` - user guide
- `COMPLETE_DOC.md` - this file

---

## References

Papers that helped:
- U-Net: Ronneberger et al 2015
- Spleeter: Deezer 2019
- Wave-U-Net: Stoller et al 2018

Libraries:
- librosa for audio processing
- scipy for signal processing
- numpy for everything else

---

## Random Notes

The 400-point frequency profile was the breakthrough. Before that I was just using 6 band energies (bass/mid/treble/etc) and it wasn't unique enough. Going to 400 points gave enough resolution to actually distinguish sources.

The 18 slices thing - started with just 3 (raw, horizontal, vertical). Kept adding more until it worked. 18 seems like enough. Maybe could use fewer, haven't tested.

Optimization is sensitive to learning rate. Too high and it diverges, too low and it takes forever. 0.01 seems to work okay. 100 iterations is probably enough for proof-of-concept but more would be better.

Phase preservation is important. We're only adjusting magnitude, keeping the original phase. Probably not optimal but it works and it's simple.

The fingerprint comparison happens at the slice_0_raw level for simplicity. Could probably improve results by comparing across all 18 slices but that's more compute.

---

## FAQ

**Why not just use Spleeter?**  
This isn't about making a production tool. It's about understanding what's actually happening before training a neural network. Makes the architecture decisions way more informed.

**Why manual optimization instead of just training?**  
Want to prove the approach works first. If this doesn't work at all, then training won't help. Better to find out in a few minutes than after days of training.

**Can this work for other instruments?**  
Probably yeah. Just need (mixture, target instrument) pairs. The fingerprinting approach should work for anything with distinct spectral characteristics.

**How long would it take to train a U-Net?**  
Depends on dataset size and hardware. Probably a few days on a decent GPU. Few hours if you have multiple GPUs.

**What's the dataset size needed?**  
Thousands of examples minimum. More is better. Quality matters more than quantity though. Better to have 1000 good pairs than 10000 mediocre pairs.

---

That's about it. This document is mostly for my own reference but if someone else is trying to understand the approach, hopefully this helps.
