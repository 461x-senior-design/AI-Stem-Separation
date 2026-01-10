# Vocal Separation Sanity Check

Trying to prove that you can separate vocals from a mix without actually training a neural network. Just wanted to see if the core idea works before spending days training a U-Net.

## The Idea

Basically: if I can manually extract a vocal by creating a unique enough "fingerprint" of what a vocal looks like across 18 different views of the spectrogram, then a U-Net should be able to learn to do the same thing automatically (and way faster).

This is the manual proof-of-concept version. Takes a few minutes to process a song, gets maybe 70-80% quality. But if this works at all, then training a U-Net on thousands of examples should get us to 95%+ and do it in 10ms instead of minutes.

## How It Works

1. Take the isolated vocal and the full mix
2. Create 18 different "slices" of each spectrogram (different conv filters basically - horizontal lines, vertical lines, diagonals, edges, etc)
3. Run each slice through 4 encoder layers to compress it down to a bottleneck
4. At the bottleneck, extract a ton of metrics - 400-point frequency profile plus 25 other features
5. Now you have ~765,000 unique measurements describing the vocal
6. Try to adjust the mixture's spectrogram until its fingerprint matches the vocal's
7. Apply those adjustments, convert back to audio

If the fingerprint is unique enough, the only way the mixture can match it is to actually sound like the vocal.

## Setup

Install dependencies:
```bash
uv pip install numpy librosa soundfile scipy matplotlib
```

## Adding Your Audio Files

### Option 1: Quick Test (100 windows, ~4.7 seconds)

1. Put your files in `process/100-window/`:
   ```
   process/100-window/
   ├── yourfile_100-full.wav    (full mixture with everything)
   └── yourfile_100-stem.wav    (isolated vocal/acapella)
   ```

2. Run preparation:
   ```bash
   python prepare_audio_files.py
   ```

3. Run sanity check:
   ```bash
   python sanity_check_complete.py
   ```

### Option 2: Full Song (entire audio, no time limit)

1. Put your files in `process/no-limit/`:
   ```
   process/no-limit/
   ├── yourfile_nl-full.wav     (full mixture with everything)
   └── yourfile_nl-stem.wav     (isolated vocal/acapella)
   ```

2. Run preparation:
   ```bash
   python prepare_audio_files.py
   ```

3. Run sanity check:
   ```bash
   python sanity_check_full_length.py
   ```

### File Naming Rules

**For 100-window version:**
- Full mix must end with `_100-full.wav`
- Isolated vocal must end with `_100-stem.wav`
- Example: `song_100-full.wav`, `song_100-stem.wav`

**For no-limit version:**
- Full mix must end with `_nl-full.wav`
- Isolated vocal must end with `_nl-stem.wav`
- Example: `track_nl-full.wav`, `track_nl-stem.wav`

The script will find them automatically based on these suffixes.

## What You Get

After running complete version:

```
output/
├── extracted_vocal.wav              # the separated vocal
├── 1_original_mixture.wav           # copy of input for comparison
├── 2_target_vocal.wav               # copy of target for comparison
├── optimization_loss.png            # shows the learning curve
└── spectrograms.png                 # visual comparison
```

Full-length version puts everything in `output_full/` instead.

## The 18 Slices

Different conv2d filters to catch different patterns:

- slice_0: raw magnitude (baseline)
- slice_1-8: basic patterns (horizontal/vertical/diagonal/blob/harmonic/edge detection)
- slice_9-15: oriented edge detectors at different angles (22.5°, 45°, 67.5°, etc)
- slice_16-18: laplacian and pooled versions

Vocals have specific patterns - sustained frequencies, harmonic stacks, mid-range energy, formants. These 18 views capture that.

## Expected Results

Not expecting perfection. This is a proof of concept.

**Should work:**
- Vocal is audible and mostly clear
- Drums are way quieter
- Bass is way quieter
- You can tell it's the vocal

**Won't work:**
- Studio-quality isolation
- Zero artifacts (there will be some clicking/phasing)
- Complete removal of all other instruments
- Perfect timbre preservation

If we hit 70-80% quality, that proves the concept. Then U-Net training should get us to 95%+.

## The Fingerprint

Each time window (every ~46ms) gets:
- 400-point frequency profile (energy at 0Hz, 50Hz, 100Hz, ... 20kHz)
- 6 band energies (bass, low-mid, mid, high-mid, presence, high)
- 6 spectral shape metrics (centroid, spread, rolloff, flatness, slope, crest)
- 5 harmonic features (fundamental freq, num harmonics, spacing, strength, etc)
- 4 formant measurements
- 4 dynamics measurements

= 425 metrics per window

With 18 slices and ~100 windows per slice, that's around 765,000 data points describing the vocal. Should be unique enough that only the actual vocal matches.

## Files

- `sanity_check.py` - phases 1-2 only (analysis, no output audio)
- `sanity_check_complete.py` - full pipeline, 4.7 second snippet
- `sanity_check_full_length.py` - full pipeline, entire song
- `prepare_audio_files.py` - helper to convert your audio to the right format
- `test_setup.py` - verify everything is installed correctly
- `requirements.txt` - dependencies

## Workflow Summary

```
1. Add your audio files to process/100-window/ or process/no-limit/
   (name them correctly: *_100-full.wav + *_100-stem.wav OR *_nl-full.wav + *_nl-stem.wav)

2. Run: python prepare_audio_files.py
   (converts and moves files to rtg/100-window/ or rtg/no-limit/)

3. Run: python sanity_check_complete.py (for 100-window)
   OR:  python sanity_check_full_length.py (for no-limit)

4. Check output/ or output_full/ for results
```

## Troubleshooting

**"NO FILES FOUND"**
Make sure your files end with the right suffixes and are in the right directory.

**"FileNotFoundError"**
Run `prepare_audio_files.py` first.

**Takes forever**
Use the 100-window version instead of no-limit. Or reduce iterations in the config.

**Out of memory**
Reduce n_fft from 2048 to 1024 in the config.

**Results sound terrible**
Make sure your files are correct. The "full" file should be the complete mix with everything. The "stem" file should be just the isolated vocal. If they're swapped or wrong, it won't work.

## What This Proves

If this works even a little bit, it means:
1. Multi-scale spectral fingerprinting is a valid approach
2. The encoder-decoder architecture makes sense
3. U-Net should be able to learn this automatically
4. Training a U-Net on this task is worth the effort

If it doesn't work, then either the fingerprint isn't unique enough, or the optimization approach is wrong, or the whole concept is flawed. Better to find out in a few minutes of manual work than after days of training.

## Next Steps

Assuming this works:
1. Figure out which slices were most useful
2. Design a U-Net that mirrors this architecture
3. Gather a dataset of thousands of (vocal, mixture) pairs
4. Train the U-Net
5. Compare trained model to this baseline

The trained model should be way better (95%+ quality) and way faster (10ms vs minutes).

