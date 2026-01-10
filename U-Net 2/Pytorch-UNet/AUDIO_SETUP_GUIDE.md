# 🎵 Audio File Setup Guide

Quick guide for setting up your audio files for the vocal separation sanity check.

## What You Need

Two versions of the same song:
1. **Acapella/Stem version** - Just the vocals (or whatever stem you want to separate)
2. **Full mix version** - Complete song with all instruments

Both must be:
- The same song
- Perfectly time-aligned (critical!)
- Any format (MP3, WAV, FLAC, etc - the script converts)

## Finding Audio

**Where to get acapellas:**
- Soulseek (https://www.slsknet.org) - great for finding acapellas
- YouTube rips (use youtube-dl or similar)
- Official acapella releases
- Rap songs often have acapella versions available

## Aligning Your Tracks (CRITICAL STEP)

Your tracks MUST start at the exact same time. Use Audacity (free) or any DAW:

### In Audacity:
1. **Import both tracks** - File → Import → Audio
2. **Zoom way in** - Ctrl/Cmd + 1 (keep zooming)
3. **Find a sharp transient** - Look for a snare hit, vocal consonant, anything both tracks share
4. **Align them** - Click and drag one track horizontally until the transients line up perfectly
   - Use the Time Shift Tool (F5)
   - Don't stretch or speed-change, just slide horizontally
5. **Select the aligned section** - Click and drag to set In/Out markers
   - For 100-window version: Select 4.7 seconds
   - For no-limit version: Select whatever length you want
6. **Export** - File → Export → Export Selected Audio
   - Export both tracks separately
   - Save as WAV

### Verify alignment:
1. Re-import both exported files
2. Select one track → Effect → Invert
3. Play both together
4. Should mostly cancel out (near silence) if perfectly aligned

## File Naming

**For quick test (100-window, 4.7 seconds):**
- `yourname_100-full.wav` - Full mix
- `yourname_100-stem.wav` - Acapella/stem

**For full song (no-limit):**
- `yourname_nl-full.wav` - Full mix
- `yourname_nl-stem.wav` - Acapella/stem

Examples:
```
intergalactic_100-full.wav
intergalactic_100-stem.wav

mysong_nl-full.wav
mysong_nl-stem.wav
```

The name before the suffix doesn't matter, just needs to match for both files.

## Where to Put Files

Place your named files in:
- `vocal_separation_sanity_check/process/100-window/` for quick test
- `vocal_separation_sanity_check/process/no-limit/` for full song

## Running the Preparation Script

```bash
cd vocal_separation_sanity_check
python prepare_audio_files.py
```

This will:
- Find your files based on the naming
- Convert to mono, 22050 Hz
- Trim to 4.7 seconds (for 100-window) or keep full length (for no-limit)
- Normalize volume
- Save to `rtg/100-window/` or `rtg/no-limit/` as ready-to-go files

## Running the Sanity Check

After preparation:

```bash
# For 100-window version (~3 min runtime)
python sanity_check_complete.py

# For no-limit version (longer runtime)
python sanity_check_full_length.py
```

Check output in `output/` or `output_full/`.

## Troubleshooting

**"NO FILES FOUND"**
- Check your files are in the right directory
- Check file names end with correct suffixes
- Run `ls process/100-window/` or `ls process/no-limit/` to see what's there

**"Files don't match"**
- Make sure both files are the same duration
- Check they're properly aligned

**"Bad audio quality"**
- Source files might be too low quality
- Try finding better quality versions (320kbps MP3 or lossless)

**"Takes too long"**
- Use 100-window version for quick tests
- Full song processing can take 30+ minutes

## Quick Workflow Summary

1. Find acapella + full mix of same song
2. Align them perfectly in Audacity
3. Export aligned sections
4. Name with correct suffixes (_100-full/_100-stem or _nl-full/_nl-stem)
5. Put in process/100-window/ or process/no-limit/
6. Run `python prepare_audio_files.py`
7. Run sanity check script
8. Check output directory for results
