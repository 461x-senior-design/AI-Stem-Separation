# Place Your Audio Files Here

For the **100-window version** (quick test, ~4.7 seconds):

## File Naming

Your files must end with these suffixes:

- `*_100-full.wav` - Full mixture (song with vocals, drums, bass, everything)
- `*_100-stem.wav` - Isolated vocal (acapella, just the vocal track)

## Examples

```
song_100-full.wav
song_100-stem.wav
```

```
test_100-full.wav
test_100-stem.wav
```

```
intergalactic_100-full.wav
intergalactic_100-stem.wav
```

The actual filename before the suffix doesn't matter, as long as both files use the same base name.

## Next Steps

1. Add your two files to this directory
2. Run: `python prepare_audio_files.py` (from parent directory)
3. Run: `python sanity_check_complete.py`
4. Check `output/` for results

## What Happens

The prepare script will:
- Find your files based on the `_100-full.wav` and `_100-stem.wav` suffixes
- Convert to mono, 22050 Hz
- Trim to exactly 4.7 seconds (ensures 100 windows)
- Normalize volume
- Save as `isolated_vocal.wav` and `stereo_mixture.wav` in the parent directory
