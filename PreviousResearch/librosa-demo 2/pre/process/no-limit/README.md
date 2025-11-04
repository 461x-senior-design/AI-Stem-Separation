# Place Your Audio Files Here

For the **no-limit version** (full song, any length):

## File Naming

Your files must end with these suffixes:

- `*_nl-full.wav` - Full mixture (song with vocals, drums, bass, everything)
- `*_nl-stem.wav` - Isolated vocal (acapella, just the vocal track)

## Examples

```
song_nl-full.wav
song_nl-stem.wav
```

```
track_nl-full.wav
track_nl-stem.wav
```

```
intergalactic_nl-full.wav
intergalactic_nl-stem.wav
```

The actual filename before the suffix doesn't matter, as long as both files use the same base name.

## Next Steps

1. Add your two files to this directory
2. Run: `python prepare_audio_files.py` (from parent directory)
3. Run: `python sanity_check_full_length.py`
4. Check `output_full/` for results

## What Happens

The prepare script will:
- Find your files based on the `_nl-full.wav` and `_nl-stem.wav` suffixes
- Convert to mono, 22050 Hz
- Keep full duration (no trimming)
- Normalize volume
- Save as `isolated_vocal.wav` and `stereo_mixture.wav` in the parent directory

## Runtime

Processing time scales with song length:
- 30 second song: ~5 minutes
- 3 minute song: ~30 minutes
- 5 minute song: ~50 minutes

The 100-window version is faster for testing.
