# Audio Visualization Demo

Minimal demo for **Progress Report #1**.
Loads an audio file (`.wav` or `.flac`) and generates:

* `figures/waveform.png` - short waveform segment with adaptive y-axis scaling
* `figures/spectrogram.png` - magnitude spectrogram (in decibels)

The program then displays each image in sequence using **feh**.

---

## Usage

```bash
python main.py <audio_file> [--sr SR] [--duration SECS] [--start SECS]
```

**Syntax**

```bash
python main.py <.wav or .flac filename> --<optional parameters>
```

**Parameters**

* `--start`                 Start time (in seconds) for sampling — default `0.0`
* `--duration`           Sampling length (in seconds) — default `0.10` ≈ 1/10 second
* `--sr`                   Sample rate (Hz) — examples `22050`, `44100`, `48000`

  * Default `None` = use the file’s native rate
  * For quick demos, `--sr 22050` runs faster while preserving clarity

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y feh ffmpeg
```

---

## Notes

* `soundfile` and `ffmpeg` enable FLAC support through `librosa`
* Output images are written to `figures/` and opened automatically with `feh`
* Adaptive amplitude scaling keeps quiet and loud sections equally visible
* This demo forms the **Preprocessing and Visualization** foundation for future work in the AI Stem Separation project
* When run, the script opens **two images sequentially**:
  1. The waveform visualization opens first.  
  2. Once that window is closed, the spectrogram automatically opens next.  


---

## Example Run

```bash
python main.py mixture.wav --sr 22050 --start 30 --duration 0.10
```

This example loads `mixture.wav`, samples a 0.10 second segment starting at 30 seconds, and resamples audio to 22.05 kHz for quick visualization.
