# U-Net Audio Source Separation

4-stem audio source separation (drums, bass, vocals, other) using U-Net.

---

## Where Did This Come From?

This codebase was synthesized using the **Ralph Wiggum Loop** method - an iterative AI agent pattern that's been gaining traction in the last few months for complex code generation tasks.

### What is the Ralph Wiggum Loop?

Named after the Simpsons character who stumbles into correct answers through persistence, the Ralph Wiggum Loop is an iterative agent pattern:

```
LOOP:
  1. ASSESS current state (what exists, what's missing, what's broken)
  2. IDENTIFY next highest-priority task
  3. EXECUTE that task completely
  4. VERIFY the output is correct
  5. UPDATE progress tracking
  6. CHECK completion criteria
  7. IF not complete → GOTO 1
  8. IF complete → FINALIZE and report
```

The agent keeps iterating until all completion criteria are met - no partial implementations, no TODOs, no placeholders.

### How This Code Was Generated

We fed an AI agent:
- Our **PRD** (Product Requirements Document)
- Our **Design Doc** with architecture decisions
- All **Q1 curriculum files** (weeks 1-10 of documentation)
- Our **test code** and validation scripts

Then asked it to synthesize:
1. Complete U-Net architecture (`models/unet.py`, `encoder.py`, `decoder.py`)
2. Audio preprocessing pipeline (`data/preprocessing.py`)
3. Training infrastructure (`train.py`, `utils/losses.py`, `utils/config.py`)

The agent iterated through all Q1 materials, extracted the relevant architecture and code patterns, skipped redundant exercises, and produced this production-ready package.

### Verification

All generated code was:
- Syntax validated
- Shape-tested (input/output dimensions verified)
- Gradient flow tested (all parameters receive gradients)
- Overfitting tested (model can learn on small data)

**This is the sum of our Q1 work, processed through a Ralph Wiggum loop, and verified by AI for accuracy.**

---

## Requirements

- Python 3.9+

## Install

```bash
pip install torch torchaudio librosa soundfile numpy
```

## Dataset

Download [MUSDB18-HQ](https://zenodo.org/records/3338373) (~15GB)

Expected structure:
```
musdb18hq/
├── train/
│   ├── song_name/
│   │   ├── mixture.wav
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   ├── vocals.wav
│   │   └── other.wav
│   └── ...
└── test/
    └── ...
```

## Train

```bash
python train.py --data-dir /path/to/musdb18hq --epochs 100
```

## Separate a song

```bash
python train.py --separate song.wav --checkpoint output/checkpoints/best_model.pth
```

## Specs

- Input: 44.1kHz 16-bit stereo
- Output: 4 stems (drums, bass, vocals, other) as stereo WAV
- Model: ~2M parameters
