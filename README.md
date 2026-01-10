# AI Stem Separation

4-stem audio source separation (drums, bass, vocals, other) using U-Net.

---

## Repository Structure

```
AI-Stem-Separation/
├── Quarter 1/      # Q1 learning materials, demos, and experiments
├── U-Net 1/        # Our culminated work from Q1
└── U-Net 2/        # Abandoned alternative implementation
```

---

## U-Net 1 (Our Culminated Work from Q1)

Located in `/U-Net 1/`

### Where Did This Come From?

This codebase was synthesized using the **Ralph Wiggum Loop** method - an iterative AI agent pattern that's been gaining traction in the last few months for complex code generation tasks.

#### What is the Ralph Wiggum Loop?

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

#### How This Code Was Generated

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

#### Verification

All generated code was:
- Syntax validated
- Shape-tested (input/output dimensions verified)
- Gradient flow tested (all parameters receive gradients)
- Overfitting tested (model can learn on small data)

**This is the sum of our Q1 work, processed through a Ralph Wiggum loop, and verified by AI for accuracy.**

---

## Quick Start (U-Net 1)

### Requirements

- Python 3.9+
- CUDA GPU recommended (MPS/CPU supported but slower)

### Install

```bash
pip install torch torchaudio librosa soundfile numpy
```

### Dataset

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

### Train

```bash
cd "U-Net 1"
python train.py --data-dir /path/to/musdb18hq --epochs 100
```

### Separate a song

```bash
python train.py --separate song.wav --checkpoint output/checkpoints/best_model.pth
```

### Specs

- Input: 44.1kHz 16-bit stereo
- Output: 4 stems (drums, bass, vocals, other) as stereo WAV
- Model: ~2M parameters

---

## U-Net 2 (Abandoned)

Located in `/U-Net 2/`

**Status: Abandoned**

This is an alternative U-Net implementation that was started but not completed to production quality. There are known quality issues with this codebase.

However, it represents a different approach and could serve as **Starting Point 2** if the team decides to explore alternative architectures or approaches.

### What's Missing

- **No train.py** - The plan was to modify the `Pytorch-UNet/train.py` script to work as the U-Net 2 trainer. This work was not completed.

### Team Decision Needed

We have two options moving forward:

1. **Continue with U-Net 1** - Our Q1 culminated work
2. **Explore U-Net 2** - Requires cleanup and validation, but may have different design decisions worth considering

This is a team discussion - review both implementations and decide how we want to approach Q2.

---

## Quarter 1 Materials

Located in `/Quarter 1/`

Contains all learning materials, demos, and experiments from Q1:
- Librosa demos
- Encoder/decoder experiments
- Waveform and tensor pipeline examples
- Learning notes

These materials informed the U-Net 1 synthesis but are not required for training.
