# Stemmy

4-stem audio source separation (drums, bass, vocals, other) using U-Net.

[![CI Pipeline](https://github.com/461x-senior-design/AI-Stem-Separation/actions/workflows/ci.yml/badge.svg)](https://github.com/461x-senior-design/AI-Stem-Separation/actions)
---


## Project Overview

Stemmy is a deep learning project for separating audio tracks into four stems:
- **Drums**
- **Bass**
- **Vocals**
- **Other**

The model uses a U-Net architecture trained on the MUSDB18-HQ dataset.

---

## Repository Structure

```
AI-Stem-Separation/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline (linting, testing)
── README.md                     # Project overview + setup + usage
├── pyproject.toml               # Packaging/build config + deps/extras
├── requirements.txt             # Pinned deps (if you use pip -r)
├── ruff.toml                    # Ruff lint/format config
├── src                          # Source package root
│   ├── __init__.py              # Marks src as a package (exports, version, etc.)
│   ├── check_cross_sisdr.py     # Metric/eval script (cross SI-SDR checks)
│   ├── inference.py             # Inference entry/utilities for separation
│   ├── logging_config.py        # Logging setup/helpers
│   ├── models                   # Model architectures
│   │   ├── __init__.py          # Model module exports
│   │   └── unet_2d.py           # 2D U-Net model definition
│   ├── postprocessing           # Post-separation audio processing
│   │   ├── __init__.py          # Postprocessing exports
│   │   ├── audio.py             # Audio I/O + waveform ops (post)
│   │   ├── pipeline.py          # Postprocessing pipeline orchestration
│   │   ├── spectral.py          # STFT/ISTFT + spectral-domain ops (post)
│   │   └── utility              # Postprocessing helpers
│   │       ├── __init__.py      # Utility exports
│   │       ├── constants.py     # Postprocessing constants/config values
│   │       └── output_validator.py # Validates outputs (paths/waveforms/etc.)
│   ├── preprocessing            # Pre-separation audio processing
│   │   ├── __init__.py          # Preprocessing exports
│   │   ├── audio.py             # Audio loading + waveform ops (pre)
│   │   ├── constants.py         # Preprocessing constants/config values
│   │   ├── pipeline.py          # Preprocessing pipeline orchestration
│   │   ├── spectral.py          # STFT/feature prep (pre)
│   │   └── utility              # Preprocessing helpers
│   │       ├── __init__.py      # Utility exports
│   │       ├── audio_file_validator.py # Validates input audio files
│   │       ├── audio_metadata_extractor.py # Reads SR/channels/duration/etc.
│   │       └── constants.py     # Utility constants/config values
│   ├── tool                     # CLI/tools/scripts
│   │   ├── __init__.py          # Tool module exports
│   │   ├── cli.py               # CLI entry point (commands/options)
│   │   ├── fullsong_eval_masked.py # Full-song evaluation script (masked)
│   │   ├── select_best_checkpoint.py # Chooses best checkpoint from runs
│   │   └── separate_one_track.py # Runs separation for a single track
│   ├── train.py                 # Training entry point / trainer runner
│   └── training                 # Training utilities + datasets
│       ├── __init__.py          # Training module exports
│       ├── checkpointing.py     # Save/load checkpoints utilities
│       ├── musdb18hq_dataset.py # MUSDB18-HQ dataset loader
│       └── stft.py              # STFT utilities used during training
└── tests                        # Test suite
    ├── __init__.py              # Tests package marker
    ├── test_imports.py          # Smoke test: imports + basic wiring
    ├── integration              # Integration tests
    │   ├── __init__.py          # Integration tests package marker
    │   └── test_pipeline.py     # End-to-end pipeline integration test(s)
    ├── postprocessor            # Postprocessing unit tests
    │   ├── __init__.py          # Postprocessor tests package marker
    │   ├── test_audio.py        # Tests post audio helpers
    │   ├── test_pipeline.py     # Tests postprocessing pipeline
    │   ├── test_spectral.py     # Tests post spectral utilities
    │   └── utility              # Postprocessing utility tests
    │       ├── __init__.py      # Utility tests package marker
    │       └── test_output_validator.py # Tests output validation
    └── preprocessor             # Preprocessing unit tests
        ├── __init__.py          # Preprocessor tests package marker
        ├── samples              # Test fixtures (audio samples)
        │   └── plinky_key.wav   # Sample audio used in tests
        ├── test_audio.py        # Tests pre audio helpers
        ├── test_ensure_stereo.py # Tests stereo conversion/validation
        ├── test_normalize_waveform.py # Tests normalization logic
        ├── test_pipeline.py     # Tests preprocessing pipeline
        └── test_spectral.py     # Tests pre spectral utilities

```

---

## Getting Started

### Prerequisites

- Python >=3.9, <3.13
- pip (Python package manager)
- Virtual environment tool (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/461x-senior-design/AI-Stem-Separation.git
   cd AI-Stem-Separation
   ```

2. **Create & source a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Activate the virtual environment:**
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (CMD): `.venv\Scripts\activate.bat`
   - Linux/Mac: `source .venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Install project in development mode:**
   ```bash
   pip install -e .
   ```
---

## Running Tests

We use `pytest` for testing:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Code Quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for fast Python linting and formatting.

### Linting
```bash
# Check for errors
ruff check .

# Auto-fix issues
ruff check . --fix
```

### Formatting
```bash
# Check formatting
ruff format --check .

# Apply formatting
ruff format .
```

Configuration is in `ruff.toml`.

---


## Development Workflow

### Branch Strategy
- `main` - Production-ready code
- `dev` - Integration branch for features
- `feature/*` - Individual feature branches

### CI/CD Pipeline
All pushes and pull requests trigger automated:
1. **Linting** with Ruff
2. **Formatting checks**
3. **Unit tests** with pytest

See `.github/workflows/ci.yml` for details.

---

## Dependencies

### Core Libraries
- **PyTorch** (≥2.0.0) - Deep learning framework
- **torchaudio** (≥2.0.0) - Audio processing for PyTorch
- **librosa** (≥0.10.0) - Audio analysis
- **soundfile** (≥0.12.0) - Audio file I/O
- **numpy** (≥1.24.0) - Numerical computing
- **scipy** (≥1.10.0) - Scientific computing

### CLI & UI
- **click** (≥8.1.0) - Command-line interface
- **rich** (≥13.0.0) - Terminal formatting

### Development Tools
- **pytest** (≥7.4.0) - Testing framework
- **pytest-cov** (≥4.1.0) - Coverage reporting
- **ruff** (≥0.1.0) - Linter and formatter

---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Ensure tests pass: `pytest tests/ -v`
4. Ensure code is formatted: `ruff format .`
5. Commit with clear messages
6. Push and open a pull request to `dev`

---

## Additional Resources

- [MUSDB18 Dataset](https://sigsep.github.io/datasets/musdb.html) - Official dataset documentation

