# Stemmy

4-stem audio source separation (drums, bass, vocals, other) using U-Net.

[![CI Pipeline](https://github.com/461x-senior-design/AI-Stem-Separation/actions/workflows/ci.yml/badge.svg)](https://github.com/461x-senior-design/AI-Stem-Separation/actions)
---

## 🎯 Project Overview

Stemmy is a deep learning project for separating audio tracks into four stems:
- **Drums** 🥁
- **Bass** 🎸
- **Vocals** 🎤
- **Other** 🎶

The model uses a U-Net architecture trained on the MUSDB18-HQ dataset.

---

## 📁 Repository Structure

```
AI-Stem-Separation/
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline (linting, testing)
├── src/                     # Source code (models, training, inference)
│   └── __init__.py
├── tests/                   # Unit and integration tests
│   ├── __init__.py
│   └── test_imports.py
├── requirements.txt         # Python dependencies
├── ruff.toml               # Code formatting/linting configuration
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python >=3.9, <3.12
- pip (Python package manager)
- Virtual environment tool (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Stem-Separation.git
   cd AI-Stem-Separation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
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

---

## 🧪 Running Tests

We use `pytest` for testing:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 🔍 Code Quality

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


## 🛠️ Development Workflow

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

## 📦 Dependencies

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

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Ensure tests pass: `pytest tests/ -v`
4. Ensure code is formatted: `ruff format .`
5. Commit with clear messages
6. Push and open a pull request to `dev`

---

## 📚 Additional Resources

- [MUSDB18 Dataset](https://sigsep.github.io/datasets/musdb.html) - Official dataset documentation

