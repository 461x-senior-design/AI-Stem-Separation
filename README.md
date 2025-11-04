# 🎵 Music Source Separator

A PyTorch U-Net model that separates music into **drums**, **bass**, **other**, and **vocals** stems.

## ✨ Features

- 🎯 **Three Usage Modes**: Web UI, Interactive CLI, or Single-file CLI
- ⚡ **Fast**: Processes audio in segments with GPU/MPS acceleration
- 🔧 **Simple**: No complex dependencies - works on Python 3.10+
- 🎸 **Quality**: Trained on MUSDB-18 HQ dataset

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd music-separator-space

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Usage

**Web Interface** (Gradio UI):
```bash
python app.py
# Opens http://localhost:7860
```

**Interactive CLI**:
```bash
python app.py --interactive
# Prompts for audio file path
# Asks where to save outputs
# Loop to process multiple files
```

**Single-file CLI**:
```bash
python app.py -i "path/to/song.wav" -o "output/directory"
# Process one file and exit
```

**Help**:
```bash
python app.py --help
```

## 📊 Model Details

| Property              | Value |
|-----------------------|-------|
| **Model type**        | 2-D U-Net (6.2M params) |
| **Input**             | STFT magnitude (mono, 16 kHz) |
| **Output**            | 4 magnitude masks (drums, bass, other, vocals) |
| **Training data**     | MUSDB-18 HQ (100 train + 50 test songs) |
| **Checkpoint size**   | ~31 MB (FP32) |
| **Architecture**      | Encoder: [C32]→[C64]→[C128]→[C256]→[C512]<br>Decoder: [C256]←[C128]←[C64]←[C32] |

### Performance (MUSDB-18 test set)

| Metric | Mean | Std |
|--------|------|-----|
| **SDR** | -0.14 dB | 1.66 |
| **SIR** | 3.93 dB | 1.86 |
| **SAR** | 4.26 dB | 0.85 |

*Baseline/lightweight model - not state-of-the-art, but fast & portable*

## 🗂️ Repository Structure

```
.
├── app.py              # Main application (CLI + Web)
├── pyproject.toml      # Dependencies & project config
├── config/
│   └── default.yaml    # Audio processing parameters
├── src/
│   └── models/
│       └── unet.py     # U-Net architecture
└── samples/            # Test audio files (16 songs)
```

## 🎛️ Configuration

Edit `config/default.yaml` to customize:

```yaml
device: "mps"           # "cuda", "mps", or "cpu"
data:
  sample_rate: 16000
  n_fft: 1024
  hop_length: 512
  segment_length: 256
```

The model checkpoint is automatically downloaded from HuggingFace Hub on first run.

## 🎯 Examples

**Process with default settings**:
```bash
python app.py -i "samples/Arise - Run Run Run.wav"
# Outputs to: samples/separated/
```

**Custom output directory**:
```bash
python app.py -i "mysong.wav" -o "/tmp/stems"
# Creates: /tmp/stems/mysong_{drums,bass,other,vocals}.wav
```

**Interactive mode**:
```bash
python app.py --interactive
# Enter path to audio file (or 'q' to quit): samples/song.wav
# Output directory (default: samples/separated):
# ... processing ...
# Separate another file? (y/n):
```

## 🔧 Development

**Install with dev dependencies**:
```bash
uv sync --group dev
```

**Run tests**:
```bash
pytest
```

## ⚠️ Limitations

- **Phase reconstruction**: Uses mixture phase → causes some bleeding/artifacts
- **Training data**: Only MUSDB-18 HQ → may struggle with genres not in dataset (classical, EDM)
- **Performance**: Baseline model with negative SDR on some tracks
- **Format support**: Best results with 16 kHz mono WAV files

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Credits

- Model architecture based on classic U-Net
- Trained on [MUSDB-18 HQ dataset](https://sigsep.github.io/datasets/musdb.html)
- Original checkpoint from [theadityamittal/music-separator-unet](https://huggingface.co/theadityamittal/music-separator-unet)

## 🔗 Links

- [Original HuggingFace Model](https://huggingface.co/theadityamittal/music-separator-unet)
- [MUSDB-18 Dataset](https://sigsep.github.io/datasets/musdb.html)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
