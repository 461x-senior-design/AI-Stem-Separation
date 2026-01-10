# Q1-OUT Manifest

## Quarter 1 Deliverables - U-Net Architecture for Audio Source Separation

**Status:** COMPLETE - Ready for Quarter 2 Training
**Date:** 2026-01-10
**Validation:** All tests passing (6/6)

---

## Summary

This package contains a complete, production-ready U-Net implementation for audio source separation, built from the Quarter 1 curriculum documentation. The architecture processes stereo spectrograms and outputs separated audio sources.

### Key Specifications
- **Input:** Stereo spectrogram tensors (batch, 2, freq, time)
- **Output:** Separated stereo spectrogram (same shape as input)
- **Architecture:** 4-level encoder-decoder with skip connections
- **Parameters:** 1,943,922 trainable parameters
- **Model Size:** 7.42 MB

---

## File Structure

```
Q1-OUT/
|-- MANIFEST.md           # This file - lists all deliverables
|-- SKIPPED.md            # Documents skipped exercises and why
|-- requirements.txt      # Python dependencies
|-- run_validation.py     # Main validation entry point
|
|-- models/
|   |-- __init__.py       # Package exports
|   |-- encoder.py        # EncoderBlock implementation
|   |-- decoder.py        # DecoderBlock implementation
|   |-- unet.py           # Complete U-Net architecture
|
|-- data/
|   |-- __init__.py       # Package exports
|   |-- preprocessing.py  # Audio-to-spectrogram pipeline
|   |-- synthetic.py      # Synthetic audio generator for testing
|
|-- tests/
|   |-- __init__.py       # Package exports
|   |-- test_shapes.py    # Shape validation tests
|   |-- test_gradients.py # Gradient flow tests
|   |-- test_overfitting.py # Learning capability tests
|   |-- run_all_tests.py  # Complete test suite
|
|-- utils/
    |-- __init__.py       # Package exports
    |-- config.py         # Hyperparameter configuration
    |-- losses.py         # Loss functions (L1, MSE, Combined)
```

---

## Deliverables by Component

### 1. Models (Core Architecture)

#### `models/encoder.py`
- **Class:** `EncoderBlock`
- **Purpose:** Downsampling path - extract features, halve spatial dimensions
- **Features:**
  - Two conv layers with BatchNorm and LeakyReLU
  - MaxPool2d for 2x downsampling
  - Returns (encoded, skip) tuple for skip connections
- **Validated:** Shape tests, gradient flow

#### `models/decoder.py`
- **Class:** `DecoderBlock`
- **Purpose:** Upsampling path - upsample and integrate skip connections
- **Features:**
  - ConvTranspose2d for 2x upsampling
  - Skip connection concatenation
  - Two conv layers with BatchNorm and ReLU
  - Optional dropout for regularization
- **Validated:** Shape tests, gradient flow

#### `models/unet.py`
- **Class:** `UNet`
- **Purpose:** Complete encoder-decoder architecture
- **Features:**
  - 4 encoder blocks (2->16->32->64->128 channels)
  - Bottleneck layer (128->256 channels)
  - 4 decoder blocks (256->128->64->32->16 channels)
  - Skip connections wiring encoder to decoder
  - 1x1 final convolution for output projection
- **Validated:** All tests passing

### 2. Data Utilities

#### `data/preprocessing.py`
- **Class:** `AudioPreprocessor`
- **Purpose:** Convert audio files to spectrogram tensors
- **Features:**
  - STFT computation (configurable n_fft, hop_length)
  - Magnitude/phase separation
  - Multiple normalization methods (log, minmax, standardize)
  - Save/load functionality
  - Audio reconstruction from magnitude+phase
- **Dependencies:** librosa, soundfile (optional for Q1)

#### `data/synthetic.py`
- **Class:** `SyntheticAudioGenerator`
- **Purpose:** Generate test audio for validation
- **Features:**
  - Sine wave generation at specific frequencies
  - White/pink noise generation
  - Mixture creation with isolated stems
  - Spectrogram pair generation for training
- **Use Case:** Q1 validation without real datasets

### 3. Test Suite

#### `tests/test_shapes.py`
- Shape validation for encoder, decoder, full U-Net
- Tests various input sizes (128x128 to 512x512)
- Tests batch sizes 1-16
- Tests mono and stereo inputs

#### `tests/test_gradients.py`
- Verifies all parameters receive gradients
- Checks for vanishing gradients
- Checks for exploding gradients
- Validates skip connection gradient flow

#### `tests/test_overfitting.py`
- Single-example overfitting test
- Learning rate sensitivity analysis
- Proves model can learn from data
- Target: >50% loss reduction in 100 iterations

#### `tests/run_all_tests.py`
- Complete validation suite
- Runs all tests in sequence
- Generates pass/fail summary
- Validates Q2 readiness

### 4. Utilities

#### `utils/config.py`
- Centralized configuration
- Model, audio, training hyperparameters
- Device selection (CUDA/MPS/CPU)

#### `utils/losses.py`
- **L1Loss:** Mean absolute error
- **MSELoss:** Mean squared error
- **CombinedLoss:** Weighted L1 + MSE
- **SpectralConvergenceLoss:** Frequency domain loss

---

## How to Run

### Quick Validation
```bash
cd Q1-OUT
python run_validation.py --quick
```

### Full Validation Suite
```bash
cd Q1-OUT
python run_validation.py
```

### Test Individual Components
```bash
# Test U-Net only
python -m models.unet

# Test encoder only
python -m models.encoder

# Run shape tests
python -m tests.test_shapes

# Run overfitting test
python -m tests.test_overfitting
```

### Use in Code
```python
from models import UNet
from utils import get_config, L1Loss

# Create model
model = UNet(input_channels=2, output_channels=2, initial_filters=16)

# Forward pass
x = torch.randn(4, 2, 256, 256)  # Batch of spectrograms
output = model(x)  # Same shape as input

# Training setup
criterion = L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

---

## Validation Results

### Test Suite Output (2026-01-10)
```
[PASSED] 6 tests
   [OK] Shape validation - All input sizes work correctly
   [OK] Gradient flow - All params receive gradients (1,943,922 total)
   [OK] Overfitting test - Loss reduced 58.3%
   [OK] Audio quality - Mean=0.0860, Std=0.1528
   [OK] Save/load - Model saves and loads correctly
   [OK] Parameter count - 1,943,922 parameters (7.42 MB)

ALL TESTS PASSED!
Ready for Quarter 2 training!
```

---

## Architecture Decisions

### 1. Filter Progression
- **Decision:** 16 -> 32 -> 64 -> 128 -> 256 (bottleneck)
- **Rationale:** Standard doubling pattern, manageable memory for CPU testing
- **Alternative:** Could start at 32 for more capacity

### 2. Activation Functions
- **Encoder:** LeakyReLU(0.2) - prevents dying ReLU problem
- **Decoder:** ReLU - standard choice for reconstruction
- **Rationale:** Follows dcyoung tutorial and Spleeter architecture

### 3. Normalization
- **Choice:** BatchNorm2d with eps=1e-3, momentum=0.01
- **Rationale:** Matches Spleeter/Keras defaults

### 4. Downsampling
- **Choice:** MaxPool2d(2, 2)
- **Alternative:** Strided convolution available via `use_pooling=False`

### 5. Upsampling
- **Choice:** ConvTranspose2d(kernel=2, stride=2)
- **Rationale:** Clean 2x upsampling, learnable

---

## Dependencies

### Required
- `torch>=2.0.0` - Core deep learning framework

### Optional (for audio processing)
- `librosa>=0.10.0` - Audio loading and STFT
- `soundfile>=0.12.0` - Audio file I/O
- `numpy>=1.24.0` - Numerical operations

### Development
- `pytest>=7.0.0` - Testing framework

---

## What's Ready for Quarter 2

1. **Complete U-Net architecture** - Tested and validated
2. **Loss functions** - L1, MSE, Combined losses implemented
3. **Preprocessing pipeline** - Ready for real audio (needs librosa)
4. **Synthetic data generator** - For continued testing
5. **Test suite** - For regression testing during Q2

### Q2 Next Steps
1. Acquire real training data (MUSDB18, internal stems)
2. Implement DataLoader for batched training
3. Create training loop with learning rate scheduling
4. Add evaluation metrics (SDR, SIR, SAR)
5. Train and evaluate on real music

---

## Known Limitations

1. **Depth is hardcoded to 4** - Flexible depth requires architectural changes
2. **No attention mechanism** - Standard U-Net without attention gates
3. **Single stem output** - Outputs one separated source per forward pass
4. **Untrained weights** - Random initialization, not pretrained

---

## Source Documentation

Built from Quarter 1 materials:
- `/quarter-1/weeks-06-07-encoder-decoder/` - Encoder/decoder blocks
- `/quarter-1/weeks-08-09-full-unet/` - Complete U-Net architecture
- `/quarter-1/week-10-testing/` - Testing strategies
- `/quarter-1/weeks-03-05-audio-fundamentals/` - Audio preprocessing

Primary reference: dcyoung U-Net PyTorch implementation

---

**Generated by Q1 U-Net Builder**
**Ready for Quarter 2 Training**
