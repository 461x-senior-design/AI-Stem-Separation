# Proto-U-Net Demos

Four focused demonstrations exploring the key components of U-Net architecture using real acapella vocals.

## Overview

These demos refine the core concepts from the `librosa-demo 2` audio processor, focusing on **why U-Net works** through hands-on exploration:

1. **Skip Connections** - The defining feature of U-Net
2. **Encoder Architecture** - Hierarchical feature extraction
3. **Decoder Architecture** - Learned reconstruction
4. **BatchNorm** - Training stabilization

Each demo processes your acapella vocal to demonstrate a specific U-Net principle.

## Prerequisites

### Audio Files
The demos expect audio files from the parent `librosa-demo 2` directory:
```
../PreviousResearch/librosa-demo 2/pre/rtg/100-window/isolated_vocal.wav
```

If you don't have this file, you can:
1. Run the preparation script in `librosa-demo 2`:
   ```bash
   cd ../PreviousResearch/librosa-demo\ 2/
   python prepare_audio_files.py
   ```

2. Or modify the `vocal_path` in each demo's `CONFIG` section to point to your own acapella vocal file.

### Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install numpy librosa soundfile matplotlib scipy
```

**Requirements:**
- Python 3.8+
- numpy >= 1.21.0
- librosa >= 0.9.2
- soundfile >= 0.11.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

## Organization

Demos are organized by U-Net component:

```
proto-unet-demos/
├── SkipConnection/
│   └── demo1_skip_connections.py
├── Encoder/
│   └── demo2_encoder_hierarchy.py
├── Decoder/
│   └── demo3_decoder_reconstruction.py
├── BatchNorm/
│   └── demo4_batchnorm_stability.py
├── README.md
├── requirements.txt
└── OVERVIEW.txt
```

## Usage

### Quick Start

Run all demos in sequence:

```bash
python SkipConnection/demo1_skip_connections.py
python Encoder/demo2_encoder_hierarchy.py
python Decoder/demo3_decoder_reconstruction.py
python BatchNorm/demo4_batchnorm_stability.py
```

Each demo will:
- Process the acapella vocal
- Generate visualizations in `figures/`
- Save audio outputs in `output/`
- Print insights to the console

### Demo 1: Skip Connections

**Explores:** Why skip connections are the defining U-Net feature

```bash
python SkipConnection/demo1_skip_connections.py
```

**What it does:**
- Processes vocal through encoder → bottleneck → decoder
- Compares reconstruction WITH and WITHOUT skip connections
- Shows information loss vs. preservation

**Outputs:**
- `output/demo1_no_skip.wav` - Reconstruction without skips (lossy)
- `output/demo1_with_skip.wav` - Reconstruction with skips (preserved)
- `figures/demo1_skip_comparison.png` - Visual comparison

**Key Insight:** Skip connections preserve fine details lost in the bottleneck compression. Listen to the difference in audio quality!

---

### Demo 2: Encoder Hierarchy

**Explores:** Multi-scale feature extraction through progressive downsampling

```bash
python Encoder/demo2_encoder_hierarchy.py
```

**What it does:**
- Builds 4-level encoder pyramid
- Extracts multiple feature types at each level
- Analyzes receptive field growth and compression

**Outputs:**
- `figures/demo2_encoder_pyramid.png` - Progressive downsampling visualization
- `figures/demo2_feature_diversity.png` - Different feature types (edges, blur, etc.)
- `figures/demo2_encoder_statistics.png` - Quantitative analysis

**Key Insight:** Each encoder level captures different scales:
- Level 1: Fine details (edges, onsets)
- Level 2: Local patterns (formants)
- Level 3: Mid-level structures (phonemes)
- Level 4: High-level features (global characteristics)

---

### Demo 3: Decoder Reconstruction

**Explores:** Learned reconstruction through different upsampling strategies

```bash
python Decoder/demo3_decoder_reconstruction.py
```

**What it does:**
- Compresses vocal through encoder to bottleneck
- Reconstructs using 4 different upsampling methods:
  - **Nearest neighbor** (fast, blocky)
  - **Bilinear** (smooth, blurry)
  - **Transposed convolution** (learnable)
  - **Sub-pixel convolution** (sharp)
- Computes quality metrics (PSNR, SSIM, high-frequency preservation)

**Outputs:**
- `output/demo3_nearest.wav` - Nearest neighbor upsampling
- `output/demo3_bilinear.wav` - Bilinear upsampling
- `output/demo3_transposed.wav` - Transposed convolution upsampling
- `output/demo3_subpixel.wav` - Sub-pixel upsampling
- `figures/demo3_decoder_comparison.png` - Spectrogram comparison
- `figures/demo3_metrics_comparison.png` - Quality metrics

**Key Insight:** Upsampling strategy matters! Different methods have different trade-offs for reconstruction quality. Compare the audio files to hear the difference.

---

### Demo 4: BatchNorm Stabilization

**Explores:** Why normalization stabilizes training

```bash
python BatchNorm/demo4_batchnorm_stability.py
```

**What it does:**
- Simulates training dynamics WITH and WITHOUT BatchNorm
- Shows internal covariate shift problem
- Demonstrates gradient flow improvement
- Analyzes activation distributions across layers

**Outputs:**
- `figures/demo4_activation_distributions.png` - Histograms per layer
- `figures/demo4_statistics_comparison.png` - Quantitative metrics
- `figures/demo4_feature_maps.png` - Spatial patterns

**Key Insight:** BatchNorm stabilizes training by:
1. Preventing internal covariate shift
2. Enabling smooth gradient flow
3. Reducing sensitivity to initialization
4. Allowing higher learning rates

---

## Output Structure

```
proto-unet-demos/
├── SkipConnection/
│   └── demo1_skip_connections.py
├── Encoder/
│   └── demo2_encoder_hierarchy.py
├── Decoder/
│   └── demo3_decoder_reconstruction.py
├── BatchNorm/
│   └── demo4_batchnorm_stability.py
├── requirements.txt
├── README.md
├── OVERVIEW.txt
├── figures/               # Visualizations
│   ├── demo1_skip_comparison.png
│   ├── demo2_encoder_pyramid.png
│   ├── demo2_feature_diversity.png
│   ├── demo2_encoder_statistics.png
│   ├── demo3_decoder_comparison.png
│   ├── demo3_metrics_comparison.png
│   ├── demo4_activation_distributions.png
│   ├── demo4_statistics_comparison.png
│   └── demo4_feature_maps.png
└── output/                # Audio files
    ├── demo1_original.wav
    ├── demo1_no_skip.wav
    ├── demo1_with_skip.wav
    ├── demo3_original.wav
    ├── demo3_nearest.wav
    ├── demo3_bilinear.wav
    ├── demo3_transposed.wav
    └── demo3_subpixel.wav
```

## Understanding the Demos

### Relationship to Full U-Net

These demos use **simplified proto-U-Net implementations** to isolate and demonstrate specific concepts:

| Demo | U-Net Component | Simplification |
|------|-----------------|----------------|
| Demo 1 | Skip Connections | Fixed weights, simple averaging |
| Demo 2 | Encoder | Fixed convolution kernels |
| Demo 3 | Decoder | Non-learnable upsampling |
| Demo 4 | BatchNorm | Simulated training dynamics |

**Key Difference:** Real U-Nets have **learnable parameters** trained via backpropagation. These demos use fixed operations to demonstrate the *principles* without requiring training.

### Computational Notes

- **Runtime:** Each demo takes 10-60 seconds
- **Memory:** ~500MB peak usage
- **Audio duration:** 4.7 seconds (configurable in `CONFIG`)
- **Sample rate:** 22050 Hz (configurable)

### Customization

Each demo has a `CONFIG` dictionary at the top:

```python
CONFIG = {
    'vocal_path': '../path/to/your/vocal.wav',
    'output_dir': 'output',
    'figures_dir': 'figures',
    'sr': 22050,          # Sample rate
    'duration': 4.7,      # Audio duration in seconds
    'n_fft': 2048,        # FFT window size
    'hop_length': 512,    # STFT hop length
}
```

Modify these values to experiment with different audio files or processing parameters.

## Educational Goals

### What You'll Learn

1. **Demo 1:** Why U-Net's skip connections are essential for preserving fine details
2. **Demo 2:** How hierarchical feature extraction works in encoder architectures
3. **Demo 3:** Why decoder upsampling strategy affects reconstruction quality
4. **Demo 4:** How BatchNorm stabilizes training and enables deeper networks

### Next Steps

After completing these demos, you'll understand:
- ✓ Why U-Net works for image/audio tasks requiring fine detail
- ✓ How encoder-decoder architectures compress and reconstruct information
- ✓ Trade-offs between different architectural choices
- ✓ Importance of normalization in deep learning

**To build a real U-Net for vocal separation:**
1. Implement learnable convolutional layers (PyTorch/TensorFlow)
2. Add trainable BatchNorm parameters
3. Create training loop with backpropagation
4. Train on paired mixture/vocal datasets

See the parent `librosa-demo 2/audio_processing.py` for a more complete (non-learnable) vocal separation implementation.

## Troubleshooting

### Common Issues

**Problem:** `FileNotFoundError: vocal_path not found`
- **Solution:** Modify `CONFIG['vocal_path']` to point to your audio file

**Problem:** `ImportError: No module named 'librosa'`
- **Solution:** Install dependencies: `pip install -r requirements.txt`

**Problem:** Audio quality is poor
- **Solution:** This is expected! These are simplified demos. Real U-Nets with learned parameters perform much better.

**Problem:** Visualizations don't display
- **Solution:** Check `figures/` directory for saved PNG files

### Performance Tips

- Reduce `duration` in `CONFIG` for faster processing
- Use smaller `n_fft` for lower memory usage
- Run demos individually instead of in sequence

## References

### U-Net Architecture
- Original paper: Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Skip connections inspired by ResNet (He et al., 2016)

### Audio Source Separation
- Wave-U-Net: Stoller et al. (2018)
- Conv-TasNet: Luo & Mesgarani (2019)
- Demucs: Défossez et al. (2019)

### Normalization Techniques
- Batch Normalization: Ioffe & Szegedy (2015)
- Layer Normalization: Ba et al. (2016)
- Instance Normalization: Ulyanov et al. (2016)

## License

Educational demos for understanding U-Net architecture concepts.

## Acknowledgments

Built on top of the `librosa-demo 2` audio processing framework.
