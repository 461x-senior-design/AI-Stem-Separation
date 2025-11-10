import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import zoom
import soundfile as sf
import IPython.diisplay as ipd

print("="*70)
print("Batch Normalization Demo")
print("="*70)

# Configuration ADAPTED FROM CAMERON'S PROTO-UNET DEMO
CONFIG = {
    'vocal_path': './input_files/acapella.mp3',
    'output_dir': './output_batch',
    'figures_dir': './figures',
    'sr': 22050,
    'duration': 7.0,
    'n_fft': 2048,
    'hop_length': 512,
}

# Create output directories
Path(CONFIG['output_dir']).mkdir(exist_ok=True)
Path(CONFIG['figures_dir']).mkdir(exist_ok=True)

print("✓ Configuration loaded")

