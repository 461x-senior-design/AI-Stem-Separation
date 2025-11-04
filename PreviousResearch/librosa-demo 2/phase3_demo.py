import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import os

# ============================================
# VISUAL AND CONFIGURATION SETUP
# ============================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

C = Colors

# --- CONFIGURATION ---
# This dictionary holds all file paths and settings.
CONFIG = {
    'mixture_path': 'output/1_original_mixture.wav',
    'vocal_path': 'output/2_target_vocal.wav',
    'output_dir': 'output_phase3',
    'sr': 22050,
    'n_fft': 2048,
    'hop_length': 1024,
    'win_length': 2048,
}

# Create output directories if they don't exist
OUTPUT_DIR = Path(CONFIG['output_dir'])
OUTPUT_DIR.mkdir(exist_ok=True)
Path('output').mkdir(exist_ok=True) # For robustness

# ============================================
# HELPER FUNCTIONS FOR VISUALS
# ============================================

def print_header(title):
    """Prints a styled header."""
    print("\n" + "="*70)
    print(f"{C.BOLD}{C.BLUE}// {title.upper()}{C.ENDC}")
    print("="*70)
    time.sleep(0.5)

def print_progress(text, duration=1.0):
    """Prints text with a loading animation."""
    print(f"  {C.YELLOW}> {text}{C.ENDC}", end='', flush=True)
    for _ in range(3):
        time.sleep(duration / 3)
        print(f"{C.YELLOW}.{C.ENDC}", end='', flush=True)
    print(f"{C.GREEN} ✓{C.ENDC}")
    time.sleep(0.3)

def print_command(command):
    """Prints a simulated shell command."""
    print(f"\n{C.GREEN}$ {command}{C.ENDC}")
    time.sleep(0.8)


# ============================================
# PHASE 3: 4-STEM U-NET SPLITTER DEMO
# ============================================

print_header("Phase 3: The 4-Stem AI Splitter Architecture")
print(f"This demo outlines the advanced {C.BOLD}U-Net model{C.ENDC} designed to separate a")
print("stereo mix into four distinct stems: Vocals, Drums, Bass, and Other.")

# ============================================
# STEP 1: OUR ARCHITECTURE: A TEAM OF EXPERTS
# ============================================

print_header("Step 1: Our Architecture - A Team of Expert Analysts")
print(f"Our U-Net model will produce {C.BOLD}8 channels of audio{C.ENDC} (4 stereo stems).")
print("Think of it as a team of specialists analyzing the music.")
time.sleep(1)

print(f"\n  - {C.BOLD}The ENCODER Team{C.ENDC}:")
print("    Analyzes the mix at multiple levels, from broad rhythm to fine details.")

print(f"\n  - {C.BOLD}The BOTTLENECK Specialist{C.ENDC}:")
print("    Forms a deep, abstract understanding of the song's core components.")

print(f"\n  - {C.BOLD}The DECODER Team{C.ENDC}:")
print("    Reconstructs the 4 stems with high fidelity, guided by the other teams.")


# ============================================
# STEP 2: THE FORWARD PASS & SKIP CONNECTIONS
# ============================================

print_header("Step 2: The Data Flow - How Information Travels")
print("The key is 'skip connections'—messages passed between the teams.")
print("This ensures no detail is lost. Let's trace the data's journey.")
time.sleep(1.5)

print("\n--- U-NET DATA PATH VISUALIZED ---")
ascii_unet = [
    f"{C.CYAN}INPUT: (B, 2, H, W) -------------------------------------------------> FINAL OUTPUT: (B, 8, H, W){C.ENDC}",
    f"{C.CYAN}  |                                                                        ^{C.ENDC}",
    f"{C.CYAN}  ├─[ENC 1] -> H/2 ------> {C.YELLOW}skip 1{C.CYAN} ------> joins ---- [DEC 1]{C.ENDC}",
    f"{C.CYAN}  |    |                                                                   ^{C.ENDC}",
    f"{C.CYAN}  |    ├─[ENC 2] -> H/4 ---> {C.YELLOW}skip 2{C.CYAN} ---> joins ---- [DEC 2]{C.ENDC}",
    f"{C.CYAN}  |    |    |                                                              ^{C.ENDC}",
    f"{C.CYAN}  |    |    ├─[ENC 3] -> H/8 ---> {C.YELLOW}skip 3{C.CYAN} ---> joins ---- [DEC 3]{C.ENDC}",
    f"{C.CYAN}  |    |    |    |                                                         ^{C.ENDC}",
    f"{C.CYAN}  |    |    |    ├─[ENC 4] -> H/16 --> {C.YELLOW}skip 4{C.CYAN} --> joins ---- [DEC 4]{C.ENDC}",
    f"{C.CYAN}  v    v    v    v                                                          |{C.ENDC}",
    f"{C.CYAN}  [ BOTTLENECK ] ----------------------------------------------------------┘{C.ENDC}"
]
for line in ascii_unet:
    print(line)
    time.sleep(0.1)

print_command("encoded_features, skips = model.encoder(audio_mix)")
print_progress("Encoder team analyzes the mix, saving notes (skips)")

print_command("bottleneck_plan = model.bottleneck(encoded_features)")
print_progress("Specialist forms the master separation plan")

print_command("stems = model.decoder(bottleneck_plan, skips)")
print_progress("Decoder team rebuilds stems using the plan and notes IN REVERSE")

print(f"\n{C.BOLD}This precise, memory-driven process is what allows for high-quality separation.{C.ENDC}")


# ============================================
# STEP 3: SIMULATING THE 4-STEM SEPARATION
# ============================================

print_header("Step 3: Simulating the High-Quality 4-Stem Output")
print("Training the real model takes days. We will now simulate its expected")
print("output to demonstrate the target quality.")

def load_audio_wav(path, sr):
    if not Path(path).exists():
        dummy_audio = np.zeros(sr * 5)
        wavfile.write(path, sr, (dummy_audio * 32767).astype(np.int16))
    file_sr, audio = wavfile.read(path)
    if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    return audio, sr

print_command(f"mixture, sr = load_audio('{CONFIG['mixture_path']}')")
mixture_audio, sr = load_audio_wav(CONFIG['mixture_path'], CONFIG['sr'])

print_command(f"vocals_ref, _ = load_audio('{CONFIG['vocal_path']}')")
vocal_audio, _ = load_audio_wav(CONFIG['vocal_path'], CONFIG['sr'])

print_command("sim_vocals, sim_drums, sim_bass, sim_other = simulate_stems(mixture, vocals_ref)")
instrumental_audio = mixture_audio - vocal_audio
sos_bass = signal.butter(10, 400, 'low', fs=sr, output='sos')
sim_bass = signal.sosfilt(sos_bass, instrumental_audio)
sos_drums = signal.butter(10, [400, 4000], 'bandpass', fs=sr, output='sos')
sim_drums = signal.sosfilt(sos_drums, instrumental_audio)
sim_other = instrumental_audio - (sim_bass * 0.7) - (sim_drums * 0.9)
sim_vocals = vocal_audio
print_progress("Simulating stems using digital signal processing")

STEMS_DIR = OUTPUT_DIR / "stems"
STEMS_DIR.mkdir(exist_ok=True)
print_command(f"save_stems(output_dir='{STEMS_DIR}')")
def normalize_and_convert_to_int16(audio):
    normalized = audio / (np.max(np.abs(audio)) + 1e-8)
    return (normalized * 32767).astype(np.int16)
wavfile.write(STEMS_DIR / "vocals.wav", sr, normalize_and_convert_to_int16(sim_vocals))
wavfile.write(STEMS_DIR / "drums.wav", sr, normalize_and_convert_to_int16(sim_drums))
wavfile.write(STEMS_DIR / "bass.wav", sr, normalize_and_convert_to_int16(sim_bass))
wavfile.write(STEMS_DIR / "other.wav", sr, normalize_and_convert_to_int16(sim_other))
print_progress(f"Saving 4 separated stems to '{STEMS_DIR}/'")


# ============================================
# STEP 4: VISUALIZING THE SEPARATED STEMS
# ============================================

print_header("Step 4: Visualizing the Final Separation")
print("A spectrogram allows us to SEE the frequency content of the separated audio.")

def to_spectrogram_scipy(audio):
    f, t, Sxx = signal.stft(audio, fs=sr, nperseg=CONFIG['win_length'], noverlap=CONFIG['win_length']-CONFIG['hop_length'])
    return 20 * np.log10(np.maximum(np.abs(Sxx), 1e-7))

print_command("generate_spectrogram_plot('output_phase3/4_stem_plot.png')")
spec_mix = to_spectrogram_scipy(mixture_audio)
spec_vocals = to_spectrogram_scipy(sim_vocals)
spec_drums = to_spectrogram_scipy(sim_drums)
spec_bass = to_spectrogram_scipy(sim_bass)
spec_other = to_spectrogram_scipy(sim_other)
print_progress("Calculating spectrograms for all audio tracks")

fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True, sharey=True)
plt.style.use('dark_background')
vmax = spec_mix.max(); vmin = vmax - 80
def plot_spec(ax, spec, title):
    img = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14); ax.set_ylabel('Frequency Bins')
    return img
plot_spec(axes[0], spec_mix, 'Original Mixture')
plot_spec(axes[1], spec_vocals, 'Separated Stem: Vocals')
plot_spec(axes[2], spec_drums, 'Separated Stem: Drums')
plot_spec(axes[3], spec_bass, 'Separated Stem: Bass')
img = plot_spec(axes[4], spec_other, 'Separated Stem: Other')
axes[4].set_xlabel('Time Frames')
fig.colorbar(img, ax=axes, format='%+2.0f dB', label='Magnitude (dB)')
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.suptitle('Visualizing the 4-Stem Separation', fontsize=20, weight='bold')
output_image_path = OUTPUT_DIR / '4_stem_separation_spectrograms.png'
plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
plt.close()
print_progress(f"Saving full comparison visualization")


# ============================================
# FINAL SUMMARY
# ============================================

print_header("Phase 3 Demo Complete!")
print(f"All output files can be found in the '{C.BOLD}{CONFIG['output_dir']}/{C.ENDC}' directory.")

print(f"\n{C.BOLD}➡️ WHAT TO DO NOW:{C.ENDC}")
print(f"  {C.BOLD}1. LISTEN:{C.ENDC} Go to the '{C.CYAN}{STEMS_DIR}/{C.ENDC}' folder and play the four .wav files.")
print(f"  {C.BOLD}2. SEE:{C.ENDC}    Open the '{C.CYAN}{output_image_path.name}{C.ENDC}' image to visualize the split.")

print("\nThis concludes the demonstration of our planned architecture.")