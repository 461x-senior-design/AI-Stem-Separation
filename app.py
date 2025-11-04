# serve.py

import os
import sys
import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr
import yaml
from huggingface_hub import hf_hub_download

from src.models.unet import UNet

def load_config(config_path):
    """Load YAML config and convert to nested namespace for dot notation access."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        """Recursively convert dict to SimpleNamespace."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    return dict_to_namespace(config_dict)

# 1) Load your config and model once at startup
CFG = load_config("config/default.yaml")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL = UNet(
    in_ch=1,
    num_sources=len(CFG.data.sources) - 1,
    chans=CFG.model.chans,
    num_pool_layers=CFG.model.num_pool_layers
).to(DEVICE)

# Load checkpoint - try local first, fallback to HuggingFace
ckpt_path = Path("checkpoints/unet_best.pt")
if ckpt_path.exists():
    print(f"Loading local checkpoint: {ckpt_path}")
    ckpt_file = str(ckpt_path)
else:
    print("Local checkpoint not found, downloading from HuggingFace...")
    ckpt_file = hf_hub_download(
        repo_id="theadityamittal/music-separator-unet",
        filename="checkpoints/unet_best.pt"
    )
MODEL.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
MODEL.eval()


def separate_file(mix_path, output_dir=None):
    """
    Given a file path to the uploaded mixture WAV, returns
    a dict of { "drums": path, "bass": path, ... } to the separated .wav files.

    Args:
        mix_path: Path to input audio file
        output_dir: Optional directory to save outputs. If None, uses temp files.
    """
    # 1. Load audio & STFT
    print(f"Loading audio from: {mix_path}")
    wav, sr = librosa.load(mix_path, sr=CFG.data.sample_rate, mono=True)
    print(f"Computing STFT...")
    stft = librosa.stft(
        wav, n_fft=CFG.data.n_fft, hop_length=CFG.data.hop_length
    )
    mag, phase = np.abs(stft), np.angle(stft)
    F, T = mag.shape

    # 2. Pad to multiple of segment_length
    SEG = CFG.data.segment_length
    pad = (SEG - (T % SEG)) % SEG
    if pad:
        mag   = np.pad(mag,   ((0,0),(0,pad)), constant_values=0)
        phase = np.pad(phase, ((0,0),(0,pad)), constant_values=0)
    n_seg = mag.shape[1] // SEG

    # 3. Inference in chunks
    print(f"Running inference on {n_seg} segments...")
    preds = []
    with torch.no_grad():
        for i in range(n_seg):
            mseg = mag[:, i*SEG:(i+1)*SEG]
            x = torch.from_numpy(mseg).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
            y = MODEL(x)  # (1, S, F, SEG)
            preds.append(y.squeeze(0).cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_seg} segments")
    pred_mag = np.concatenate(preds, axis=2)[:, :, :T]
    phase    = phase[:, :T]

    # 4. Reconstruct waveforms and write files
    print("Reconstructing audio stems...")
    out_paths = {}
    for idx, src in enumerate(CFG.data.sources[1:]):
        spec = pred_mag[idx] * np.exp(1j * phase)
        est  = librosa.istft(
            spec,
            hop_length=CFG.data.hop_length,
            win_length=CFG.data.n_fft
        )

        # Determine output path
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_name = Path(mix_path).stem
            path = str(output_dir / f"{input_name}_{src}.wav")
        else:
            # Use temp file for web interface
            fd, path = tempfile.mkstemp(suffix=f"_{src}.wav")
            os.close(fd)

        sf.write(path, est, sr)
        out_paths[src] = path
        print(f"  Saved {src}: {path}")

    # return in the order drums, bass, other, vocals
    return [out_paths[src] for src in CFG.data.sources[1:]]


# 5) Build Gradio interface
description = """
## Music Source Separation

Upload a mix `.wav` and get back **drums**, **bass**, **other**, and **vocals** stems separated by a U-Net model.
"""

iface = gr.Interface(
    fn=separate_file,
    inputs=gr.Audio(label="Mixture (.wav)", type="filepath"),
    outputs=[
         gr.Audio(label="Drums",  type="filepath"),
         gr.Audio(label="Bass",   type="filepath"),
         gr.Audio(label="Other",  type="filepath"),
         gr.Audio(label="Vocals", type="filepath"),
     ],
    title="U-Net Music Separator",
    description=description,
    allow_flagging="never",
)

def run_interactive_cli():
    """Interactive CLI mode for audio separation."""
    print("\n" + "="*60)
    print("🎵 Music Source Separator - Interactive CLI Mode")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model loaded: U-Net ({CFG.model.chans} channels, {CFG.model.num_pool_layers} layers)")
    print("="*60 + "\n")

    while True:
        # Get input file path
        audio_path = input("Enter path to audio file (or 'q' to quit): ").strip()

        if audio_path.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break

        # Remove quotes if user pasted path with quotes
        audio_path = audio_path.strip("'\"")
        audio_file = Path(audio_path)

        # Validate input file
        if not audio_file.exists():
            print(f"❌ Error: File not found: {audio_path}\n")
            continue

        if not audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            print(f"⚠️  Warning: {audio_file.suffix} may not be supported. Supported: .wav, .mp3, .flac, .ogg, .m4a\n")

        # Get output directory
        default_output = audio_file.parent / "separated"
        output_input = input(f"Output directory (default: {default_output}): ").strip()
        output_dir = Path(output_input) if output_input else default_output

        print(f"\n🎵 Processing: {audio_file.name}")
        print(f"📁 Output directory: {output_dir}\n")

        try:
            # Run separation
            output_files = separate_file(str(audio_file), output_dir=str(output_dir))

            print(f"\n✅ Separation complete! Generated {len(output_files)} stems:")
            for stem_path in output_files:
                stem_name = Path(stem_path).name
                print(f"   • {stem_name}")

            print()

        except Exception as e:
            print(f"\n❌ Error during separation: {e}\n")

        # Ask if user wants to continue
        cont = input("Separate another file? (y/n): ").strip().lower()
        if cont not in ['y', 'yes']:
            print("Goodbye!")
            break
        print()


def run_web_interface():
    """Launch Gradio web interface."""
    print(f"\n🌐 Launching web interface on http://localhost:7860")
    print(f"Device: {DEVICE}\n")
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Source Separator - Split audio into drums, bass, other, vocals")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive CLI mode"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web interface (default if no flags specified)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input audio file path (non-interactive mode)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output directory (non-interactive mode)"
    )

    args = parser.parse_args()

    # Non-interactive mode: single file processing
    if args.input:
        if not Path(args.input).exists():
            print(f"❌ Error: File not found: {args.input}")
            sys.exit(1)

        output_dir = args.output or Path(args.input).parent / "separated"
        print(f"\n🎵 Processing: {args.input}")
        print(f"📁 Output directory: {output_dir}\n")

        try:
            output_files = separate_file(args.input, output_dir=str(output_dir))
            print(f"\n✅ Separation complete! Generated {len(output_files)} stems:")
            for stem_path in output_files:
                print(f"   • {stem_path}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

    # Interactive CLI mode
    elif args.interactive:
        run_interactive_cli()

    # Web interface mode (default)
    else:
        run_web_interface()
