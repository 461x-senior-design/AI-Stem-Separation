import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa, librosa.display
from PIL import Image
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.audio_utils import load_audio
from src.models.mock_encoder import MockEncoder

os.makedirs("figures", exist_ok=True)


def save_waveform(audio, sr, path):
    """
    Save waveform plot to the specified path.
    """
    plt.figure(figsize=(8, 3))
    plt.plot(audio)
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def save_spectrogram(audio, sr, path):
    """
    Save magnitude spectrogram (in dB). Works for very short or multi-channel inputs.
    """
    # average the model's output channels 
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # adaptively choose FFT size
    n_fft = 2048 if len(audio) >= 2048 else max(64, 2 ** int(np.floor(np.log2(len(audio)))))

    S = np.abs(librosa.stft(audio, n_fft=n_fft))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(8, 3))
    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
    )
    plt.title(f"Spectrogram (dB) — n_fft={n_fft}")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="PR2: Encoder downsampling demo")
    parser.add_argument(
        "--input",
        type=str,
        default="audio/mixture.wav",
        help="Input audio file (wav or flac)",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument(
        "--start", type=float, default=30.0, help="Start time (seconds)"
    )
    parser.add_argument(
        "--duration", type=float, default=0.10, help="Duration (seconds)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display generated images automatically"
    )
    args = parser.parse_args()

    infile = Path(args.input)
    if not infile.exists():
        sys.exit(f"ERROR: file not found: {infile}")

    # Load /slice audio
    audio, sr = load_audio(str(infile), sr=args.sr)
    start = int(sr * args.start)
    end = start + int(sr * args.duration)
    end = min(end, len(audio))
    segment = audio[start:end]
    print(f"Segment length: {len(segment)} samples ({len(segment)/sr:.4f} sec)")

    save_waveform(segment, sr, "figures/original_waveform_pr2.png")
    save_spectrogram(segment, sr, "figures/original_spectrogram_pr2.png")

    # Convert to tensor
    tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"Tensor shape before model: {tuple(tensor.shape)}")

    # Pass tensor through encoder
    model = MockEncoder()
    with torch.no_grad():
        encoded = model(tensor)
    print(f"Tensor shape after encoder: {tuple(encoded.shape)}")

    # Convert for visualization
    encoded_np = encoded.squeeze().cpu().numpy()
    encoded_np = encoded_np / np.max(np.abs(encoded_np) + 1e-9)

    # Downsampled waveform + spectrogram
    down_wave_path = "figures/downsampled_waveform_pr2.png"
    down_spec_path = "figures/downsampled_spectrogram_pr2.png"

    save_waveform(encoded_np, sr // 16, down_wave_path)  # because 2 pools of /4 each
    save_spectrogram(encoded_np, sr // 16, down_spec_path)

    # Optionally show images
    if args.show:
        print("Displaying images sequentially... (close each window to continue)")
        img_paths = [
            "figures/original_waveform_pr2.png",
            "figures/original_spectrogram_pr2.png",
            down_wave_path,
            down_spec_path,
        ]
        for path in img_paths:
            try:
                # Prefer feh if available for cleaner behavior
                viewer = os.environ.get("IMAGE_VIEWER", "feh")
                # Pillow show uses viewer fallback, but we'll invoke manually to block
                exit_code = os.system(f"{viewer} '{path}'")
                if exit_code != 0:
                    # fallback to Pillow if feh missing
                    with Image.open(path) as img:
                        img.show()
                        input(f"Press Enter after closing {path}...")
            except Exception as e:
                print(f"Could not display {path}: {e}")
    else:
        print("Images saved to figures/. Use --show to open automatically.")


if __name__ == "__main__":
    main()
