import argparse
import subprocess
import sys
from pathlib import Path
from audio_utils import load_audio, plot_waveform, show_spectrogram


def main():
    parser = argparse.ArgumentParser(
        description="Visualize audio waveform and spectrogram."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="mixture.wav",
        help="Path to input audio file (.wav, .flac, etc.)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sample rate (Hz). Default: native file rate"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.10,
        help="Duration (seconds) of waveform preview. Default: 0.10 (100 ms)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Starting point (seconds) for waveform preview. Default: 0.0",
    )
    args = parser.parse_args()

    infile = Path(args.input_file)
    if not infile.exists():
        sys.exit(f"ERROR: file not found: {infile}")

    audio, sr = load_audio(str(infile), sr=args.sr)
    plot_waveform(audio, sr, duration_s=args.duration, start_s=args.start)
    show_spectrogram(audio, sr)

    try:
        subprocess.run(["feh", "--title", "Waveform", "figures/waveform.png"], check=True)
        subprocess.run(["feh", "--title", "Spectrogram", "figures/spectrogram.png"], check=True)
    except FileNotFoundError:
        print("NOTE: program requires 'feh' to display images")


if __name__ == "__main__":
    main()
