import os
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def load_audio(path, sr=22050):
    """
    Load audio (wav/flac/etc.) at target sample rate.
    """
    audio, sr = librosa.load(path, sr=sr, mono=True)
    print(f"Loaded {path} - {len(audio)/sr:.2f}s, {sr}Hz")
    return audio, sr


def plot_waveform(
    audio,
    sr,
    duration_s=0.10,
    start_s=0.0,
    out_prefix="figures/waveform",
):
    """
    Save a waveform PNG of the selected segment.

    out_prefix: base path WITHOUT extension, e.g. "figures/original"
                -> saves to "figures/original.png"
    """
    start = int(sr * start_s)
    end = start + int(sr * duration_s)
    end = min(end, len(audio))

    segment = audio[start:end]
    plt.figure(figsize=(8, 3))
    plt.plot(segment)
    plt.title(
        f"Waveform - {start_s:.1f}s to {start_s + duration_s:.1f}s (≈ {duration_s*1000:.0f} ms)"
    )
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # adaptive vertical scaling (same logic as before)
    peak = np.max(np.abs(segment)) if len(segment) > 0 else 0
    if 0 < peak < 0.001:
        plt.ylim(-0.01, 0.01)
    elif peak > 1:
        plt.ylim(-1, 1)
    else:
        plt.ylim(-peak * 1.1, peak * 1.1)

    plt.tight_layout()
    out_path = f"{out_prefix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved waveform to {out_path}")


def show_spectrogram(
    audio,
    sr,
    out_prefix="figures/spectrogram",
):
    """
    Save a magnitude spectrogram in dB.

    out_prefix: base path WITHOUT extension, e.g. "figures/original_spec"
                -> saves to "figures/original_spec.png"
    """
    S = np.abs(librosa.stft(audio))
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        sr=sr,
        x_axis="time",
        y_axis="hz",
    )
    plt.title("Spectrogram (dB)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    out_path = f"{out_prefix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved spectrogram to {out_path}")
