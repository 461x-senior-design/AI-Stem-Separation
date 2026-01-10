#!/usr/bin/env python3
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np

def stereo_to_mono(x):
    if x.ndim == 1: return x
    return x.mean(axis=1)

def safe_read(path):
    data, sr = sf.read(path, always_2d=False)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    return data, sr

def write_wav(path, data, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.as_posix(), data, sr, subtype="PCM_16")

def slice_segments(total_len, sr, segment_sec, stride_sec):
    seg = int(round(segment_sec * sr))
    hop = int(round(stride_sec * sr))
    if total_len < seg: return []
    return [(s, s+seg) for s in range(0, total_len - seg + 1, hop)]

def main():
    ap = argparse.ArgumentParser(description="Prep micro-dataset from a single MUSDB track folder (no musdb pkg).")
    ap.add_argument("--track_dir", required=True, help="Folder like 'AvaLuna - Waterduct' containing stems or mixture.")
    ap.add_argument("--out", required=True, help="Output dataset dir (will create train/val/{mix,vocals}).")
    ap.add_argument("--segment", type=float, default=6.0, help="Segment length (s).")
    ap.add_argument("--stride", type=float, default=6.0, help="Stride (s). Use <segment for overlap.")
    ap.add_argument("--val_last_n", type=int, default=2, help="How many tail segments go to val.")
    args = ap.parse_args()

    track_dir = Path(args.track_dir)
    if not track_dir.exists():
        raise SystemExit(f"Track directory does not exist: {track_dir}")

    vocals_path = track_dir / "vocals.wav"
    mix_path = track_dir / "mixture.wav"
    if not vocals_path.exists():
        raise SystemExit(f"Could not find vocals.wav in {track_dir}")

    voc, sr_v = safe_read(vocals_path)

    if mix_path.exists():
        mix, sr_m = safe_read(mix_path)
        if sr_m != sr_v:
            raise SystemExit(f"Sample rate mismatch: mixture={sr_m}, vocals={sr_v}")
    else:
        parts = []
        for name in ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]:
            p = track_dir / name
            if p.exists():
                x, sr_p = safe_read(p)
                if sr_p != sr_v:
                    raise SystemExit(f"Sample rate mismatch in stem {name}: {sr_p} vs {sr_v}")
                parts.append(x)
        if not parts:
            raise SystemExit("No mixture.wav and no individual stems found.")
        max_len = max(len(x) for x in parts)
        aligned = []
        for x in parts:
            if len(x) < max_len:
                pad = np.zeros((max_len - len(x),) + (() if x.ndim == 1 else ()), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=0)
            aligned.append(x)
        mix = np.sum(aligned, axis=0)

    mix_mono = stereo_to_mono(mix)
    voc_mono = stereo_to_mono(voc)
    L = min(len(mix_mono), len(voc_mono))
    mix_mono = mix_mono[:L]; voc_mono = voc_mono[:L]

    segments = slice_segments(len(mix_mono), sr_v, args.segment, args.stride)
    if not segments:
        raise SystemExit("Track too short for the requested segment length. Try smaller --segment.")

    out = Path(args.out)
    (out / "train" / "mix").mkdir(parents=True, exist_ok=True)
    (out / "train" / "vocals").mkdir(parents=True, exist_ok=True)
    (out / "val" / "mix").mkdir(parents=True, exist_ok=True)
    (out / "val" / "vocals").mkdir(parents=True, exist_ok=True)

    if args.val_last_n > 0 and len(segments) > args.val_last_n:
        train_segs = segments[:-args.val_last_n]
        val_segs = segments[-args.val_last_n:]
    else:
        train_segs = segments[:1]
        val_segs = segments[-1:]

    base = track_dir.name.replace("/", "_")
    for i, (s, e) in enumerate(train_segs):
        write_wav(out / "train" / "mix" / f"{base}_{i:03d}.wav", mix_mono[s:e], sr_v)
        write_wav(out / "train" / "vocals" / f"{base}_{i:03d}.wav", voc_mono[s:e], sr_v)
    for i, (s, e) in enumerate(val_segs):
        write_wav(out / "val" / "mix" / f"{base}_{i:03d}.wav", mix_mono[s:e], sr_v)
        write_wav(out / "val" / "vocals" / f"{base}_{i:03d}.wav", voc_mono[s:e], sr_v)

    print("Wrote dataset to:", out.resolve())
    print("Train segments:", len(train_segs), " Val segments:", len(val_segs))
    print("Sample rate:", sr_v)
    print("Example:", (out / "train" / "mix").glob("*.wav").__next__())

if __name__ == "__main__":
    main()
