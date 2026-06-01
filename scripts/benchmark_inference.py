#!/usr/bin/env python3
"""Benchmark Stemmy end-to-end separation inference.

The timed region uses stemmy.inference.separate_audio_file(), so results include
preprocessing, model forward, postprocessing, optional validation, and optional
WAV export. Model loading is intentionally outside the timed region by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import resource
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("STEMMY_DISABLE_PROGRESS", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from stemmy.constants import STEMS_4  # noqa: E402
from stemmy.inference import (  # noqa: E402
    InferenceConfig,
    config_from_checkpoint,
    load_pth_model,
    load_torchscript_model,
    separate_audio_file,
)


@dataclass(frozen=True)
class BenchCase:
    label: str
    chunk_frames: int
    overlap_frames: int
    amp: bool


DEFAULT_CASES: tuple[BenchCase, ...] = (
    BenchCase("full", 0, 0, False),
    BenchCase("chunked256x64", 256, 64, False),
    BenchCase("amp_chunked256x64", 256, 64, True),
)
DEFAULT_AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3")


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_case(raw: str) -> BenchCase:
    parts = raw.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--case must use LABEL:CHUNK_FRAMES:OVERLAP_FRAMES:AMP, "
            "for example small:128:32:false"
        )
    label, chunk_raw, overlap_raw, amp_raw = parts
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("case label cannot be empty")
    try:
        chunk = int(chunk_raw)
        overlap = int(overlap_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk and overlap must be integers") from exc
    amp = amp_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if chunk < 0:
        raise argparse.ArgumentTypeError("chunk_frames must be >= 0")
    if overlap < 0:
        raise argparse.ArgumentTypeError("overlap_frames must be >= 0")
    if chunk == 0 and overlap != 0:
        raise argparse.ArgumentTypeError("overlap_frames must be 0 when chunk_frames is 0")
    if chunk > 0 and overlap >= chunk:
        raise argparse.ArgumentTypeError("overlap_frames must be < chunk_frames")
    return BenchCase(label, chunk, overlap, amp)


def _audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise RuntimeError(f"Invalid samplerate for {path}: {info.samplerate}")
    return float(info.frames) / float(info.samplerate)


def _safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem).strip("_")


def _discover_audio_files(songs_dir: Path, extensions: Iterable[str]) -> list[Path]:
    suffixes = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
        if ext.strip()
    }
    if not songs_dir.exists():
        raise SystemExit(f"Songs directory not found: {songs_dir}")
    if not songs_dir.is_dir():
        raise SystemExit(f"Expected songs directory, got file: {songs_dir}")

    songs = sorted(
        path.resolve()
        for path in songs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not songs:
        formatted = ", ".join(sorted(suffixes))
        raise SystemExit(f"No audio files found in {songs_dir} with extensions: {formatted}")
    return songs


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _peak_cuda_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    dev = torch.device(device)
    return float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0)


def _rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _load_model(
    checkpoint: Path | None,
    torchscript: Path | None,
    device: str,
) -> tuple[torch.nn.Module, InferenceConfig, object | None]:
    if checkpoint is not None:
        model, ckpt = load_pth_model(checkpoint, device=device, stems=len(STEMS_4))
        return model, config_from_checkpoint(ckpt), ckpt

    if torchscript is None:
        raise ValueError("checkpoint or torchscript is required")

    model = load_torchscript_model(torchscript, device=device)
    return model, InferenceConfig(), None


def _run_once(
    *,
    audio_path: Path,
    model: torch.nn.Module,
    cfg: InferenceConfig,
    output_dir: Path,
    export_files: bool,
    checkpoint_obj: object | None,
) -> tuple[float, float | None, float]:
    device = str(cfg.device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    gc.collect()
    _sync(device)
    started = time.perf_counter()
    separate_audio_file(
        audio_path=audio_path,
        model=model,
        cfg=cfg,
        output_dir=output_dir,
        export_files=export_files,
        stems=list(STEMS_4),
        checkpoint=checkpoint_obj if isinstance(checkpoint_obj, dict) else None,
    )
    _sync(device)
    elapsed = time.perf_counter() - started
    return elapsed, _peak_cuda_memory_mb(device), _rss_mb()


def _iter_cases(selected: Iterable[str] | None, custom: Iterable[BenchCase] | None) -> list[BenchCase]:
    if custom:
        return list(custom)

    all_cases = {case.label: case for case in DEFAULT_CASES}
    if not selected:
        return list(DEFAULT_CASES)

    cases: list[BenchCase] = []
    for label in selected:
        if label not in all_cases:
            names = ", ".join(all_cases)
            raise ValueError(f"Unknown case '{label}'. Built-in cases: {names}")
        cases.append(all_cases[label])
    return cases


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("-c", "--checkpoint", type=Path, help="Path to a .pth checkpoint.")
    artifact.add_argument("-t", "--torchscript", type=Path, help="Path to a .pt TorchScript model.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("-i", "--input-file", type=Path, help="Benchmark one input audio file.")
    inputs.add_argument(
        "--songs-dir",
        type=Path,
        default=Path("songs"),
        help="Benchmark every supported audio file under this directory. Default: songs.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(ext.lstrip(".") for ext in DEFAULT_AUDIO_EXTENSIONS),
        help="Comma-separated audio extensions for --songs-dir. Default: wav,flac,mp3.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("runs/bench_inference"),
        help="Directory for benchmark CSV and optional separated WAVs.",
    )
    parser.add_argument(
        "-d",
        "--devices",
        default="auto",
        help="Comma-separated devices: auto, cpu, cuda, cuda:N. Default: auto.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per case.")
    parser.add_argument("--warmups", type=int, default=1, help="Untimed warmups per case.")
    parser.add_argument(
        "--cases",
        default="full,chunked256x64,amp_chunked256x64",
        help="Comma-separated built-in cases: full, chunked256x64, amp_chunked256x64.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help="Custom case as LABEL:CHUNK_FRAMES:OVERLAP_FRAMES:AMP. Overrides --cases.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable WAV export to isolate preprocessing/model/postprocessing timing.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable output validation during benchmark runs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Default: OUTPUT_DIR/inference_benchmark.csv.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0")

    if args.input_file is not None:
        audio_paths = [args.input_file.expanduser().resolve()]
        if not audio_paths[0].is_file():
            raise SystemExit(f"Input file not found: {audio_paths[0]}")
    else:
        audio_paths = _discover_audio_files(
            args.songs_dir.expanduser().resolve(),
            _parse_csv(args.extensions),
        )

    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    torchscript = args.torchscript.expanduser().resolve() if args.torchscript else None
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if torchscript is not None and not torchscript.is_file():
        raise SystemExit(f"TorchScript model not found: {torchscript}")

    if args.devices == "auto":
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    else:
        devices = _parse_csv(args.devices)

    cases = _iter_cases(_parse_csv(args.cases), args.case)
    durations = {audio_path: _audio_duration_seconds(audio_path) for audio_path in audio_paths}

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv.expanduser().resolve() if args.csv else output_root / "inference_benchmark.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"skip device={device}: CUDA is not available", file=sys.stderr)
            continue

        for case in cases:
            if case.amp and not device.startswith("cuda"):
                print(f"skip case={case.label} device={device}: AMP only applies to CUDA")
                continue

            print(
                f"benchmark device={device} case={case.label} songs={len(audio_paths)} "
                f"chunk={case.chunk_frames} overlap={case.overlap_frames} amp={case.amp}"
            )
            model, base_cfg, checkpoint_obj = _load_model(checkpoint, torchscript, device)
            cfg = replace(
                base_cfg,
                device=device,
                stems=list(STEMS_4),
                export_files=not args.no_export,
                validate_outputs=not args.no_validate,
                renorm_masks=True,
                chunk_frames=case.chunk_frames,
                overlap_frames=case.overlap_frames,
                amp=case.amp,
            )

            case_dir = output_root / device.replace(":", "_") / case.label
            case_elapsed_values: list[float] = []

            for song_idx, audio_path in enumerate(audio_paths, start=1):
                duration = durations[audio_path]
                song_dir = case_dir / f"{song_idx:03d}_{_safe_name(audio_path)}"
                print(
                    f"  song={song_idx}/{len(audio_paths)} input={audio_path.name} "
                    f"duration={duration:.3f}s"
                )

                for warmup_idx in range(args.warmups):
                    _run_once(
                        audio_path=audio_path,
                        model=model,
                        cfg=cfg,
                        output_dir=song_dir / f"warmup_{warmup_idx + 1}",
                        export_files=not args.no_export,
                        checkpoint_obj=checkpoint_obj,
                    )

                elapsed_values: list[float] = []
                for repeat_idx in range(args.repeats):
                    elapsed, cuda_peak_mb, rss_mb = _run_once(
                        audio_path=audio_path,
                        model=model,
                        cfg=cfg,
                        output_dir=song_dir / f"repeat_{repeat_idx + 1}",
                        export_files=not args.no_export,
                        checkpoint_obj=checkpoint_obj,
                    )
                    elapsed_values.append(elapsed)
                    case_elapsed_values.append(elapsed)
                    rows.append(
                        {
                            "input_file": str(audio_path),
                            "song_index": song_idx,
                            "song_count": len(audio_paths),
                            "audio_duration_seconds": f"{duration:.6f}",
                            "artifact": str(checkpoint or torchscript),
                            "device": device,
                            "case": case.label,
                            "chunk_frames": case.chunk_frames,
                            "overlap_frames": case.overlap_frames,
                            "amp": case.amp,
                            "export_files": not args.no_export,
                            "validate_outputs": not args.no_validate,
                            "repeat": repeat_idx + 1,
                            "wall_seconds": f"{elapsed:.6f}",
                            "realtime_factor": f"{elapsed / duration:.6f}",
                            "peak_cuda_memory_mb": (
                                "" if cuda_peak_mb is None else f"{cuda_peak_mb:.3f}"
                            ),
                            "max_rss_mb": f"{rss_mb:.3f}",
                        }
                    )
                    print(
                        f"    repeat={repeat_idx + 1} wall={elapsed:.3f}s "
                        f"rtf={elapsed / duration:.3f}x "
                        f"cuda_peak_mb={cuda_peak_mb if cuda_peak_mb is not None else 'n/a'}"
                    )

                spread = stdev(elapsed_values) if len(elapsed_values) > 1 else 0.0
                print(
                    f"  summary song={audio_path.name}: "
                    f"mean={mean(elapsed_values):.3f}s stdev={spread:.3f}s "
                    f"mean_rtf={mean(elapsed_values) / duration:.3f}x"
                )

            spread = stdev(case_elapsed_values) if len(case_elapsed_values) > 1 else 0.0
            print(
                f"summary device={device} case={case.label} all_songs: "
                f"mean={mean(case_elapsed_values):.3f}s stdev={spread:.3f}s"
            )

            del model
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No benchmark rows were produced.")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
