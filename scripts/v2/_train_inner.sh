#!/usr/bin/env bash
# scripts/_train_inner.sh
#
# Inner training+eval+export pipeline. Invoked by scripts/stemmy (via the
# rendered sbatch template) which creates the run directory and pre-renders
# the config env file. Can also be invoked directly for ad-hoc runs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train_eval_export.sh [--env PATH] --partition dgx2|dgxh|gpu|ampere [overrides...]

This runs:
  1+2) training + parallel eval watcher (concurrent)
  2.5) upload eval metrics to wandb
  3) best checkpoint selection
  4) torchscript export (.pt)

Directory layout (single timestamped directory per run):

  RUNS_BASE/run_<partition>_<stamp>/
    logs/
      console.log
    run_info.txt
    checkpoints/
    eval/
    best/
      unet_best_<partition>_<stamp>.pth
      unet_best_<partition>_<stamp>.pt

It also writes stable "latest" pointers at RUNS_BASE:
  RUNS_BASE/latest_run_<partition>      (symlink to run directory)
  RUNS_BASE/latest_best_<partition>.pth (symlink to best .pth within latest run)
  RUNS_BASE/latest_best_<partition>.pt  (symlink to best .pt within latest run)
  RUNS_BASE/latest_<partition>.env      (exports paths for downstream scripts)

Defaults are loaded from scripts/stemmy.env if present. CLI flags override env.

Common overrides:
  --onid ONID
  --data-root PATH
  --runs-base PATH
  --epochs N
  --lr FLOAT
  --weight-decay FLOAT
  --time-frames 256|512
  --max-tracks N          (train.py: limit number of tracks per split; 0 = no limit)
  --base-channels N
  --num-workers N         (0 allowed)
  --batch-size N          (skip probing)
  --device cpu|cuda|cuda:N
  --n-eval-tracks N
  --max-seconds N
  --waveform-norm peak|rms|none
  --spectrogram-norm freq_minmax|none
  --amp 0|1

Optional training retry behavior (OFF by default):
  --lr-backoff 0|1
  --lr-backoff-factor FLOAT      (multiplier in (0,1); default 0.5)
  --lr-backoff-max-tries N       (default 3)

Eval progress + durability overrides (passed into stemmy.tool.fullsong_eval_masked.py):
  --eval-progress 0|1
  --eval-print-metrics 0|1
  --eval-flush-every N
  --eval-fsync-every N     (0 disables)
  --eval-print-every-tracks N

Example (minimal run):
  PYTHONUNBUFFERED=1 scripts/train_eval_export.sh --partition gpu \
    --epochs 1 --max-tracks 1 --time-frames 256 --batch-size 1 --num-workers 0 \
    --save-every-epochs 1 --n-eval-tracks 1 --max-seconds 5 --device cuda \
    --eval-progress 1 --eval-print-metrics 0 --eval-flush-every 1 --eval-fsync-every 0 \
    --lr-backoff 0
EOF
}

ENV_FILE=""
RUN_DIR_ARG=""

PARTITION=""

ONID=""
DATA_ROOT=""
RUNS_BASE=""

EPOCHS=""
LR=""
WEIGHT_DECAY=""
TIME_FRAMES=""
MAX_TRACKS=""
BASE_CHANNELS=""
LR_FACTOR=""
LR_PATIENCE=""
MIN_LR=""
SAVE_EVERY_EPOCHS=""
WAVEFORM_NORM=""
SPECTROGRAM_NORM=""
AMP=""
DEVICE=""

N_EVAL_TRACKS=""
MAX_SECONDS=""
EVAL_EVERY_N_CKPTS=""
CHUNK_FRAMES=""

NUM_WORKERS=""
BATCH_SIZE=""   # if set, skip probing

RESUME=""          # explicit path to a .pth checkpoint to resume from
RESUME_LATEST=""   # if 1, resolve RUNS_BASE/latest_best_<partition>.pth automatically
RESUME_LAST=""     # if 1, resolve RUNS_BASE/latest_last_<partition>.pth (highest epoch)

# Optional LR backoff / retry (default OFF)
LR_BACKOFF=""
LR_BACKOFF_FACTOR=""
LR_BACKOFF_MAX_TRIES=""

# Eval controls (forwarded as env vars to stemmy.tool.fullsong_eval_masked.py)
EVAL_PROGRESS=""
EVAL_PRINT_METRICS=""
EVAL_FLUSH_EVERY=""
EVAL_FSYNC_EVERY=""
EVAL_PRINT_EVERY_TRACKS=""
EVAL_WATCH_POLL_SECONDS=""

# Reproducibility
SEED=""

# Augmentation (audiomentations). All probabilities default 0 (disabled).
AUG_PROFILE=""
AUG_GAIN_P=""
AUG_GAIN_DB=""
AUG_PITCH_P=""
AUG_PITCH_SEMITONES=""
AUG_TIME_STRETCH_P=""
AUG_TIME_STRETCH_RANGE=""
AUG_SHIFT_P=""
AUG_SHIFT_FRACTION=""
AUG_POLARITY_P=""
AUG_NOISE_P=""
AUG_NOISE_AMPLITUDE=""

# Load env defaults FIRST so CLI flags override.
if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_FILE="${2:-}"
      shift 2
      if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        set +a
      fi
      ;;

    --partition) PARTITION="${2:-}"; shift 2 ;;
    --run-dir) RUN_DIR_ARG="${2:-}"; shift 2 ;;

    --onid) ONID="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --runs-base) RUNS_BASE="${2:-}"; shift 2 ;;

    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --lr) LR="${2:-}"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="${2:-}"; shift 2 ;;
    --time-frames) TIME_FRAMES="${2:-}"; shift 2 ;;
    --max-tracks) MAX_TRACKS="${2:-}"; shift 2 ;;
    --base-channels) BASE_CHANNELS="${2:-}"; shift 2 ;;
    --lr-factor) LR_FACTOR="${2:-}"; shift 2 ;;
    --lr-patience) LR_PATIENCE="${2:-}"; shift 2 ;;
    --min-lr) MIN_LR="${2:-}"; shift 2 ;;
    --save-every-epochs) SAVE_EVERY_EPOCHS="${2:-}"; shift 2 ;;
    --waveform-norm) WAVEFORM_NORM="${2:-}"; shift 2 ;;
    --spectrogram-norm) SPECTROGRAM_NORM="${2:-}"; shift 2 ;;
    --amp) AMP="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;

    --n-eval-tracks) N_EVAL_TRACKS="${2:-}"; shift 2 ;;
    --max-seconds) MAX_SECONDS="${2:-}"; shift 2 ;;
    --eval-every-n-ckpts) EVAL_EVERY_N_CKPTS="${2:-}"; shift 2 ;;
    --chunk-frames) CHUNK_FRAMES="${2:-}"; shift 2 ;;

    --num-workers) NUM_WORKERS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --resume) RESUME="${2:-}"; shift 2 ;;
    --resume-latest) RESUME_LATEST="1"; shift ;;
    --resume-last) RESUME_LAST="1"; shift ;;

    --lr-backoff) LR_BACKOFF="${2:-}"; shift 2 ;;
    --lr-backoff-factor) LR_BACKOFF_FACTOR="${2:-}"; shift 2 ;;
    --lr-backoff-max-tries) LR_BACKOFF_MAX_TRIES="${2:-}"; shift 2 ;;

    --eval-progress) EVAL_PROGRESS="${2:-}"; shift 2 ;;
    --eval-print-metrics) EVAL_PRINT_METRICS="${2:-}"; shift 2 ;;
    --eval-flush-every) EVAL_FLUSH_EVERY="${2:-}"; shift 2 ;;
    --eval-fsync-every) EVAL_FSYNC_EVERY="${2:-}"; shift 2 ;;
    --eval-print-every-tracks) EVAL_PRINT_EVERY_TRACKS="${2:-}"; shift 2 ;;
    --eval-watch-poll-seconds) EVAL_WATCH_POLL_SECONDS="${2:-}"; shift 2 ;;

    --seed) SEED="${2:-}"; shift 2 ;;

    --aug-profile) AUG_PROFILE="${2:-}"; shift 2 ;;
    --aug-gain-p) AUG_GAIN_P="${2:-}"; shift 2 ;;
    --aug-gain-db) AUG_GAIN_DB="${2:-}"; shift 2 ;;
    --aug-pitch-p) AUG_PITCH_P="${2:-}"; shift 2 ;;
    --aug-pitch-semitones) AUG_PITCH_SEMITONES="${2:-}"; shift 2 ;;
    --aug-time-stretch-p) AUG_TIME_STRETCH_P="${2:-}"; shift 2 ;;
    --aug-time-stretch-range) AUG_TIME_STRETCH_RANGE="${2:-}"; shift 2 ;;
    --aug-shift-p) AUG_SHIFT_P="${2:-}"; shift 2 ;;
    --aug-shift-fraction) AUG_SHIFT_FRACTION="${2:-}"; shift 2 ;;
    --aug-polarity-p) AUG_POLARITY_P="${2:-}"; shift 2 ;;
    --aug-noise-p) AUG_NOISE_P="${2:-}"; shift 2 ;;
    --aug-noise-amplitude) AUG_NOISE_AMPLITUDE="${2:-}"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Partition required
if [[ -z "${PARTITION}" ]]; then
  echo "ERROR: --partition required (dgx2|dgxh|gpu|ampere)" >&2
  exit 2
fi
if [[ "${PARTITION}" != "dgx2" && "${PARTITION}" != "dgxh" && "${PARTITION}" != "gpu" && "${PARTITION}" != "ampere" ]]; then
  echo "ERROR: --partition must be dgx2 or dgxh or gpu or ampere" >&2
  exit 2
fi

# Pull from env if flags were not provided
ONID="${ONID:-}"
DATA_ROOT="${DATA_ROOT:-}"
RUNS_BASE="${RUNS_BASE:-}"

if [[ -z "${ONID}" ]]; then
  echo "ERROR: ONID missing. Set ONID in scripts/stemmy.env or pass --onid" >&2
  exit 2
fi
if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT missing. Set DATA_ROOT in scripts/stemmy.env or pass --data-root" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: data root not found: ${DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}/train" || ! -d "${DATA_ROOT}/test" ]]; then
  echo "ERROR: expected train/ and test/ under DATA_ROOT: ${DATA_ROOT}" >&2
  exit 2
fi

if [[ -z "${RUNS_BASE}" ]]; then
  RUNS_BASE="/nfs/hpc/share/${ONID}/stemmy/runs"
fi
mkdir -p "${RUNS_BASE}"

DEVICE="${DEVICE:-cuda}"
if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" && "${DEVICE}" != cuda:* ]]; then
  echo "ERROR: DEVICE must be cpu|cuda|cuda:N (got: ${DEVICE})" >&2
  exit 2
fi

# Require GPU if device is cuda*
if [[ "${DEVICE}" == "cuda" || "${DEVICE}" == cuda:* ]]; then
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available (torch.cuda.is_available() is False). Run this inside a GPU allocation." >&2
    exit 2
  fi
fi

# Avoid CPU oversubscription from BLAS/OMP inside DataLoader workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

GPU_NAME=""
GPU_MEM_MIB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r' || true)"
  GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '\r' || true)"
fi
# Fall back to torch if nvidia-smi failed or returned nothing (e.g. MIG slices)
if [[ -z "${GPU_NAME}" && ("${DEVICE}" == "cuda" || "${DEVICE}" == cuda:*) ]]; then
  GPU_NAME="$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || true)"
fi
if [[ -z "${GPU_MEM_MIB}" && ("${DEVICE}" == "cuda" || "${DEVICE}" == cuda:*) ]]; then
  GPU_MEM_MIB="$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**2))" 2>/dev/null || true)"
fi

# Defaults (env can override, CLI overrides env already)
EPOCHS="${EPOCHS:-300}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
TIME_FRAMES="${TIME_FRAMES:-512}"
MAX_TRACKS="${MAX_TRACKS:-0}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LR_PATIENCE="${LR_PATIENCE:-10}"
MIN_LR="${MIN_LR:-1e-6}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
WAVEFORM_NORM="${WAVEFORM_NORM:-peak}"
SPECTROGRAM_NORM="${SPECTROGRAM_NORM:-freq_minmax}"
AMP="${AMP:-0}"
RESUME_LATEST="${RESUME_LATEST:-0}"

N_EVAL_TRACKS="${N_EVAL_TRACKS:-30}"
MAX_SECONDS="${MAX_SECONDS:-30}"
EVAL_EVERY_N_CKPTS="${EVAL_EVERY_N_CKPTS:-1}"
CHUNK_FRAMES="${CHUNK_FRAMES:-0}"

# Optional LR backoff defaults (OFF)
LR_BACKOFF="${LR_BACKOFF:-0}"
LR_BACKOFF_FACTOR="${LR_BACKOFF_FACTOR:-0.5}"
LR_BACKOFF_MAX_TRIES="${LR_BACKOFF_MAX_TRIES:-3}"

# Eval controls default from env (or fallback)
EVAL_PROGRESS="${EVAL_PROGRESS:-1}"
EVAL_PRINT_METRICS="${EVAL_PRINT_METRICS:-0}"
EVAL_FLUSH_EVERY="${EVAL_FLUSH_EVERY:-1}"
EVAL_FSYNC_EVERY="${EVAL_FSYNC_EVERY:-0}"
EVAL_PRINT_EVERY_TRACKS="${EVAL_PRINT_EVERY_TRACKS:-1}"
EVAL_WATCH_POLL_SECONDS="${EVAL_WATCH_POLL_SECONDS:-60}"

# Reproducibility default (matches recon-p10-valid14-fullsong-baseline-seed42_20260514)
SEED="${SEED:-42}"

# AUG_PROFILE dispatcher: fills probabilities that are still unset (empty).
# Must run BEFORE the probability :-0 defaults below so the profile can
# distinguish "unset" from "explicitly 0" — `--aug-gain-p 0` with a profile
# now correctly keeps gain disabled.
AUG_PROFILE="${AUG_PROFILE:-}"
case "${AUG_PROFILE}" in
  ""|off)
    : # leave probabilities as-is; the :-0 defaults below disable any unset ones.
    ;;
  all_medium)
    AUG_GAIN_P="${AUG_GAIN_P:-0.5}"
    AUG_PITCH_P="${AUG_PITCH_P:-0.5}"
    AUG_TIME_STRETCH_P="${AUG_TIME_STRETCH_P:-0.5}"
    AUG_SHIFT_P="${AUG_SHIFT_P:-0.5}"
    AUG_POLARITY_P="${AUG_POLARITY_P:-0.5}"
    AUG_NOISE_P="${AUG_NOISE_P:-0.5}"
    ;;
  *)
    echo "ERROR: unknown AUG_PROFILE: ${AUG_PROFILE} (expected one of: off, all_medium)" >&2
    exit 2
    ;;
esac

# Augmentation defaults: any probability still unset disables that aug; strengths
# default to neutral values.
AUG_GAIN_P="${AUG_GAIN_P:-0}"
AUG_GAIN_DB="${AUG_GAIN_DB:-6.0}"
AUG_PITCH_P="${AUG_PITCH_P:-0}"
AUG_PITCH_SEMITONES="${AUG_PITCH_SEMITONES:-2.0}"
AUG_TIME_STRETCH_P="${AUG_TIME_STRETCH_P:-0}"
AUG_TIME_STRETCH_RANGE="${AUG_TIME_STRETCH_RANGE:-0.1}"
AUG_SHIFT_P="${AUG_SHIFT_P:-0}"
AUG_SHIFT_FRACTION="${AUG_SHIFT_FRACTION:-0.1}"
AUG_POLARITY_P="${AUG_POLARITY_P:-0}"
AUG_NOISE_P="${AUG_NOISE_P:-0}"
AUG_NOISE_AMPLITUDE="${AUG_NOISE_AMPLITUDE:-0.005}"

# Validate key ints
if [[ "${TIME_FRAMES}" != "256" && "${TIME_FRAMES}" != "512" ]]; then
  echo "ERROR: TIME_FRAMES must be 256 or 512" >&2
  exit 2
fi

for vname in EPOCHS BASE_CHANNELS LR_PATIENCE SAVE_EVERY_EPOCHS N_EVAL_TRACKS MAX_SECONDS EVAL_EVERY_N_CKPTS MAX_TRACKS LR_BACKOFF_MAX_TRIES; do
  val="${!vname}"
  if ! [[ "${val}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${vname} must be an integer (got: ${val})" >&2
    exit 2
  fi
done

if ! [[ "${LR_BACKOFF}" =~ ^[01]$ ]]; then
  echo "ERROR: LR_BACKOFF must be 0 or 1 (got: ${LR_BACKOFF})" >&2
  exit 2
fi

python - <<PY
import sys
try:
    f = float("${LR_BACKOFF_FACTOR}")
except Exception:
    print("ERROR: LR_BACKOFF_FACTOR must be a float (got: ${LR_BACKOFF_FACTOR})", file=sys.stderr)
    sys.exit(2)
if not (0.0 < f < 1.0):
    print("ERROR: LR_BACKOFF_FACTOR must be in (0,1) (got: %s)" % "${LR_BACKOFF_FACTOR}", file=sys.stderr)
    sys.exit(2)
PY

if [[ "${EPOCHS}" -le 0 ]]; then
  echo "ERROR: EPOCHS must be > 0" >&2
  exit 2
fi
if [[ "${N_EVAL_TRACKS}" -le 0 ]]; then
  echo "ERROR: N_EVAL_TRACKS must be > 0" >&2
  exit 2
fi
if [[ "${EVAL_EVERY_N_CKPTS}" -lt 1 ]]; then
  echo "ERROR: EVAL_EVERY_N_CKPTS must be >= 1" >&2
  exit 2
fi
if [[ "${MAX_SECONDS}" -lt 0 ]]; then
  echo "ERROR: MAX_SECONDS must be >= 0" >&2
  exit 2
fi
if [[ "${SAVE_EVERY_EPOCHS}" -le 0 ]]; then
  echo "ERROR: SAVE_EVERY_EPOCHS must be > 0" >&2
  exit 2
fi
if [[ "${LR_BACKOFF_MAX_TRIES}" -le 0 ]]; then
  echo "ERROR: LR_BACKOFF_MAX_TRIES must be > 0" >&2
  exit 2
fi

case "${WAVEFORM_NORM}" in
  peak|rms|none) ;;
  *) echo "ERROR: WAVEFORM_NORM must be peak|rms|none (got: ${WAVEFORM_NORM})" >&2; exit 2 ;;
esac

case "${SPECTROGRAM_NORM}" in
  log1p|freq_minmax|none) ;;
  *) echo "ERROR: SPECTROGRAM_NORM must be log1p|freq_minmax|none (got: ${SPECTROGRAM_NORM})" >&2; exit 2 ;;
esac

if ! [[ "${AMP}" =~ ^[01]$ ]]; then
  echo "ERROR: AMP must be 0 or 1 (got: ${AMP})" >&2
  exit 2
fi

for vname in EVAL_PROGRESS EVAL_PRINT_METRICS; do
  val="${!vname}"
  if ! [[ "${val}" =~ ^[01]$ ]]; then
    echo "ERROR: ${vname} must be 0 or 1 (got: ${val})" >&2
    exit 2
  fi
done

for vname in EVAL_FLUSH_EVERY EVAL_FSYNC_EVERY EVAL_PRINT_EVERY_TRACKS; do
  val="${!vname}"
  if ! [[ "${val}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${vname} must be an integer (got: ${val})" >&2
    exit 2
  fi
done
if [[ "${EVAL_FLUSH_EVERY}" -le 0 ]]; then
  echo "ERROR: EVAL_FLUSH_EVERY must be > 0" >&2
  exit 2
fi
if [[ "${EVAL_PRINT_EVERY_TRACKS}" -le 0 ]]; then
  echo "ERROR: EVAL_PRINT_EVERY_TRACKS must be > 0" >&2
  exit 2
fi

# Derive NUM_WORKERS if not explicitly set
NUM_WORKERS_MAX="${NUM_WORKERS_MAX:-24}"
if ! [[ "${NUM_WORKERS_MAX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NUM_WORKERS_MAX must be an integer (got: ${NUM_WORKERS_MAX})" >&2
  exit 2
fi

if [[ -z "${NUM_WORKERS:-}" ]]; then
  if [[ "${PARTITION}" == "dgxh" ]]; then
    NUM_WORKERS="24"
  else
    NUM_WORKERS="16"
  fi

  if [[ "${NUM_WORKERS}" -gt "${NUM_WORKERS_MAX}" ]]; then
    NUM_WORKERS="${NUM_WORKERS_MAX}"
  fi

  if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]]; then
    cap=$(( SLURM_CPUS_PER_TASK - 2 ))
    if [[ "${cap}" -lt 0 ]]; then cap=0; fi
    if [[ "${NUM_WORKERS}" -gt "${cap}" ]]; then
      NUM_WORKERS="${cap}"
    fi
  fi
else
  if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || [[ "${NUM_WORKERS}" -lt 0 ]]; then
    echo "ERROR: --num-workers must be an integer >= 0" >&2
    exit 2
  fi
fi

# Resolve --resume-latest to the actual symlink path
if [[ "${RESUME_LATEST}" == "1" ]]; then
  if [[ -n "${RESUME}" ]]; then
    echo "ERROR: --resume and --resume-latest are mutually exclusive" >&2
    exit 2
  fi
  LATEST_PTH_CANDIDATE="${RUNS_BASE}/latest_best_${PARTITION}.pth"
  if [[ ! -e "${LATEST_PTH_CANDIDATE}" ]]; then
    echo "ERROR: --resume-latest: no previous run found at ${LATEST_PTH_CANDIDATE}" >&2
    exit 2
  fi
  RESUME="$(realpath "${LATEST_PTH_CANDIDATE}")"
  echo "Resuming from latest best checkpoint: ${RESUME}"
fi

# Resolve --resume-last (highest epoch, not best SI-SDR) — preferred for
# continuing training.
if [[ "${RESUME_LAST}" == "1" ]]; then
  if [[ -n "${RESUME}" ]]; then
    echo "ERROR: --resume and --resume-last are mutually exclusive" >&2
    exit 2
  fi
  LAST_PTH_CANDIDATE="${RUNS_BASE}/latest_last_${PARTITION}.pth"
  if [[ ! -e "${LAST_PTH_CANDIDATE}" ]]; then
    echo "ERROR: --resume-last: no previous run found at ${LAST_PTH_CANDIDATE}" >&2
    exit 2
  fi
  RESUME="$(realpath "${LAST_PTH_CANDIDATE}")"
  echo "Resuming from latest (highest-epoch) checkpoint: ${RESUME}"
fi

if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "ERROR: --resume checkpoint not found: ${RESUME}" >&2
  exit 2
fi

# Batch size selection: probe unless BATCH_SIZE provided
if [[ -n "${BATCH_SIZE:-}" ]]; then
  if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -le 0 ]]; then
    echo "ERROR: --batch-size must be a positive integer" >&2
    exit 2
  fi
else
  echo "Selecting batch size by probing GPU memory with real forward/backward..."

  CANDIDATES=(8 6 4 2 1)
  if [[ -n "${GPU_MEM_MIB}" && "${GPU_MEM_MIB}" =~ ^[0-9]+$ ]]; then
    if [[ "${GPU_MEM_MIB}" -ge 78000 ]]; then
      CANDIDATES=(32 24 16 12 8 6 4 2 1)
    elif [[ "${GPU_MEM_MIB}" -ge 38000 ]]; then
      CANDIDATES=(16 12 8 6 4 2 1)
    elif [[ "${GPU_MEM_MIB}" -ge 24000 ]]; then
      CANDIDATES=(12 8 6 4 2 1)
    fi
  fi

  N_FFT="${N_FFT:-4096}"
  if ! [[ "${N_FFT}" =~ ^[0-9]+$ ]] || [[ "${N_FFT}" -le 0 ]]; then
    echo "ERROR: N_FFT must be a positive integer if set (got: ${N_FFT})" >&2
    exit 2
  fi

  BATCH_SIZE=""
  for bs in "${CANDIDATES[@]}"; do
    set +e
    python - <<PY
import sys
import torch
from stemmy.models.unet_2d import UNet2D

bs = int("${bs}")
T = int("${TIME_FRAMES}")
base_channels = int("${BASE_CHANNELS}")
device_str = "${DEVICE}"

n_fft = int("${N_FFT}")
F = (n_fft // 2) + 1

device = torch.device(device_str)
if device.type == "cuda" and not torch.cuda.is_available():
    print("cuda requested but not available")
    sys.exit(10)

model = UNet2D(stems=4, base_channels=base_channels).to(device)
model.train()
x = torch.randn((bs, 2, F, T), device=device, dtype=torch.float32)

try:
    y = model(x)
    loss = y.mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sys.exit(0)
except RuntimeError as e:
    msg = str(e).lower()
    if "out of memory" in msg or "cuda error" in msg:
        sys.exit(10)
    raise
PY
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      BATCH_SIZE="${bs}"
      break
    fi
  done

  if [[ -z "${BATCH_SIZE}" ]]; then
    echo "ERROR: Could not find a working batch size (even 1) during probe." >&2
    exit 2
  fi
fi

# --------------------------
# Single run directory layout
# --------------------------
# If the caller (scripts/stemmy) provided --run-dir, trust it as-is. Otherwise
# create a fresh timestamped directory under RUNS_BASE for backward-compatible
# ad-hoc invocation.
if [[ -n "${RUN_DIR_ARG}" ]]; then
  RUN_DIR="${RUN_DIR_ARG}"
  STAMP="$(basename "${RUN_DIR}")"
else
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${RUNS_BASE}/run_${PARTITION}_${STAMP}"
fi
LOG_DIR="${RUN_DIR}/logs"
CKPT_DIR="${RUN_DIR}/checkpoints"
EVAL_DIR="${RUN_DIR}/eval"
BEST_DIR="${RUN_DIR}/best"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${EVAL_DIR}" "${BEST_DIR}"

CONSOLE_LOG="${LOG_DIR}/console.log"

# Capture all stdout/stderr into the run directory log while still showing it.
# This covers the entire script and all child processes.
exec > >(tee -a "${CONSOLE_LOG}") 2>&1

SCRIPT_START=$(date +%s)
TIMING_FILE="${RUN_DIR}/timing.txt"
printf "script_start_epoch=%s\n" "${SCRIPT_START}" > "${TIMING_FILE}"

# Print HH:MM:SS elapsed since SCRIPT_START
fmt_elapsed() {
  local secs=$(( $(date +%s) - SCRIPT_START ))
  printf "%02d:%02d:%02d" $(( secs / 3600 )) $(( (secs % 3600) / 60 )) $(( secs % 60 ))
}

RUN_INFO="${RUN_DIR}/run_info.txt"

echo "=== RUN INFO === [$(date '+%Y-%m-%d %H:%M:%S')]"
{
  echo "timestamp=${STAMP}"
  echo "onid=${ONID}"
  echo "partition=${PARTITION}"
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "nodelist=${SLURM_NODELIST:-}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  echo "mem_per_node=${SLURM_MEM_PER_NODE:-}"
  echo "gpu_name=${GPU_NAME}"
  echo "gpu_mem_mib=${GPU_MEM_MIB}"
  echo "python=$(command -v python || true)"
  echo "pwd=$(pwd)"
  echo
  echo "data_root=${DATA_ROOT}"
  echo
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "lr=${LR}"
  echo "weight_decay=${WEIGHT_DECAY}"
  echo "num_workers=${NUM_WORKERS}"
  echo "time_frames=${TIME_FRAMES}"
  echo "max_tracks=${MAX_TRACKS}"
  echo "base_channels=${BASE_CHANNELS}"
  echo "lr_factor=${LR_FACTOR}"
  echo "lr_patience=${LR_PATIENCE}"
  echo "min_lr=${MIN_LR}"
  echo "save_every_epochs=${SAVE_EVERY_EPOCHS}"
  echo "waveform_norm=${WAVEFORM_NORM}"
  echo "spectrogram_norm=${SPECTROGRAM_NORM}"
  echo "amp=${AMP}"
  echo "device=${DEVICE}"
  echo
  echo "n_eval_tracks=${N_EVAL_TRACKS}"
  echo "max_seconds=${MAX_SECONDS}"
  echo "eval_every_n_ckpts=${EVAL_EVERY_N_CKPTS}"
  echo "chunk_frames=${CHUNK_FRAMES}"
  echo
  echo "lr_backoff=${LR_BACKOFF}"
  echo "lr_backoff_factor=${LR_BACKOFF_FACTOR}"
  echo "lr_backoff_max_tries=${LR_BACKOFF_MAX_TRIES}"
  echo
  echo "eval_progress=${EVAL_PROGRESS}"
  echo "eval_print_metrics=${EVAL_PRINT_METRICS}"
  echo "eval_flush_every=${EVAL_FLUSH_EVERY}"
  echo "eval_fsync_every=${EVAL_FSYNC_EVERY}"
  echo "eval_print_every_tracks=${EVAL_PRINT_EVERY_TRACKS}"
  echo
  echo "seed=${SEED}"
  echo "aug_profile=${AUG_PROFILE}"
  echo "aug_gain_p=${AUG_GAIN_P} aug_gain_db=${AUG_GAIN_DB}"
  echo "aug_pitch_p=${AUG_PITCH_P} aug_pitch_semitones=${AUG_PITCH_SEMITONES}"
  echo "aug_time_stretch_p=${AUG_TIME_STRETCH_P} aug_time_stretch_range=${AUG_TIME_STRETCH_RANGE}"
  echo "aug_shift_p=${AUG_SHIFT_P} aug_shift_fraction=${AUG_SHIFT_FRACTION}"
  echo "aug_polarity_p=${AUG_POLARITY_P}"
  echo "aug_noise_p=${AUG_NOISE_P} aug_noise_amplitude=${AUG_NOISE_AMPLITUDE}"
  echo
  echo "run_dir=${RUN_DIR}"
  echo "log_dir=${LOG_DIR}"
  echo "console_log=${CONSOLE_LOG}"
  echo "checkpoints_dir=${CKPT_DIR}"
  echo "eval_dir=${EVAL_DIR}"
  echo "best_dir=${BEST_DIR}"
  echo
  echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "start_epoch=${SCRIPT_START}"
} | tee "${RUN_INFO}"
echo

PHASE12_START=$(date +%s)
printf "phase12_start_epoch=%s\n" "${PHASE12_START}" >> "${TIMING_FILE}"
echo "=== Phase 1+2/4: Train + Eval watcher (parallel) === [$(date '+%Y-%m-%d %H:%M:%S'), elapsed: $(fmt_elapsed)]"

run_train_once() {
  local lr_val="$1"
  local -a train_args=(
    --data-root "${DATA_ROOT}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${lr_val}"
    --weight-decay "${WEIGHT_DECAY}"
    --num-workers "${NUM_WORKERS}"
    --time-frames "${TIME_FRAMES}"
    --max-tracks "${MAX_TRACKS}"
    --base-channels "${BASE_CHANNELS}"
    --lr-factor "${LR_FACTOR}"
    --lr-patience "${LR_PATIENCE}"
    --min-lr "${MIN_LR}"
    --waveform-norm "${WAVEFORM_NORM}"
    --spectrogram-norm "${SPECTROGRAM_NORM}"
    --checkpoint-dir "${CKPT_DIR}"
    --save-every-epochs "${SAVE_EVERY_EPOCHS}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --aug-gain-p "${AUG_GAIN_P}"
    --aug-gain-db "${AUG_GAIN_DB}"
    --aug-pitch-p "${AUG_PITCH_P}"
    --aug-pitch-semitones "${AUG_PITCH_SEMITONES}"
    --aug-time-stretch-p "${AUG_TIME_STRETCH_P}"
    --aug-time-stretch-range "${AUG_TIME_STRETCH_RANGE}"
    --aug-shift-p "${AUG_SHIFT_P}"
    --aug-shift-fraction "${AUG_SHIFT_FRACTION}"
    --aug-polarity-p "${AUG_POLARITY_P}"
    --aug-noise-p "${AUG_NOISE_P}"
    --aug-noise-amplitude "${AUG_NOISE_AMPLITUDE}"
  )

  if [[ "${AMP}" == "1" ]]; then
    train_args+=(--amp)
  fi

  if [[ -n "${RESUME}" ]]; then
    train_args+=(--resume "${RESUME}")
  fi

  python -m stemmy.train "${train_args[@]}"
}

# File touched by training-done signal so the eval watcher knows when to stop.
TRAIN_DONE_FILE="${RUN_DIR}/training_done.flag"
EVAL_WATCH_POLL_SECONDS="${EVAL_WATCH_POLL_SECONDS:-60}"

# Eval watcher: single Python process that holds one wandb run open for the full training
# duration. It polls CKPT_DIR for new checkpoints, evaluates each as it appears, and exits
# when TRAIN_DONE_FILE is written.
EVAL_WATCH_MODE="1" \
TRAIN_DONE_FILE="${TRAIN_DONE_FILE}" \
EVAL_WATCH_POLL_SECONDS="${EVAL_WATCH_POLL_SECONDS}" \
EVAL_EVERY_N_CKPTS="${EVAL_EVERY_N_CKPTS}" \
EVAL_PROGRESS="${EVAL_PROGRESS}" \
EVAL_PRINT_METRICS="${EVAL_PRINT_METRICS}" \
EVAL_FLUSH_EVERY="${EVAL_FLUSH_EVERY}" \
EVAL_FSYNC_EVERY="${EVAL_FSYNC_EVERY}" \
EVAL_PRINT_EVERY_TRACKS="${EVAL_PRINT_EVERY_TRACKS}" \
EVAL_CHUNK_FRAMES="${CHUNK_FRAMES}" \
DATA="${DATA_ROOT}" \
CKPT_DIR="${CKPT_DIR}" \
EVAL_DIR="${EVAL_DIR}" \
DEVICE="${DEVICE}" \
N_EVAL_TRACKS="${N_EVAL_TRACKS}" \
MAX_SECONDS="${MAX_SECONDS}" \
PYTHONUNBUFFERED=1 python -u -m stemmy.tool.fullsong_eval_masked &
EVAL_WATCHER_PID=$!

# Give the watcher a moment to start, then verify it didn't crash immediately.
sleep 5
if ! kill -0 "${EVAL_WATCHER_PID}" 2>/dev/null; then
  echo "ERROR: Eval watcher (PID ${EVAL_WATCHER_PID}) died immediately after launch. Check logs above." >&2
  exit 1
fi
echo "Eval watcher started (PID ${EVAL_WATCHER_PID})."

# Run training in foreground (existing retry logic preserved).
TRAIN_LR="${LR}"
TRAIN_RC=0

if [[ "${LR_BACKOFF}" == "0" ]]; then
  run_train_once "${TRAIN_LR}" || TRAIN_RC=$?
else
  attempt=1
  while [[ "${attempt}" -le "${LR_BACKOFF_MAX_TRIES}" ]]; do
    echo "Train attempt ${attempt}/${LR_BACKOFF_MAX_TRIES} with lr=${TRAIN_LR}"
    set +e
    run_train_once "${TRAIN_LR}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      break
    fi

    if [[ "${attempt}" -ge "${LR_BACKOFF_MAX_TRIES}" ]]; then
      TRAIN_RC="${rc}"
      break
    fi

    next_lr="$(python - <<PY
lr = float("${TRAIN_LR}")
factor = float("${LR_BACKOFF_FACTOR}")
print(lr * factor)
PY
)"
    below_min="$(python - <<PY
lr = float("${next_lr}")
min_lr = float("${MIN_LR}")
print("1" if lr < min_lr else "0")
PY
)"
    if [[ "${below_min}" == "1" ]]; then
      echo "ERROR: Next lr (${next_lr}) would fall below min_lr (${MIN_LR})." >&2
      TRAIN_RC=2
      break
    fi

    TRAIN_LR="${next_lr}"
    attempt=$(( attempt + 1 ))
  done
fi

# Signal eval watcher that training is done, then wait for it to finish.
touch "${TRAIN_DONE_FILE}"
WATCHER_RC=0
wait "${EVAL_WATCHER_PID}" || WATCHER_RC=$?
if [[ "${WATCHER_RC}" -ne 0 ]]; then
  echo "WARNING: Eval watcher exited with code ${WATCHER_RC} — SI-SDR metrics may be incomplete." >&2
fi

PHASE12_END=$(date +%s)
PHASE12_SECS=$(( PHASE12_END - PHASE12_START ))
printf "phase12_end_epoch=%s\nphase12_seconds=%s\n" "${PHASE12_END}" "${PHASE12_SECS}" >> "${TIMING_FILE}"
printf "Phase 1+2 done in %02d:%02d:%02d (total elapsed: %s)\n" \
  $(( PHASE12_SECS / 3600 )) $(( (PHASE12_SECS % 3600) / 60 )) $(( PHASE12_SECS % 60 )) \
  "$(fmt_elapsed)"

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "ERROR: Training failed with exit code ${TRAIN_RC}." >&2
  exit "${TRAIN_RC}"
fi

PHASE3_START=$(date +%s)
printf "phase3_start_epoch=%s\n" "${PHASE3_START}" >> "${TIMING_FILE}"
echo "=== Phase 3/4: Select best checkpoint === [$(date '+%Y-%m-%d %H:%M:%S'), elapsed: $(fmt_elapsed)]"
SUMMARY_CSV="${EVAL_DIR}/fullsong_eval_summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "ERROR: Expected summary CSV not found: ${SUMMARY_CSV}" >&2
  exit 2
fi

BEST_PTH="${BEST_DIR}/unet_best_${PARTITION}_${STAMP}.pth"

python -m stemmy.tool.select_best_checkpoint \
  --summary-csv "${SUMMARY_CSV}" \
  --ckpt-dir "${CKPT_DIR}" \
  --metric mean_sisdr \
  --top-k 10 \
  --copy-to "${BEST_PTH}"

PHASE3_END=$(date +%s)
PHASE3_SECS=$(( PHASE3_END - PHASE3_START ))
printf "phase3_end_epoch=%s\nphase3_seconds=%s\n" "${PHASE3_END}" "${PHASE3_SECS}" >> "${TIMING_FILE}"
printf "Phase 3 done in %02d:%02d:%02d (total elapsed: %s)\n" \
  $(( PHASE3_SECS / 3600 )) $(( (PHASE3_SECS % 3600) / 60 )) $(( PHASE3_SECS % 60 )) \
  "$(fmt_elapsed)"

PHASE4_START=$(date +%s)
printf "phase4_start_epoch=%s\n" "${PHASE4_START}" >> "${TIMING_FILE}"
echo "=== Phase 4/4: Export TorchScript (.pt) === [$(date '+%Y-%m-%d %H:%M:%S'), elapsed: $(fmt_elapsed)]"
BEST_PT="${BEST_DIR}/unet_best_${PARTITION}_${STAMP}.pt"

python - <<PY
from stemmy.models.unet_2d import UNet2D
from stemmy.training.checkpointing import load_checkpoint, export_torchscript

ckpt_path = r"${BEST_PTH}"
out_path = r"${BEST_PT}"

model = UNet2D(stems=4, base_channels=int("${BASE_CHANNELS}"))
load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu")
export_torchscript(out_path, model)
print(out_path)
PY

# --------------------------
# Stable pointers in RUNS_BASE
# --------------------------
LATEST_RUN="${RUNS_BASE}/latest_run_${PARTITION}"
LATEST_PTH="${RUNS_BASE}/latest_best_${PARTITION}.pth"
LATEST_PT="${RUNS_BASE}/latest_best_${PARTITION}.pt"
LATEST_ENV="${RUNS_BASE}/latest_${PARTITION}.env"

ln -sfn "${RUN_DIR}" "${LATEST_RUN}"
ln -sfn "${BEST_PTH}" "${LATEST_PTH}"
ln -sfn "${BEST_PT}" "${LATEST_PT}"

{
  echo "PARTITION=${PARTITION}"
  echo "STAMP=${STAMP}"
  echo "RUNS_BASE=${RUNS_BASE}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "CONSOLE_LOG=${CONSOLE_LOG}"
  echo "RUN_INFO=${RUN_INFO}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "EVAL_DIR=${EVAL_DIR}"
  echo "BEST_DIR=${BEST_DIR}"
  echo "BEST_PTH=${BEST_PTH}"
  echo "BEST_PT=${BEST_PT}"
  echo "LATEST_RUN=${LATEST_RUN}"
  echo "LATEST_PTH=${LATEST_PTH}"
  echo "LATEST_PT=${LATEST_PT}"
} > "${LATEST_ENV}"

PHASE4_END=$(date +%s)
PHASE4_SECS=$(( PHASE4_END - PHASE4_START ))
printf "phase4_end_epoch=%s\nphase4_seconds=%s\n" "${PHASE4_END}" "${PHASE4_SECS}" >> "${TIMING_FILE}"
printf "Phase 4 done in %02d:%02d:%02d (total elapsed: %s)\n" \
  $(( PHASE4_SECS / 3600 )) $(( (PHASE4_SECS % 3600) / 60 )) $(( PHASE4_SECS % 60 )) \
  "$(fmt_elapsed)"

TOTAL_SECS=$(( $(date +%s) - SCRIPT_START ))
TOTAL_HMS=$(printf "%02d:%02d:%02d" $(( TOTAL_SECS / 3600 )) $(( (TOTAL_SECS % 3600) / 60 )) $(( TOTAL_SECS % 60 )))
printf "total_seconds=%s\ntotal_minutes=%s\n" \
  "${TOTAL_SECS}" \
  "$(python -c "print(round(${TOTAL_SECS}/60, 2))")" >> "${TIMING_FILE}"

echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')" >> "${RUN_INFO}"
echo "total_runtime_seconds=${TOTAL_SECS}" >> "${RUN_INFO}"
echo "total_runtime=${TOTAL_HMS}" >> "${RUN_INFO}"

echo "=== Uploading timing metrics to wandb ==="
python - <<PY
import os, sys

if os.environ.get("NO_WANDB", "0") == "1":
    print("NO_WANDB=1 — skipping timing upload.")
    sys.exit(0)

timing = {}
with open(r"${TIMING_FILE}") as f:
    for line in f:
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            try:
                timing[k.strip()] = float(v.strip())
            except ValueError:
                pass

import wandb
from stemmy.wandb_config import (
    get_wandb_project,
    get_wandb_environment,
    make_wandb_run_name,
    WANDB_ENTITY,
)

env = get_wandb_environment()
project = os.environ.get("WANDB_PROJECT", get_wandb_project(env))
entity  = os.environ.get("WANDB_ENTITY", WANDB_ENTITY)
base    = os.environ.get("WANDB_RUN_NAME", "job")
run_name = make_wandb_run_name(f"{base}_timing" if base else "job_timing")

try:
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="job_timing",
        settings=wandb.Settings(_disable_stats=True),
    )
    metrics = {f"time/{k}": v for k, v in timing.items()}
    wandb.log(metrics)
    run.finish()
    print(f"Timing uploaded to wandb run: {run_name}")
except Exception as exc:
    print(f"WARNING: wandb timing upload failed: {exc}", file=sys.stderr)
PY

echo "Done. Total runtime: ${TOTAL_HMS}"
echo "Run dir:      ${RUN_DIR}"
echo "Console log:  ${CONSOLE_LOG}"
echo "Run info:     ${RUN_INFO}"
echo "Best .pth:    ${BEST_PTH}"
echo "Best .pt:     ${BEST_PT}"
echo "Latest run:   ${LATEST_RUN}"
echo "Latest .pth:  ${LATEST_PTH}"
echo "Latest .pt:   ${LATEST_PT}"
echo "Latest env:   ${LATEST_ENV}"
