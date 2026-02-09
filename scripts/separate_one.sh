#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/separate_one.sh [--env PATH]
                          [--partition dgx2|dgxh]
                          [--checkpoint PATH]
                          [--input-wav PATH] [--data-root PATH --input-rel RELPATH]
                          [--output-base DIR]
                          [--device cpu|cuda|cuda:N]

Defaults are loaded from scripts/stemmy.env if present.

Checkpoint resolution (in order):
  1) --checkpoint PATH (explicit)
  2) RUNS_BASE/latest_best_<partition>.pth (stable symlink written by train_eval_export.sh)

Input resolution:
  - If --input-wav is provided, use it.
  - Else use DATA_ROOT + "/" + INPUT_REL (DATA_ROOT and INPUT_REL can come from env or flags).

Examples:
  scripts/separate_one.sh --partition dgxh
  scripts/separate_one.sh --partition dgxh --input-wav "/path/to/song.wav"
  scripts/separate_one.sh --checkpoint "/path/to/model.pth" --input-wav "/path/to/song.wav"
EOF
}

ENV_FILE="scripts/stemmy.env"

PARTITION=""
CHECKPOINT=""
INPUT_WAV=""
DATA_ROOT=""
INPUT_REL=""
OUTPUT_BASE=""
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_FILE="${2:-}"; shift 2 ;;
    --partition) PARTITION="${2:-}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --input-wav) INPUT_WAV="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --input-rel) INPUT_REL="${2:-}"; shift 2 ;;
    --output-base) OUTPUT_BASE="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# Load defaults
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Partition: used only for resolving latest checkpoint name
if [[ -z "${PARTITION}" ]]; then
  PARTITION="${DEFAULT_PARTITION:-}"
fi
if [[ -z "${PARTITION}" ]]; then
  echo "ERROR: --partition required (dgx2|dgxh), or set DEFAULT_PARTITION in scripts/stemmy.env" >&2
  exit 2
fi
if [[ "${PARTITION}" != "dgx2" && "${PARTITION}" != "dgxh" ]]; then
  echo "ERROR: --partition must be dgx2 or dgxh" >&2
  exit 2
fi

ONID="${ONID:-}"
RUNS_BASE="${RUNS_BASE:-}"
if [[ -z "${RUNS_BASE}" && -n "${ONID}" ]]; then
  RUNS_BASE="/nfs/hpc/share/${ONID}/stemmy/runs"
fi

# Resolve checkpoint
if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -z "${RUNS_BASE}" ]]; then
    echo "ERROR: --checkpoint not provided and RUNS_BASE is not set (set RUNS_BASE in scripts/stemmy.env)" >&2
    exit 2
  fi
  CHECKPOINT="${RUNS_BASE}/latest_best_${PARTITION}.pth"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

# Resolve input wav
if [[ -n "${INPUT_WAV}" ]]; then
  if [[ ! -f "${INPUT_WAV}" ]]; then
    echo "ERROR: input wav not found: ${INPUT_WAV}" >&2
    exit 2
  fi
else
  DATA_ROOT="${DATA_ROOT:-}"
  INPUT_REL="${INPUT_REL:-}"

  if [[ -z "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT missing. Set DATA_ROOT in scripts/stemmy.env or pass --data-root" >&2
    exit 2
  fi
  if [[ -z "${INPUT_REL}" ]]; then
    echo "ERROR: INPUT_REL missing. Set INPUT_REL in scripts/stemmy.env or pass --input-rel" >&2
    exit 2
  fi
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: data root not found: ${DATA_ROOT}" >&2
    exit 2
  fi

  INPUT_WAV="${DATA_ROOT%/}/${INPUT_REL}"
  if [[ ! -f "${INPUT_WAV}" ]]; then
    echo "ERROR: derived input wav not found: ${INPUT_WAV}" >&2
    exit 2
  fi
fi

OUTPUT_BASE="${OUTPUT_BASE:-${SEPARATE_OUT_BASE:-runs/separated_cli}}"
DEVICE="${DEVICE:-cpu}"

if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" && "${DEVICE}" != cuda:* ]]; then
  echo "ERROR: --device must be cpu|cuda|cuda:N (got: ${DEVICE})" >&2
  exit 2
fi

mkdir -p "${OUTPUT_BASE}"

python -m stemmy.tool.cli separate \
  --input-file "${INPUT_WAV}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_BASE}" \
  --device "${DEVICE}"

