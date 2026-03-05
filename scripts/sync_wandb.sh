#!/usr/bin/env bash
# scripts/sync_wandb.sh
# Pull wandb offline runs from the HPC and sync them to wandb.ai.
#
# Usage:
#   scripts/sync_wandb.sh <onid>
#
# Reads HPC_HOST from environment (default: submit.hpc.oregonstate.edu).
# Reads HPC_STEMMY_DIR from environment (default: /nfs/hpc/share/<onid>/stemmy).
#
# Example:
#   scripts/sync_wandb.sh doej
#   HPC_HOST=login.hpc.oregonstate.edu scripts/sync_wandb.sh doej
set -euo pipefail

ONID="${1:-}"
if [[ -z "${ONID}" ]]; then
  echo "Usage: $0 <onid>" >&2
  exit 1
fi

HPC_HOST="${HPC_HOST:-submit.hpc.engr.oregonstate.edu}"
HPC_STEMMY_DIR="${HPC_STEMMY_DIR:-/nfs/hpc/share/${ONID}/stemmy}"
LOCAL_WANDB_DIR="${LOCAL_WANDB_DIR:-${HOME}/stemmy-wandb}"

echo "=== Pulling wandb offline runs from HPC ==="
echo "  Source: ${ONID}@${HPC_HOST}:${HPC_STEMMY_DIR}/wandb/"
echo "  Dest:   ${LOCAL_WANDB_DIR}"
echo

mkdir -p "${LOCAL_WANDB_DIR}"
rsync -av --progress \
  "${ONID}@${HPC_HOST}:${HPC_STEMMY_DIR}/wandb/" \
  "${LOCAL_WANDB_DIR}/"

echo
echo "=== Syncing to wandb.ai ==="
wandb sync --sync-all "${LOCAL_WANDB_DIR}"

echo
echo "Done."
