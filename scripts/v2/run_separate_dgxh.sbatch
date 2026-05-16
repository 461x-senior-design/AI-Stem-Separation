#!/usr/bin/env bash
#SBATCH -J stemmy_sep
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --export=ALL
#SBATCH -o /nfs/hpc/share/%u/stemmy/runs/slurm-sep-%j.out
#SBATCH -e /nfs/hpc/share/%u/stemmy/runs/slurm-sep-%j.err
set -euo pipefail
cd /nfs/stak/users/$USER/mystemmy
source .venv/bin/activate
scripts/separate_one.sh --partition dgxh
