#!/usr/bin/env bash
# Preflight checks before submitting an HPC sweep.
#
# Catches the failures we've seen so far without paying queue-wait:
#   1. bash syntax errors in _train_inner.sh
#   2. SyntaxErrors in any Python heredoc inside _train_inner.sh (the `cry:` bug)
#   3. Click / CLI surface regressions in `stemmy dev train --help`
#   4. Broken matrix YAML (`stemmy matrix --dry-run`)
#   5. Dataset+model wiring on a synthetic 3-track MUSDB tree, 1 CPU epoch, augs on
#
# Run from the repo root:  scripts/v2/preflight.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PASS() { printf "  \033[32mOK\033[0m %s\n" "$1"; }
FAIL() { printf "  \033[31mFAIL\033[0m %s\n" "$1"; exit 1; }
STEP() { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }

STEP "1/5 bash -n _train_inner.sh"
bash -n scripts/v2/_train_inner.sh && PASS "no bash syntax errors" \
  || FAIL "bash syntax errors above"

STEP "2/5 Python heredoc compile check"
python3 - <<'PY'
import re, pathlib, sys
src = pathlib.Path("scripts/v2/_train_inner.sh").read_text()
pattern = re.compile(r"python\s+-\s+<<\s*(\w+)\n(.*?)\n\1\b", re.DOTALL)
blocks = pattern.findall(src)
if not blocks:
    print("  WARN no python heredocs found")
    sys.exit(0)
ok = True
for i, (tag, body) in enumerate(blocks):
    # Replace ${VAR} placeholders with dummy literal so compile() doesn't choke.
    sanitized = re.sub(r"\$\{[^}]+\}", "1", body)
    try:
        compile(sanitized, f"<heredoc#{i}:{tag}>", "exec")
    except SyntaxError as e:
        print(f"  FAIL heredoc #{i} ({tag}): {e}")
        ok = False
sys.exit(0 if ok else 1)
PY
PASS "$(grep -c '^python - <<' scripts/v2/_train_inner.sh) Python heredoc(s) compile"

STEP "3/5 stemmy CLI surface"
stemmy --help > /dev/null && PASS "stemmy --help"
stemmy dev train --help > /dev/null && PASS "stemmy dev train --help"
HELP_OUT="$(stemmy dev train --help)"
for flag in --seed --aug-gain-p --aug-pitch-p --aug-time-stretch-p \
            --aug-shift-p --aug-polarity-p --aug-noise-p; do
  echo "$HELP_OUT" | grep -q -- "$flag" || FAIL "flag $flag missing from stemmy dev train"
done
PASS "all --aug-* flags + --seed registered"

STEP "4/5 matrix dry-runs"
for yaml in scripts/v2/matrices/aug-sweep-screening.yaml \
            scripts/v2/matrices/aug-sweep-ablation.yaml; do
  stemmy matrix --dry-run "$yaml" > /dev/null && PASS "$yaml"
done

STEP "5/5 end-to-end CPU training (synthetic 3-track MUSDB, 1 epoch, augs on)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 - <<PY
from pathlib import Path
import numpy as np, soundfile as sf
root = Path("$TMP/data")
SR, N = 44100, 44100 * 8
for split in ("train", "test"):
    for i in range(3):
        d = root / split / f"track{i:02d}"
        d.mkdir(parents=True)
        rng = np.random.default_rng(seed=i)
        stems = {s: 0.1 * rng.standard_normal((N, 2)).astype(np.float32)
                 for s in ("drums","bass","vocals","other")}
        mix = sum(stems.values())
        for name, w in [("mixture", mix)] + list(stems.items()):
            sf.write(d / f"{name}.wav", w, SR, subtype="FLOAT")
PY

# Disable wandb so preflight doesn't require an API key
export WANDB_MODE=disabled
export SPLIT_SOURCE=random
export VALID_FRACTION=0.34
export TRAIN_SPLIT_SEED=0
export STEMMY_DISABLE_PROGRESS=1

stemmy dev train \
  --data-root "$TMP/data" \
  --device cpu \
  --epochs 1 --batch-size 1 --num-workers 0 \
  --max-tracks 1 --time-frames 256 \
  --base-channels 16 \
  --save-every-epochs 1 \
  --checkpoint-dir "$TMP/ckpt" \
  --seed 42 \
  --aug-gain-p 0.5 --aug-pitch-p 0.5 --aug-shift-p 0.5 \
  --potato \
  > "$TMP/train.log" 2>&1 \
  || { echo "  FAIL: training crashed. Last 40 lines:"; tail -40 "$TMP/train.log"; exit 1; }
PASS "1-epoch CPU train completed with augs active"

printf "\n\033[1;32m✓ ALL PREFLIGHT CHECKS PASSED\033[0m — safe to submit to HPC.\n"
