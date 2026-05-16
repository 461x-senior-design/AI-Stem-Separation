# HPC Run Cheatsheet

All commands below are run **from the HPC cluster** after logging in and
`cd`ing into your checkout of this repo. The `stemmy` binary is the single
entry point and is installed as a console script by `pip install -e .`.

```bash
ssh <onid>@submit.hpc.engr.oregonstate.edu
cd ~/mystemmy                           # or wherever you cloned
source .venv/bin/activate               # ensure the stemmy script is on PATH
stemmy --help
```

---

## 1. Submit a training run

### Named config

```bash
stemmy run <config-name> --name <short-label>
# e.g.
stemmy run recon-p10 --name recon-p10-try1
```

### With CLI overrides (repeatable)

```bash
stemmy run recon-p10 --name recon-p10-lr5e5 \
    -O LR=5e-5 -O BATCH_SIZE=8 -O EPOCHS=120
```

### Dry-run (resolve config + print sbatch script, do not submit)

```bash
stemmy run recon-p10 --name sanity --dry-run
```

### Pick a partition

```bash
stemmy run recon-p10 --name foo --partition dgxh    # default
stemmy run recon-p10 --name foo --partition dgx2
```

---

## 2. List / inspect / compare runs

```bash
stemmy ls                         # all runs, newest first
stemmy ls --sort best_sisdr       # rank by best SI-SDR
stemmy ls --status submitted      # filter by status
stemmy ls --name recon            # substring match on name

stemmy show <run-id>              # config + metrics + log tail
stemmy compare <run-id-a> <run-id-b> [<run-id-c> ...]
```

---

## 3. Configs

Named configs live under `scripts/v2/configs/` — `shared/` is checked in,
`<username>/` is per-user and wins on conflicts.

```bash
stemmy config ls                  # list all configs, show shadowing
stemmy config show recon-p10      # print resolved layer stack
stemmy config new my-variant --from recon-p10
stemmy config new base-v2 --shared
stemmy config edit recon-p10              # user copy
stemmy config edit recon-p10 --shared     # shared copy
```

Resolution order: `scripts/v2/defaults.env` → shared named config → user
named config → CLI `-O KEY=VALUE` overrides.

---

## 4. Matrix sweep (cartesian)

YAML spec under `scripts/v2/matrices/<name>.yaml`:

```yaml
name: lr-sweep
base: recon-p10
partition: dgxh
sweep:
  LR: [1e-4, 5e-5, 1e-5]
  BATCH_SIZE: [8, 12]
```

Submit:

```bash
stemmy matrix scripts/v2/matrices/lr-sweep.yaml
stemmy matrix scripts/v2/matrices/lr-sweep.yaml --dry-run
```
