# Weights & Biases (wandb) Guide

Our team's ML experiment tracking and dataset management hub.

**Team:** `brooksc3-oreogn-state`
**Project:** `stem-separator`
**Dashboard:** https://wandb.ai/brooksc3-oreogn-state/stem-separator

---

## Current State

### Dataset: MUSDB18-HQ

Our training dataset is uploaded and ready:

| Property | Value |
|----------|-------|
| Artifact Name | `musdb18-hq:v0` |
| Size | ~29GB |
| Tracks | 150 (100 train / 50 test) |
| Format | 44.1kHz 16-bit stereo WAV |
| Stems | mixture, drums, bass, vocals, other |

**Access the dataset in code:**
```python
import wandb

run = wandb.init(project="stem-separator", entity="brooksc3-oreogn-state")
artifact = run.use_artifact("musdb18-hq:v0")
data_path = artifact.download()  # Downloads to ./artifacts/musdb18-hq:v0/
```

---

## What wandb Gives Us Going Forward

### 1. Experiment Tracking
Every training run automatically logs:
- Loss curves (train/val)
- Learning rate schedules
- Hyperparameters
- System metrics (GPU usage, memory)

```python
wandb.log({"train_loss": loss, "val_loss": val_loss, "epoch": epoch})
```

### 2. Model Checkpoints as Artifacts
Save and version trained models:
```python
model_artifact = wandb.Artifact(f"unet-{run.id}", type="model")
model_artifact.add_file("best_model.pth")
run.log_artifact(model_artifact)
```

### 3. Compare Experiments
- Side-by-side loss curves
- Hyperparameter sweeps
- Find what works best

### 4. Team Collaboration
- Everyone sees all experiments
- Comment on runs
- Share findings via links

### 5. Audio Logging (for listening tests)
```python
wandb.log({"separated_vocals": wandb.Audio(audio_array, sample_rate=44100)})
```

### 6. Hyperparameter Sweeps
Automated search for best settings:
```yaml
# sweep.yaml
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [4, 8, 16]
```

---

## Quick Reference

### Login
```bash
wandb login
# Paste API key from https://wandb.ai/authorize
```

### Start a Training Run
```python
import wandb

run = wandb.init(
    project="stem-separator",
    entity="brooksc3-oreogn-state",
    config={
        "model": "unet",
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 4
    }
)

# Training loop
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

run.finish()
```

### Download Dataset
```python
artifact = run.use_artifact("brooksc3-oreogn-state/stem-separator/musdb18-hq:v0")
data_path = artifact.download()
```

### Upload a New Artifact (CLI)
```bash
wandb artifact put ./my_data --name "brooksc3-oreogn-state/stem-separator/artifact-name" --type dataset
```

### View Runs
- Web UI: https://wandb.ai/brooksc3-oreogn-state/stem-separator
- CLI: `wandb sync` (sync offline runs)

---

## Integrating with U-Net 1 Training

The `U-Net 1/train.py` can be modified to use wandb. Key integration points:

```python
# At the start of training
run = wandb.init(project="stem-separator", entity="brooksc3-oreogn-state", config=config)

# Get dataset from wandb instead of local path
artifact = run.use_artifact("musdb18-hq:v0")
data_dir = artifact.download()

# In training loop (after each epoch)
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "epoch": epoch
})

# When saving best model
if is_best:
    model_artifact = wandb.Artifact(f"unet-{run.id}", type="model")
    model_artifact.add_file(checkpoint_path)
    run.log_artifact(model_artifact)

# At end
run.finish()
```

---

## Team Access

All team members at `brooksc3-oreogn-state` have access to:
- All logged experiments
- All artifacts (datasets, models)
- Project dashboard

To invite new members: Settings → Team → Invite

---

## Useful Links

- [Project Dashboard](https://wandb.ai/brooksc3-oreogn-state/stem-separator)
- [Artifacts Browser](https://wandb.ai/brooksc3-oreogn-state/stem-separator/artifacts)
- [wandb Docs](https://docs.wandb.ai/)
- [Python SDK Reference](https://docs.wandb.ai/ref/python/)
- [CLI Reference](https://docs.wandb.ai/ref/cli/)

---

## Storage & Limits

We're on the **Academic Plan** (free):
- 200GB storage
- 100 team seats
- All Pro features

Current usage: ~29GB (MUSDB18-HQ dataset)
