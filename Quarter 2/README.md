# Quarter 2 Planning Hub

**Project:** AI Stem Separation
**Quarter 2 Goal:** Trained model that separates stems well (SDR > 6 dB)
**Status:** Ready to execute

---

## Quick Status

### What We Have (Quarter 1 Deliverables)

| Asset | Status | Notes |
|-------|--------|-------|
| **U-Net 1** | Complete | Full architecture, train.py ready, all tests passing |
| **U-Net 2** | Complete | Gradio web interface with pre-trained model (for comparison) |
| **WandB Setup** | Complete | Team created, MUSDB18-HQ uploaded (29GB) |
| **Curriculum Docs** | Complete | Q1 curriculum with 150+ files |

### U-Net 1 Details

- **Location:** `/AI-Stem-Separation/U-Net 1/`
- **Parameters:** 1,943,922 (~2M)
- **Model Size:** 7.42 MB
- **Input:** Stereo spectrogram (batch, 2, freq, time)
- **Output:** 4 stems (drums, bass, vocals, other) as stereo
- **Tests:** All 6/6 passing (shapes, gradients, overfitting)

Key files:
```
U-Net 1/
|-- train.py           # Complete training pipeline with MUSDB18 dataset class
|-- models/
|   |-- unet.py        # Full U-Net architecture
|   |-- encoder.py     # Encoder blocks
|   |-- decoder.py     # Decoder blocks
|-- utils/
|   |-- losses.py      # L1, MSE, Combined losses
|   |-- config.py      # Configuration
|-- data/
|   |-- preprocessing.py  # STFT/audio processing
```

### U-Net 2 Details

- **Location:** `/AI-Stem-Separation/U-Net 2/`
- **Purpose:** Demo/comparison baseline
- **Interface:** Gradio web UI + CLI
- **Pre-trained Model:** HuggingFace (theadityamittal/music-separator-unet)
- **Performance:** SDR -0.14 dB (baseline we should beat)

### WandB Infrastructure

- **Team:** `brooksc3-oreogn-state`
- **Project:** `stem-separator`
- **Dashboard:** https://wandb.ai/brooksc3-oreogn-state/stem-separator
- **Dataset:** `musdb18-hq:v0` (29GB, 150 tracks, 44.1kHz 16-bit stereo)
- **Plan:** Academic (200GB free storage)

---

## Documents in This Folder

| # | Document | Purpose |
|---|----------|---------|
| 0 | `0-quarter-transition-audit-report.md` | Initial Q1->Q2 transition audit |
| 1 | `1-repo-sync-and-audit-complete.md` | Repo sync status |
| 2 | `2-site-sync-verification.md` | Astro site verification |
| 3 | `3-terminology-cleanup-complete.md` | Semester->Quarter cleanup |
| 4 | `4-q1-q2-alignment-assessment.md` | Curriculum alignment check |
| **5** | **`5-quarter-2-execution-plan.md`** | **Week-by-week execution plan (START HERE)** |

### Supporting Documents

- `SummarQ1Docs/` - Quarter 1 summary documents (PDD, Design Doc, Requirements)

---

## Quarter 2 Execution Plan Summary

**Duration:** 10 weeks
**End Goal:** Working stem separator with SDR > 6 dB

### Week-by-Week Overview

| Week | Phase | Focus | Key Deliverable |
|------|-------|-------|-----------------|
| 1-2 | Integration | WandB + First Training | First training run complete |
| 3-4 | Debugging | Stable Training | 100+ epoch model |
| 5-6 | Tuning | Hyperparameter Sweeps | Best configuration |
| 7-8 | Evaluation | SDR/Listening Tests | Performance metrics |
| 9-10 | Refinement | Polish + Documentation | Final report |

### Success Criteria

| Level | SDR Target | Description |
|-------|------------|-------------|
| Pass | > 3 dB | Model is learning something |
| Target | > 6 dB | Good separation quality |
| Excellent | > 8 dB | Publication quality |

---

## Getting Started

### 1. Set up environment
```bash
cd "/Users/cameronbrooks/Server/AI STEM SEPARATION/AI-Stem-Separation/U-Net 1"
pip install -r requirements.txt
pip install wandb
```

### 2. Login to WandB
```bash
wandb login
# Paste API key from https://wandb.ai/authorize
```

### 3. Download dataset
```python
import wandb
run = wandb.init(project="stem-separator", entity="brooksc3-oreogn-state")
artifact = run.use_artifact("musdb18-hq:v0")
data_path = artifact.download()
print(f"Dataset downloaded to: {data_path}")
```

### 4. Start training
```bash
python train.py --data-dir /path/to/musdb18hq --epochs 100
```

---

## Key Links

- **WandB Dashboard:** https://wandb.ai/brooksc3-oreogn-state/stem-separator
- **MUSDB18-HQ Info:** https://sigsep.github.io/datasets/musdb.html
- **WandB Docs:** https://docs.wandb.ai/

---

## Next Actions (Immediate)

1. Read `5-quarter-2-execution-plan.md` for detailed week-by-week plan
2. Add wandb integration to train.py (Week 1 task)
3. Run first training experiment
4. Monitor via WandB dashboard

---

## Questions / Notes

*Add notes here as Quarter 2 progresses*

---

**Last Updated:** 2026-01-10
