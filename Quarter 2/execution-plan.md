# Quarter 2 Execution Plan

**Document #5 in Quarter 2 Planning Series**
**Date:** 2026-01-10
**Purpose:** Practical 10-week plan to achieve a working stem separator by end of Quarter 2
**Goal:** Trained model that separates stems well (SDR > 6 dB target)

---

## Executive Summary

**Great news:** Quarter 1 delivered MORE than expected. We have:
- A complete U-Net architecture with train.py ready for MUSDB18
- WandB infrastructure set up with dataset uploaded
- A Gradio web interface for demo/comparison (U-Net 2)

**Quarter 2 focus:** Train, evaluate, iterate, and ship a working model.

---

## Current Assets Inventory

### U-Net 1 (Primary - Training Ready)

| Component | Status | Location |
|-----------|--------|----------|
| U-Net Architecture | Complete | `/AI-Stem-Separation/U-Net 1/models/unet.py` |
| Encoder/Decoder | Complete | `/AI-Stem-Separation/U-Net 1/models/encoder.py`, `decoder.py` |
| Training Script | Complete | `/AI-Stem-Separation/U-Net 1/train.py` |
| MUSDB18 Dataset Class | Complete | Built into train.py |
| Loss Functions | Complete | `/AI-Stem-Separation/U-Net 1/utils/losses.py` |
| Checkpointing | Complete | Built into train.py |
| Early Stopping | Complete | Built into train.py |
| Data Augmentation | Complete | Built into train.py |
| Inference/Separation | Complete | Built into train.py |

**Model Specs:**
- Parameters: ~1.94M (7.42 MB)
- Input: Stereo spectrogram (batch, 2, freq, time)
- Output: 4 stems x stereo (batch, 8, freq, time)
- Stems: drums, bass, vocals, other
- Audio: 44.1kHz 16-bit stereo

**All Q1 tests passing:**
- Shape validation
- Gradient flow
- Overfitting test (58.3% loss reduction)
- Save/load verification

### U-Net 2 (Secondary - Demo/Comparison)

| Component | Status | Location |
|-----------|--------|----------|
| Gradio Web Interface | Complete | `/AI-Stem-Separation/U-Net 2/app.py` |
| Pre-trained Model | Available | HuggingFace (theadityamittal/music-separator-unet) |
| CLI Interface | Complete | Built into app.py |

**Use case:** Baseline comparison, demo, UI patterns for future Q3 deployment.

**Note:** This uses a DIFFERENT U-Net architecture (6.2M params, mono 16kHz) and has poor metrics (SDR: -0.14 dB). Our U-Net 1 architecture should outperform this.

### WandB Infrastructure

| Component | Status | Details |
|-----------|--------|---------|
| Team | Created | `brooksc3-oreogn-state` |
| Project | Created | `stem-separator` |
| Dataset Artifact | Uploaded | `musdb18-hq:v0` (29GB, 150 tracks) |
| Storage | Academic Plan | 200GB free (29GB used) |

**Dashboard:** https://wandb.ai/brooksc3-oreogn-state/stem-separator

---

## Quarter 2 Week-by-Week Plan

### Phase 1: Quality Enhancement Research & Integration (Weeks 1-2)

**Goal:** Understand HOW to achieve high-quality separation, then integrate wandb

#### Week 1: Training Quality Enhancement Research

**Why this comes first:** Before blindly training, we need to understand what makes stem separation models produce high-quality output. This research informs all subsequent training decisions.

**Research Topics:**

1. **Loss Functions for Audio Quality**
   - L1 vs L2 vs combined losses
   - Multi-resolution STFT loss (time-frequency trade-off)
   - Perceptual losses (comparing spectral features, not raw values)
   - Phase-aware losses (important for reconstruction)

2. **Architecture Enhancements**
   - Attention mechanisms (focus on important frequency bands)
   - Deeper vs wider networks (capacity vs efficiency)
   - Skip connection variations (concatenation vs addition)
   - Frequency-aware convolutions

3. **Training Techniques**
   - Learning rate warmup (prevents early divergence)
   - Gradient clipping (stability)
   - Batch size vs learning rate scaling
   - Data augmentation strategies specific to audio:
     - Random gain per stem
     - Time stretching
     - Pitch shifting
     - Mixing stems from different songs
   - Curriculum learning (start with "easy" songs)

4. **State-of-the-Art Review**
   - What does Demucs do? (Facebook's SOTA)
   - What does Spleeter do? (Deezer's popular model)
   - What does Open-Unmix do? (Open source baseline)
   - What can we learn from U-Net 2's approach?

**Deliverables:**
- [ ] Quality enhancement research document
- [ ] List of techniques to implement (prioritized)
- [ ] Modified loss function strategy
- [ ] Data augmentation plan

**Key Resources:**
- [Demucs Paper](https://arxiv.org/abs/2111.03600) - Hybrid architecture, multi-domain
- [Open-Unmix Paper](https://arxiv.org/abs/1911.13254) - LSTM-based reference
- [Wave-U-Net Paper](https://arxiv.org/abs/1806.03185) - Waveform domain
- U-Net 2 codebase - See what techniques were attempted

---

#### Week 2: WandB Integration + First Training Run

**Tasks:**
1. Add wandb logging to U-Net 1 train.py
2. **Implement quality enhancements from Week 1 research:**
   - Add multi-resolution STFT loss option
   - Enhance data augmentation
   - Add learning rate warmup
3. Download MUSDB18-HQ from wandb artifact
4. Run first training experiment with enhanced settings

**Deliverables:**
- [ ] `train.py` logs to wandb (loss curves, learning rate, system metrics)
- [ ] Dataset downloads from wandb artifact successfully
- [ ] First wandb run visible on dashboard

**Code Changes to train.py:**
```python
# Add at start of training
import wandb
run = wandb.init(project="stem-separator", entity="brooksc3-oreogn-state", config=config)

# Get dataset from wandb
artifact = run.use_artifact("musdb18-hq:v0")
data_dir = artifact.download()

# In training loop
wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

# Save model to wandb
if is_best:
    model_artifact = wandb.Artifact(f"unet-{run.id}", type="model")
    model_artifact.add_file(checkpoint_path)
    run.log_artifact(model_artifact)
```

#### Week 2: First Full Training Run

**Tasks:**
1. Run training on full MUSDB18-HQ dataset (100 train songs)
2. Monitor for issues (memory, convergence, data loading)
3. Fix any bugs that emerge
4. Document baseline hyperparameters

**Deliverables:**
- [ ] First complete training run (target: 20+ epochs)
- [ ] Training curves showing loss decreasing
- [ ] Documented any issues encountered
- [ ] Initial hyperparameter baseline documented

**Expected Training Time:**
- MUSDB18-HQ: 100 train songs, ~150 segments each = ~15,000 training samples
- At batch_size=4, lr=1e-3: estimate 30-60 min/epoch on GPU
- 20 epochs = 10-20 hours total

---

### Phase 2: Baseline Training & Debugging (Weeks 3-4)

**Goal:** Achieve consistent training convergence and reasonable initial results

#### Week 3: Training Debugging

**Tasks:**
1. Analyze training curves from Week 2
2. Identify issues: overfitting, underfitting, instability
3. Tune learning rate (try 1e-3, 3e-4, 1e-4)
4. Tune batch size (try 4, 8, 16)
5. Run multiple short experiments to find stable config

**Deliverables:**
- [ ] Documented learning rate sweep results
- [ ] Identified optimal batch size for available GPU memory
- [ ] Training stable for 50+ epochs

**Debugging Checklist:**
- Loss not decreasing? Check learning rate (too high/low)
- Loss exploding? Add gradient clipping
- Overfitting quickly? Add augmentation, reduce model capacity
- GPU OOM? Reduce batch size, segment length

#### Week 4: Extended Training

**Tasks:**
1. Run extended training with tuned hyperparameters (100+ epochs)
2. Implement learning rate scheduling if not converging
3. Save intermediate checkpoints for comparison
4. Listen to first separated outputs!

**Deliverables:**
- [ ] Model trained for 100+ epochs
- [ ] Multiple checkpoints saved (epoch 25, 50, 75, 100)
- [ ] First audio outputs generated and listened to
- [ ] Notes on subjective quality

---

### Phase 3: Training Iteration & Hyperparameter Tuning (Weeks 5-6)

**Goal:** Systematically improve model performance through experimentation

#### Week 5: Hyperparameter Sweeps

**Tasks:**
1. Set up wandb hyperparameter sweep
2. Sweep over:
   - Learning rate: [1e-4, 3e-4, 1e-3]
   - Loss function: [l1, mse, combined]
   - Scheduler: [reduce_on_plateau, cosine]
3. Analyze sweep results
4. Select best configuration

**Deliverables:**
- [ ] Wandb sweep created and run
- [ ] Best hyperparameter combination identified
- [ ] Sweep analysis documented

**Sweep Configuration:**
```yaml
# sweep.yaml
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    values: [0.0001, 0.0003, 0.001]
  loss_type:
    values: ["l1", "mse", "combined"]
  batch_size:
    values: [4, 8, 16]
  scheduler:
    values: ["reduce_on_plateau", "cosine"]
```

#### Week 6: Final Training Run

**Tasks:**
1. Train with optimal hyperparameters for full duration (200+ epochs)
2. Implement early stopping with patience=20
3. Save best model checkpoint
4. Generate separated outputs for test set

**Deliverables:**
- [ ] Final trained model (best_model.pth)
- [ ] Full training history logged to wandb
- [ ] Separated outputs for all 50 test songs

---

### Phase 4: Evaluation & Analysis (Weeks 7-8)

**Goal:** Objectively measure and understand model performance

#### Week 7: SDR/SIR/SAR Evaluation

**Tasks:**
1. Install museval: `pip install museval`
2. Run evaluation on MUSDB18 test set
3. Compute per-song and aggregate metrics
4. Compare to baselines (Spleeter, Open-Unmix, U-Net 2)

**Deliverables:**
- [ ] SDR computed for all stems (drums, bass, vocals, other)
- [ ] Comparison table with baseline models
- [ ] Per-song analysis (best/worst performers)

**Evaluation Script:**
```python
import museval
import numpy as np

results = []
for song in test_songs:
    estimates = separate_stems(model, song)
    references = load_references(song)

    scores = museval.evaluate(references, estimates)
    results.append({
        'song': song,
        'SDR': scores['SDR'].mean(),
        'SIR': scores['SIR'].mean(),
        'SAR': scores['SAR'].mean()
    })

# Aggregate
print(f"Mean SDR: {np.mean([r['SDR'] for r in results]):.2f} dB")
```

**Target Metrics:**
| Level | SDR | Notes |
|-------|-----|-------|
| Pass | > 3 dB | Model is learning |
| Target | > 6 dB | Good separation |
| Excellent | > 8 dB | Publication quality |

#### Week 8: Listening Tests & Failure Analysis

**Tasks:**
1. Set up blind listening test (A/B comparison)
2. Conduct listening tests with team members
3. Identify failure modes:
   - Genre-specific issues?
   - Frequency-specific issues?
   - Stem-specific issues (vocals vs drums)?
4. Log audio examples to wandb

**Deliverables:**
- [ ] Listening test results documented
- [ ] Failure modes categorized and documented
- [ ] Audio examples logged to wandb
- [ ] Recommendations for Q3 improvements

**Listening Test Protocol:**
1. Select 10 diverse test songs
2. For each song: play original, then each separated stem
3. Rate 1-5: clarity, artifacts, bleeding
4. Note specific issues heard

---

### Phase 5: Refinement & Documentation (Weeks 9-10)

**Goal:** Polish the model and prepare for Quarter 3

#### Week 9: Model Refinement

**Tasks:**
1. Address top 3 failure modes (if fixable with current architecture)
2. Try data augmentation variations
3. Fine-tune on problematic genres if needed
4. Explore ensemble approaches (if time permits)

**Deliverables:**
- [ ] Refined model checkpoint
- [ ] Documentation of refinement attempts
- [ ] Final SDR measurements

**Potential Quick Wins:**
- Increase augmentation diversity
- Focus training on vocals (usually most important stem)
- Adjust segment length for better frequency resolution

#### Week 10: Documentation & Q3 Planning

**Tasks:**
1. Write Quarter 2 final report
2. Clean and document code
3. Create model card for wandb
4. Plan Quarter 3 priorities

**Deliverables:**
- [ ] Final report with metrics, examples, lessons learned
- [ ] Clean, documented codebase
- [ ] Model card on wandb
- [ ] Q3 roadmap

**Final Report Structure:**
1. Executive Summary
2. Model Architecture
3. Training Process
4. Evaluation Results
5. Failure Analysis
6. Lessons Learned
7. Q3 Recommendations

---

## Logic Pro Stem Splitter Research

**Parallel Task (any week):** Analyze Apple's proprietary stem splitter

**Purpose:** Understand the competition, identify quality benchmarks

**Tasks:**
1. Test Logic Pro stem splitter on same songs
2. Measure SDR if possible (may need to reverse-engineer)
3. Document subjective quality differences
4. Note any features we should target

**Deliverables:**
- [ ] Comparison document: Our model vs Logic Pro
- [ ] Feature gap analysis
- [ ] Recommendations for Q3

---

## Success Criteria

### Minimum (Pass)
- [ ] Model trains without crashing for 50+ epochs
- [ ] Loss curve shows consistent decrease
- [ ] Model produces non-silent output
- [ ] SDR > 3 dB on test set

### Target (A)
- [ ] SDR > 6 dB on test set
- [ ] Listening tests show clear stem isolation
- [ ] Failure modes identified and documented
- [ ] Code is clean and reproducible

### Excellent (A+)
- [ ] SDR > 8 dB on test set
- [ ] Outperforms U-Net 2 baseline
- [ ] Comprehensive evaluation across genres
- [ ] Ready for Q3 optimization

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training doesn't converge | Medium | High | Start with proven hyperparameters, debug early |
| GPU memory issues | Medium | Medium | Start with small batch size, use gradient accumulation |
| Low SDR scores | Medium | Medium | Focus on vocals first, iterate on architecture Q3 |
| Data loading bottleneck | Low | Medium | Use num_workers, pre-compute spectrograms |
| Wandb integration issues | Low | Low | Test with small runs first |

---

## Compute Requirements

**Estimated GPU Hours:**
- Phase 1-2: 40-60 hours (initial training + debugging)
- Phase 3: 60-100 hours (sweeps + final training)
- Phase 4: 20-30 hours (evaluation runs)
- **Total: 120-190 GPU hours**

**Options:**
1. Local GPU (if available)
2. Google Colab Pro (~$10/month)
3. Lambda Labs (~$1.25/hour A10)
4. University computing cluster (free if available)

---

## Quick Reference

### Train Model
```bash
cd "AI-Stem-Separation/U-Net 1"
python train.py --data-dir /path/to/musdb18hq --epochs 100 --batch-size 4
```

### Resume Training
```bash
python train.py --data-dir /path/to/musdb18hq --resume output/checkpoints/latest.pth
```

### Separate Audio
```bash
python train.py --separate song.wav --checkpoint output/checkpoints/best_model.pth
```

### Download Dataset from WandB
```python
import wandb
run = wandb.init(project="stem-separator", entity="brooksc3-oreogn-state")
artifact = run.use_artifact("musdb18-hq:v0")
data_path = artifact.download()
```

---

## Timeline Summary

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 1 | WandB Integration | First wandb run |
| 2 | First Training | 20+ epoch training complete |
| 3 | Debugging | Stable training config |
| 4 | Extended Training | 100+ epoch model |
| 5 | Hyperparameter Sweeps | Best config identified |
| 6 | Final Training | Best model checkpoint |
| 7 | SDR Evaluation | Metrics computed |
| 8 | Listening Tests | Failure analysis |
| 9 | Refinement | Refined model |
| 10 | Documentation | Final report |

---

**Document Generated:** 2026-01-10
**Status:** Ready for Q2 execution
