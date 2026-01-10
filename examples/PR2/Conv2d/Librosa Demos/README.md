# U-Net Component Demos (Interactive Audio Learning)

This folder contains interactive audio demos that illustrate the core building blocks of a U-Net architecture **without building an actual neural network**. Each script lets you **hear and see** how specific architectural components affect a music mixture when trying to isolate a target stem (like vocals).

## 📚 How to Use These Demos

**Your Goal:** Use these tools to audibly and visually understand what each U-Net component does. Then, experiment to see if you can improve vocal isolation beyond the baseline.

**Prerequisites:**
- Run `best.py` first (from parent directory) to generate baseline results
- All demos expect audio files in `../rtg/100-window/`

---

## 📂 Organization

### 🔹 Baseline (Run First)
Located in parent directory: `../best.py`

**What it shows:** How a series of Conv2D "feature extractors" (like edge detectors for audio) can be applied sequentially to pull vocals out of a mix.

**Try this:**
- Listen to `output_progressive_8slices/1_slices/` through `8_slices/`
- Find where it sounds best vs. worst
- Why does performance degrade after 8 slices?

---

### 🔹 1-Component-Demos/ (Isolated Concepts)

These demos isolate **individual U-Net architectural components** so you can understand each one independently.

#### **Encoder-Decoder/**
- `freq_time_resolution_demo.py` - Core encoder/decoder behavior

**What it shows:** What happens when you drastically reduce the time/frequency resolution of audio (encoder bottleneck) and then try to restore it (decoder).

**Try this:**
- Does up-sampling restore lost detail?
- Why or why not?
- What information is permanently lost?

---

#### **SkipConnection/**
- `skip_connection_impact_demo.py` - Pure skip connection demonstration

**What it shows:** How adding back original high-res detail (a "skip connection") to a blurry, low-res processed signal can restore clarity.

**Try this:**
- Compare "bottleneck only" vs. "bottleneck + skip"
- What detail is restored?
- Can you hear the difference in vocal clarity?

---

#### **ProcessingStyle/**
- `masking_vs_addition_demo.py` - Two approaches to feature application

**What it shows:** Two ways to apply features: adding energy (EQ boost) vs. masking (attenuating parts of the signal).

**Try this:**
- Which sounds more like "isolation" vs. "enhancement"?
- When would you use each approach?
- Can you combine them effectively?

---

#### **ContextualRefinement/**
- `contextual_refinement_demo.py` - Bottleneck context effects

**What it shows:** How processing a signal after it's been compressed (in a bottleneck) yields different results than processing the original directly.

**Try this:**
- Does the "context" of the bottleneck help or hurt refinement?
- Why would U-Net process at different resolutions?

---

### 🔹 proto-unet-demos/ (Focused U-Net Architecture Demos)

**Advanced U-Net component demonstrations** with more architectural focus. These demos use simplified proto-U-Net implementations to isolate specific architectural features.

**Organized by component:**
- **SkipConnection/** - `demo1_skip_connections.py` - WITH vs WITHOUT skip connections comparison
- **Encoder/** - `demo2_encoder_hierarchy.py` - Multi-scale feature extraction pyramid
- **Decoder/** - `demo3_decoder_reconstruction.py` - 4 upsampling strategies comparison
- **BatchNorm/** - `demo4_batchnorm_stability.py` - Training stabilization demonstration

**See:** `proto-unet-demos/README.md` for complete documentation

**Try this:**
- Run these demos to understand U-Net architectural components in isolation
- Compare different upsampling strategies (nearest, bilinear, transposed conv, sub-pixel)
- See how BatchNorm affects training stability and gradient flow

---

### 🔹 2-Chaining-Demos/ (Combined Concepts)

These demos **combine multiple components** to simulate full U-Net processing pipelines.

#### **Post-Processing Chains** (Take best.py output → apply component)
- `chain_best_to_addition.py` - Best output → additive adjustment
- `chain_best_to_resolution.py` - Best output → resolution change
- `chain_best_to_skip.py` - Best output → skip connection

**Try this:** Does post-processing help or hurt an already-refined signal?

---

#### **Full U-Net Simulation Chains**
- `chain_resolution_then_skip.py` - Compress → process → upsample → skip connection
  - **This is the closest to a real U-Net decoder block**
- `chain_context_then_resolution.py` - Context processing → resolution refinement

**Try this:** How close can you get to the target vocal using only these components?

---

#### **Multi-Stage Processing**
- `chain_addition_then_masking.py` - Sequential processing styles

**Try this:** Do different processing styles complement or conflict with each other?

---

## 🎯 Team Challenges & Experiments

### 1. **Find the "Sweet Spot"**
In `best.py`, the best result was at 8 slices.
- Why did adding more slices make it worse?
- Can you manually design a better sequence of 8 slices?

### 2. **Improve the Chain**
Use `chain_best_to_skip.py` but replace the "original mixture" detail in the skip with another refined signal (e.g., the output of the horizontal slice).
- Does that work better?

### 3. **Fix the Bottleneck**
In `freq_time_resolution_demo.py`, the upsampled audio is blurry.
- Can you design a smarter upsampling method (e.g., using Conv2D features) to restore detail?

### 4. **Explore BatchNorm** ✅
See `proto-unet-demos/BatchNorm/demo4_batchnorm_stability.py`:
- Demonstrates how BatchNorm stabilizes training
- Shows gradient flow improvements
- Analyzes activation distributions across layers

### 5. **Four-Stem Prep**
Right now, all demos use a vocal target.
- How would you adapt `masking_vs_addition_demo.py` to isolate drums or bass instead?
- What features would you target?

---

## 🚀 Quick Start

```bash
# 1. Run baseline first (from parent directory)
cd ..
python best.py --slices 8

# 2. Explore individual audio processing components
cd "Librosa Demos/1-Component-Demos/Encoder-Decoder"
python freq_time_resolution_demo.py

cd "../SkipConnection"
python skip_connection_impact_demo.py

# 3. Explore U-Net architectural components
cd "../proto-unet-demos"
python SkipConnection/demo1_skip_connections.py
python Encoder/demo2_encoder_hierarchy.py
python Decoder/demo3_decoder_reconstruction.py
python BatchNorm/demo4_batchnorm_stability.py

# 4. Try chaining demos
cd "../2-Chaining-Demos"
python chain_resolution_then_skip.py
python chain_best_to_skip.py
```

---

## 📊 What to Listen For

**Good Vocal Isolation:**
- ✅ Clear vocal intelligibility
- ✅ Minimal drum/bass bleed
- ✅ Natural timbre (not robotic/phasey)

**Problem Signs:**
- ❌ Muffled/underwater sound
- ❌ Excessive artifacts/distortion
- ❌ Lost consonants or breathiness

---

## 🔬 Understanding U-Net Through Audio

These demos teach U-Net concepts through **direct audio manipulation**:

| Demo Category | U-Net Component | What You Learn |
|---------------|-----------------|----------------|
| Encoder-Decoder | Downsampling/Upsampling | Information loss in bottleneck |
| SkipConnection | Skip connections | How to preserve detail |
| ProcessingStyle | Learned filters | Different ways to apply features |
| ContextualRefinement | Hierarchical processing | Why multi-scale matters |
| Chaining | Full architecture | How components work together |

**Key Insight:** If you can't make it work with hand-crafted filters, a U-Net won't magically fix it. But if you *can* make it work manually, a U-Net can **learn to do it better** and **generalize to new songs**.

---

## 📖 Next Steps

After completing these demos:
1. **Understand the architecture** - You now know what each U-Net component does audibly
2. **Design your model** - Use insights to architect a U-Net for 4-stem separation
3. **Train on real data** - Replace hand-crafted filters with learned Conv2D filters
4. **Scale up** - Apply to full-length songs, multiple stems, production quality

The insights from these demos directly inform U-Net design:
- Focus on essential feature extractors (first 8 slices from best.py)
- Use dB-space processing for numerical stability
- Optimize for ~100 time windows (hop_length=1040)
- 400-point frequency representation is sufficient for bottleneck

---

**Happy experimenting!** 🎧
