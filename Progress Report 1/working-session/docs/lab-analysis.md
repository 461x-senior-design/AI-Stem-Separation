# Lab Analysis: Mapping Team Work to 5 Mini-Labs

## Team Member Contributions Summary

### 1. Cameron (Team Leader) - Astro Tutorial POC
**Focus:** Manual vocal separation via spectral fingerprinting
- **Key Technologies:** librosa, NumPy, STFT/ISTFT
- **Approach:** 18-slice multi-scale analysis (~765,000 measurements)
- **Results:** 70-80% vocal separation quality on "Intergalactic"
- **Website:** https://brookcs3.github.io/461-Design-and-Development-/
- **Key Insight:** Validates concept before U-Net investment

### 2. cervanj2 - Architecture Description
**Focus:** System architecture and pipeline design
- **Architecture:** 5-layer pipeline
  1. User Interface Layer
  2. Preprocessing Layer (librosa, NumPy, STFT)
  3. Inference Layer (PyTorch, U-Net)
  4. Post-processing Layer (librosa, NumPy, inverse STFT)
  5. Output Layer
- **Key Insight:** Clean separation of concerns

### 3. Ryan O'Rourke - PyTorch "Learn the Basics"
**Focus:** PyTorch fundamentals
- **Hardware:** NVIDIA GeForce RTX 2070 Super Max-Q
- **Topics Covered:** Tensors, DataLoader, autograd, training loops
- **Dataset:** Fashion-MNIST
- **Model:** Flatten → Linear → ReLU stack
- **Results:** 41% → 71% accuracy progression
- **Key Insight:** Hands-on PyTorch training experience

### 4. Haedon - Software Design Philosophy
**Focus:** Design principles and complexity management
- **Key Concepts:**
  - Managing complexity (change amplification, cognitive load, unknown unknowns)
  - Deep modules with simple interfaces
  - Strategic programming (invest 10-20% in design)
  - Information hiding and modularity
  - "Design it twice" approach
- **Key Insight:** Software engineering best practices for maintainable code

### 5. Yovannoa (Austin) - PyTorch "60 Minute Blitz"
**Focus:** Comprehensive PyTorch neural network training
- **Setup:** Ubuntu WSL, CUDA 13.0, Jupyter Lab
- **4 Mini-Tutorials Completed:**
  1. Tensors (initialization, operations, GPU usage)
  2. torch.autograd (automatic differentiation, forward/backward propagation)
  3. Neural Networks (torch.nn, Net class, loss functions)
  4. Training a Classifier (full training loop, 54% accuracy on CIFAR-10)
- **Key Insight:** Complete neural network training pipeline

---

## Required Lab Topics (from Astro Tutorial)

1. **U-Net Architecture**
2. **librosa (Audio Processing)**
3. **NumPy (Numerical Operations)**
4. **PyTorch (Deep Learning Framework)**
5. **Data Integration and Preparation**

---

## Lab Mapping Strategy

### Lab 1: Data Integration and Preparation
**Primary Contributors:** Cameron (Astro POC), Yovannoa (Classifier training)
**Connection to Team Work:**
- Cameron's spectral fingerprinting shows manual data preparation process
- Yovannoa's Training a Classifier section shows dataset loading and transformation
- cervanj2's architecture identifies preprocessing as a distinct layer

**"Hello World" Concept:** Load an audio file, convert to spectrogram, visualize
**Build Toward Astro:** Sets up the data pipeline that feeds into U-Net

---

### Lab 2: NumPy for Audio Processing
**Primary Contributors:** Cameron (Astro POC), cervanj2 (Architecture)
**Connection to Team Work:**
- Cameron's POC heavily uses NumPy for 765,000 measurements
- cervanj2's preprocessing/post-processing layers specify NumPy
- Both Ryan and Yovannoa show NumPy interoperability with PyTorch

**"Hello World" Concept:** Create and manipulate a simple array, perform basic audio math
**Build Toward Astro:** NumPy operations that underlie STFT/ISTFT transformations

---

### Lab 3: librosa Fundamentals
**Primary Contributors:** Cameron (Astro POC), cervanj2 (Architecture)
**Connection to Team Work:**
- Cameron's POC uses librosa for STFT/ISTFT pipeline
- cervanj2's preprocessing layer explicitly mentions librosa
- This is the bridge between raw audio and machine learning

**"Hello World" Concept:** Load audio file with librosa, compute and display STFT
**Build Toward Astro:** The exact preprocessing Cameron used in the POC

---

### Lab 4: PyTorch Essentials
**Primary Contributors:** Ryan (Learn the Basics), Yovannoa (60 Minute Blitz)
**Connection to Team Work:**
- Ryan covered tensors, DataLoader, autograd, basic training loop
- Yovannoa covered tensors, autograd, torch.nn, full classifier training
- cervanj2's inference layer specifies PyTorch and U-Net
- Both learned on image classification (Fashion-MNIST, CIFAR-10)

**"Hello World" Concept:** Create tensors, move to GPU, simple forward pass
**Build Toward Astro:** PyTorch foundation needed for U-Net training

---

### Lab 5: U-Net Architecture for Audio Separation
**Primary Contributors:** Cameron (Astro POC goal), cervanj2 (Architecture inference layer)
**Connection to Team Work:**
- Cameron's POC validates the concept before U-Net training
- cervanj2's architecture places U-Net at the inference layer core
- Ryan and Yovannoa's PyTorch knowledge provides the framework
- Haedon's design philosophy applies to U-Net implementation

**"Hello World" Concept:** Explore forked U-Net repo, understand encoder-decoder structure
**Build Toward Astro:** The actual U-Net model for vocal separation

---

## Lab Sequencing Rationale

**Progressive Build:**
1. **Data** → Start with what goes into the system
2. **NumPy** → Understand the numerical foundation
3. **librosa** → Learn audio-specific transformations
4. **PyTorch** → Master the deep learning framework
5. **U-Net** → Integrate everything into the target architecture

**Team Alignment:**
- Each lab connects to multiple team members' work
- Progression follows cervanj2's 5-layer architecture conceptually
- Builds from Cameron's manual approach toward automated U-Net
- Leverages Ryan and Yovannoa's PyTorch foundations
- Incorporates Haedon's design thinking throughout

---

## Design Principles (from Haedon's Work)

Apply throughout all labs:
- **Deep modules:** Simple interfaces, hide complexity
- **Strategic programming:** Invest time in clean, understandable examples
- **Cognitive load reduction:** One clear concept per lab
- **Progressive disclosure:** Start simple, add complexity gradually
- **Clear naming:** Precise, self-documenting variable and function names

---

## Next Steps

1. Create detailed outlines for each of the 5 labs
2. Ensure each lab:
   - Is 3-5 minutes as a demo
   - Has a "Hello World" equivalent
   - Builds toward the full Astro demo
   - References specific team member contributions
   - Includes working code examples
3. Review with user for approval
