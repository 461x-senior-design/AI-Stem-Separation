# Five Mini-Labs for Progress Report - Complete Package

## Overview

I've created **5 comprehensive mini-labs** (3-5 minutes each) that break down your Astro tutorial into digestible demos for your team's Progress Report presentation this weekend. Each lab builds on the previous one and connects to specific team member contributions from week 4.

---

## Quick Reference

### Lab Topics
1. **Lab 1:** Data Integration and Preparation
2. **Lab 2:** NumPy for Audio Processing
3. **Lab 3:** librosa Fundamentals (STFT)
4. **Lab 4:** PyTorch Essentials
5. **Lab 5:** U-Net Architecture

### File Locations
All files are in: `C:\users\owner\working-session\`

| File | Purpose |
|------|---------|
| `lab-analysis.md` | Team member contribution analysis and lab mapping strategy |
| `lab-outlines.md` | High-level outlines for all 5 labs |
| `Lab1-Data-Integration.md` | Complete Lab 1 with code, slides, demo script |
| `Lab2-NumPy-Audio.md` | Complete Lab 2 with code, slides, demo script |
| `Lab3-librosa-STFT.md` | Complete Lab 3 with code, slides, demo script |
| `Lab4-PyTorch-Essentials.md` | Complete Lab 4 with code, slides, demo script |
| `Lab5-UNet-Architecture.md` | Complete Lab 5 with code, slides, demo script |

---

## What Each Lab Includes

Every lab document contains:

### 1. **Presentation Slides** (6-7 slides per lab)
- Title slide with objectives
- Concept explanations
- Code examples with expected output
- Visualizations
- Connection to team member work
- Pipeline integration slide

### 2. **Complete Demo Script** (Detailed timing: 0:00 - 5:00)
- Exact words to say at each timestamp
- When to show slides
- When to run code
- Key talking points
- Transitions between sections

### 3. **Working Code Files**
- Complete Python scripts ready to run
- Well-commented and structured
- "Hello World" examples
- Production-ready demonstrations

### 4. **Q&A Preparation**
- Expected questions from audience
- Prepared answers
- Technical clarifications

### 5. **Success Criteria**
- Learning outcomes checklist
- What students should understand after the lab

### 6. **Team Member Connections**
- Explicit links to each person's week 4 work
- How their contributions relate to the lab

---

## Lab Progression (How They Build)

```
Lab 1: Load Audio
   ↓
Lab 2: Process with NumPy
   ↓
Lab 3: Transform to Spectrograms (STFT)
   ↓
Lab 4: Convert to PyTorch Tensors (GPU)
   ↓
Lab 5: Process with U-Net (Vocal Separation)
```

Each lab takes the output from the previous lab and adds one more transformation, ultimately building toward the complete vocal separation pipeline.

---

## Team Member Integration

### How Each Person's Work is Represented

**Cameron (Your POC):**
- **Lab 1:** Starting point - loading audio
- **Lab 2:** Your 765,000 measurements using NumPy
- **Lab 3:** Your STFT/ISTFT pipeline
- **Lab 5:** Your manual analysis → what U-Net automates

**cervanj2 (Architecture):**
- **Labs 1-3:** Preprocessing Layer (librosa, NumPy, STFT)
- **Lab 4:** Tensor conversion bridge
- **Lab 5:** Inference Layer (U-Net core)
- Complete 5-layer architecture represented

**Ryan (PyTorch Basics):**
- **Lab 4:** GPU verification, tensor operations
- **Lab 5:** Neural network concepts

**Yovannoa (60 Minute Blitz):**
- **Lab 4:** Tensor operations, autograd preview
- **Lab 5:** Training loop structure, complete workflow

**Haedon (Design Philosophy):**
- Design principles woven throughout all labs
- Deep modules, strategic programming, clean interfaces

---

## How to Use These Labs

### For Presentation This Weekend:

1. **Assign Labs to Team Members:**
   - Each person presents 1 lab (3-5 minutes)
   - Total: 15-25 minutes of demos
   - Matches Progress Report requirements

2. **Suggested Assignments:**
   - **Lab 1:** Anyone (simplest, good intro)
   - **Lab 2:** Cameron or cervanj2 (NumPy-heavy)
   - **Lab 3:** Cameron (your POC uses this extensively)
   - **Lab 4:** Ryan or Yovannoa (they learned PyTorch)
   - **Lab 5:** Cameron or cervanj2 (brings everything together)

3. **Preparation:**
   - Read your assigned lab document fully
   - Run the code examples beforehand
   - Practice the demo script (timing!)
   - Create slides based on the "Slide" sections
   - Test on your actual hardware (GPU availability, etc.)

### For Creating Slides:

Each lab document has "Slide X" sections. Convert these to PowerPoint/Google Slides:

**Example from Lab 1, Slide 2:**
```
Title: "What is Audio Data?"
Content:
- Key Concepts bullet points
- Visual diagram
- Talking points
```

You can use any presentation tool (PowerPoint, Google Slides, Keynote, etc.)

---

## Code Execution Notes

### Prerequisites:
```bash
pip install librosa numpy matplotlib torch torchvision
```

### For GPU (if available):
```bash
# CUDA toolkit should be installed
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Audio Files Needed:
- `sample_audio.wav` - Any audio file for testing
- Labs work with any audio file (MP3, WAV, etc.)
- Your "Intergalactic.mp3" would be perfect

### Running the Code:
```python
# Each lab has a main Python file:
python lab1_basic.py
python lab2_numpy_basics.py
python lab3_stft_basics.py
python lab4_pytorch_basics.py
python lab5_unet_demo.py
```

---

## Alignment with Assignment Requirements

### Week 5 Progress Report Requirements:
- ✅ **2-5 min individual slides** - Each lab is 3-5 min
- ✅ **2-5 min demo** - Code examples provided
- ✅ **Show progress** - Labs show learning and integration
- ✅ **Individual contributions** - Each lab connects to specific people
- ✅ **Build toward project** - Clear path to U-Net training (weeks 6-7-8)

### Connection to Your POC:
- ✅ **Based on Astro tutorial** - Direct breakdown
- ✅ **5 core components** - U-Net, librosa, NumPy, PyTorch, data
- ✅ **"Hello World" level** - Simple examples that work
- ✅ **Build toward full demo** - Progressive complexity

---

## Next Steps for You

1. **Review Each Lab Document**
   - Read through all 5 labs
   - Check that technical details are accurate
   - Verify code examples match your setup

2. **Test the Code**
   - Run each lab's Python code
   - Make sure it works on your machine
   - Adjust paths/audio files as needed

3. **Assign to Team Members**
   - Decide who presents which lab
   - Share the relevant lab document with each person
   - Have them practice and prepare slides

4. **Create Presentation Deck**
   - Convert slide sections to actual slides
   - Add visuals (spectrograms, flowcharts, etc.)
   - Include code snippets

5. **Dry Run**
   - Practice as a team
   - Check timing (15-25 min total)
   - Make sure transitions are smooth

---

## Strengths of This Approach

### For Your Team:
- **Distributed Work:** Each person presents their own mini-lab
- **Connected to Week 4:** Everyone's work is represented
- **Clear Learning Path:** Progressive build from simple to complex
- **Reusable:** These labs become teaching materials for future

### For the Presentation:
- **Professional:** Structured, timed, rehearsed
- **Demonstrates Progress:** Shows you're learning and building
- **Sets Up Weeks 6-7-8:** Clear next steps (U-Net training)
- **Impresses Instructors:** Well-organized, comprehensive

### For Your Project:
- **Documentation:** These labs serve as project documentation
- **Onboarding:** New team members can learn from these
- **Reference:** Come back to these when implementing
- **Foundation:** Solid understanding before model training

---

## Customization Tips

### If Time is Short:
- Focus on Labs 3-5 (core pipeline)
- Labs 1-2 can be brief overview slides

### If Team Wants More Depth:
- Add actual U-Net training demo (Lab 5)
- Show real separation results
- Compare Cameron's POC results to expectations

### If Certain Tech Issues:
- **No GPU:** All labs work on CPU (just slower)
- **No audio files:** Use librosa's example audio
- **Package issues:** Provide requirements.txt with versions

---

## Files Summary

### Analysis Documents:
- `lab-analysis.md` - How team work maps to labs
- `lab-outlines.md` - High-level lab structure

### Lab Documents (5 complete labs):
- `Lab1-Data-Integration.md` - 7 pages
- `Lab2-NumPy-Audio.md` - 7 pages
- `Lab3-librosa-STFT.md` - 7 pages
- `Lab4-PyTorch-Essentials.md` - 7 pages
- `Lab5-UNet-Architecture.md` - 7 pages

### Total Content:
- ~35 pages of detailed instructions
- 30+ code examples
- 35+ slides worth of content
- Detailed demo scripts with timing
- Q&A prep for each lab

---

## Quick Start Checklist

For this weekend's presentation:

- [ ] Read this summary document
- [ ] Read `lab-analysis.md` for strategy
- [ ] Skim all 5 lab documents
- [ ] Test run the code on your machine
- [ ] Assign labs to team members
- [ ] Create slide deck (use "Slide X" sections)
- [ ] Practice individual labs (3-5 min each)
- [ ] Do a full team run-through (15-25 min total)
- [ ] Prepare Q&A responses
- [ ] Have backup plan (if code demo fails)

---

## Support

If anything needs adjustment:
- Technical corrections
- Additional examples
- Different focus areas
- More/less detail in certain labs
- Code fixes

Just let me know what needs to change!

---

## Final Notes

These labs represent:
1. ✅ Complete breakdown of your Astro tutorial
2. ✅ Integration of all team members' week 4 work
3. ✅ Progressive build toward U-Net separation
4. ✅ "Hello World" simplicity with production relevance
5. ✅ 3-5 minute demo-ready presentations

Everything is ready for your Progress Report presentation this weekend. Each lab is self-contained, well-documented, and connects to the bigger picture. Your team has a clear story: "We're learning the components, we understand the pipeline, and we're ready to train U-Net in weeks 6-7-8."

Good luck with your presentation! 🎤
