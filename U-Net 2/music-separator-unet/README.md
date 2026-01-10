---
tags:
- audio
- music-source-separation
- u-net
- pytorch
license: mit
datasets:
- musdb18hq
metrics:
- SDR
- SIR
- SAR
---

# 🎸 Music-U-Net — 4-Stem Source Separator

A PyTorch U-Net trained to split a full-band stereo **mixture** into  
**drums · bass · other · vocals**.

| Property              | Value |
|-----------------------|-------|
| Model type            | 2-D U-Net (6.2 M params) |
| Input representation  | STFT magnitude (mono, 16 kHz) |
| Output                | 4 magnitude masks (drums, bass, other, vocals) |
| Training data         | 100 train + 50 test songs from **MUSDB-18 HQ** |
| Checkpoint size       | ~24 MB (`state_dict`, FP32) |
| License               | MIT |

---

## 🗂️ Contents  

```
checkpoints/unet\_best.pt   # model weights (state\_dict)
config/default.yaml        # sample-rate, FFT size, etc.
README.md                  # this card

```

---

## 📝 Model Details  

### Architecture  
Classic symmetric U-Net over 2-D spectra:

```
Encoder:  \[C32]→\[C64]→\[C128]→\[C256]→\[C512]
Decoder:  \[C256]←\[C128]←\[C64]←\[C32]
```

`ReLU` activations, batch-norm, skip-connections, 1×1 final conv to **4 channels**  
(one per target stem) followed by soft masks --> multiplied by mixture magnitude.

### Training  
* **Loss**: L1( pred_mag·mix_phase , ref_mag·mix_phase )  
* **Augment**: time/freq masking, Gaussian noise, ±3 dB gain  
* **Optimizer**: Adam, LR 1e-4 → 1e-5 cosine decay, 50 epochs  
* **Hardware**: single RTX 3090, 2 h total

---

## 📊 Evaluation (MUSDB-18 test, per-track average)

| Metric | Mean | Std |
|--------|------|-----|
| **SDR** | **-0.14 dB** | 1.66 |
| **SIR** | 3.93 dB | 1.86 |
| **SAR** | 4.26 dB | 0.85 |

*(baseline numbers; not state-of-the-art, but fast & lightweight)*

---

## 💻 Usage

Try it live in the **Gradio Space** 👉 **[https://huggingface.co/spaces/theadityamittal/music-separator-space](https://huggingface.co/spaces/YOUR_USERNAME/music-separator-space)**

---

## ⚖ Limitations & Biases

* Trained only on MUSDB-18 HQ → may fail on genres not represented (classical, EDM).
* Uses mixture phase → audible bleeding & artifacts, negative SDR in some tracks.
* No multi-channel or stem permutation handling.

---

## 📄 License

Released under the MIT License.

---

## 🙏 Citation

```bibtex
@misc{music-unet-2025,
  title   = {Music Source Separation with U-Net},
  author  = {Your Name},
  url     = {https://huggingface.co/YOUR_USERNAME/music-separator-unet},
  year    = 2025
}
```
