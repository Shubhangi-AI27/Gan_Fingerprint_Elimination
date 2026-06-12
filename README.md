# GAN Fingerprint Elimination & Deepfake Attribution

> **Research Internship Project** — IIT Bhilai, MIST Lab (Jan–Jun 2025)
> Supervisor: Dr. Sk. Subidh Ali
> Based on: *Lai et al., "Untraceable DeepFakes via Traceable Fingerprint Elimination", AAAI 2026*
> Paper: https://arxiv.org/abs/2412.09373

---

## Overview

This project implements two systems:

1. **GAN Fingerprint Elimination** — removes GAN-specific forensic traces from deepfake images so they cannot be attributed to their source GAN, while preserving perceptual quality. Evaluated against DNA-Det.

2. **ForensicNet** — a custom-designed 4-class deepfake attribution model (original contribution) built on EfficientNet-B0 with a novel 3-channel forensic feature extractor. No reference implementation was used.

Both systems are combined in a single **Gradio UI** (`app.py`) with 4 interactive tabs.

---

## Repository Structure

```
gan-fingerprint-elimination/
│
├── app.py                          ← Combined Gradio UI (4 tabs)
├── annotation_new.py               ← Step 1 — Dataset annotation for DNA-Det
├── train_dna-det.py                ← Step 2 — DNA-Det 6-class training
├── testing_dna-det_attribution.py  ← Step 3 — DNA-Det inference UI
├── final_elimination.py            ← Step 4 — Fingerprint elimination training
│
├── custom_attribution/             ← ForensicNet (original custom model)
│   ├── step1_dataset_builder.py    ← Builds 4-class dataset (dataset_v5/)
│   ├── forensicnet.py              ← Model definition + forensic extractor
│   ├── train.py                    ← Two-phase EfficientNet-B0 training
│   └── inference.py                ← Single image prediction CLI
│
├── generation_scripts/             ← Scripts used to generate GAN images
│   ├── stylegan2_generate.py       ← StyleGAN2-ADA (FFHQ weights, 1024×1024)
│   ├── dcgan_generation.py         ← DCGAN v1 (64px, latent=128)
│   ├── dcgan_archA.py              ← DCGAN v2 / ArchA (128px, latent=100, 2-phase)
│   ├── progan_tfhub_5000images.py  ← ProGAN via TF Hub
│   └── progan_tfhub_generate.py    ← ProGAN generation script
│
├── samples/                        ← Small set of demo output images
│   ├── before_elimination/         ← Original GAN images (input)
│   └── after_elimination/          ← Processed images (output)
│
├── checkpoints/                    ← Model weights (not tracked — see below)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Part 1 — ForensicNet (Custom Attribution Model)

> Independent original contribution. No reference implementation used.

### What It Is

A 4-class deepfake attribution model that classifies images as:
**Real / ProGAN / StyleGAN / DCGAN**

Uses EfficientNet-B0 as backbone with a custom 3-channel forensic feature extractor instead of raw RGB input:

| Channel | Feature | What It Captures |
|---|---|---|
| R | Gaussian Residual Map | High-frequency noise fingerprints |
| G | FFT Magnitude Spectrum | Frequency-domain GAN artifacts |
| B | Local Noise Std Map | Spatial texture variance |

### Pipeline

```
Step 1 — Build dataset
python custom_attribution/step1_dataset_builder.py

Step 2 — Train ForensicNet
python custom_attribution/train.py --data_dir ./dataset_v5 --ckpt_path ./checkpoints/forensicnet_best.pth

Step 3 — Inference
python custom_attribution/inference.py --image ./test.jpg --ckpt ./checkpoints/forensicnet_best.pth
```

### Dataset Sources (4-class)

**Real:**
- FFHQ: https://github.com/NVlabs/ffhq-dataset
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- 140k Real and Fake Faces (real split): https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

**StyleGAN:**
- Self-generated via `generation_scripts/stylegan2_generate.py` (StyleGAN2-ADA, truncation_psi=0.7)
- 140k Real and Fake Faces (fake split): https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

**ProGAN:**
- Generated via `generation_scripts/progan_tfhub_5000images.py`

**DCGAN:**
- Generated via `generation_scripts/dcgan_generation.py` and `generation_scripts/dcgan_archA.py`

Place generated images in `raw_inputs/<ClassName>/` before running `step1_dataset_builder.py`.

![ForensicNet Forensic Feature Map](samples/forensicnet.png)
![ForensicNet Gradio UI](samples/gradioui_forensicnet.png)


---

## Part 2 — DNA-Det Attribution Model (6-class)

Trains a custom ResNet-style attribution model to classify images as:
**Real / ProGAN / MMDGAN / SNGAN / StyleGAN / CramerGAN**

This model serves as the **target** for the fingerprint elimination framework — the elimination model is trained to fool DNA-Det.

### Pipeline

```
Step 1 — Generate annotations
python annotation_new.py

Step 2 — Train DNA-Det
python train_dna-det.py

Step 3 — Run inference UI
python testing_dna-det_attribution.py
```

### Dataset Sources (6-class)

| Class | Source |
|---|---|
| Real | FFHQ: https://github.com/NVlabs/ffhq-dataset |
| ProGAN | https://github.com/tkarras/progressive_growing_of_gans |
| StyleGAN | https://github.com/NVlabs/stylegan2-ada-pytorch |
| SNGAN | https://github.com/pfnet-research/sngan_projection |
| CramerGAN | https://github.com/StanleyFu/cramer-gan |
| MMDGAN | https://github.com/OctoberChang/MMD-GAN |

![DNA-Det Confusion Matrix](samples/confusion_matrix.webp)
![DNA-Det Gradio UI](samples/gradioui_dna-det.png)

---

## Part 3 — Fingerprint Elimination

Trains an encoder-decoder model to remove GAN fingerprints from deepfake images while preserving perceptual quality.

### Architecture

```
Input image x
    → Encoder (Conv + 5 ResidualBlocks)
    → Decoder (Upsample + Conv)
    → Residual output δ
    → x' = clamp(x + δ, -1, 1)
    → GBMS Smoother (Gaussian Blur + Mean Shift)
    → Final untraceable image x'
```

### Loss Functions

| Loss | Weight | Purpose |
|---|---|---|
| Attribution Loss | β1 | Fool DNA-Det (cross-entropy) |
| Perceptual Loss | β2 | Preserve visual quality (VGG16) |
| Spectral Loss | β3 | Preserve frequency content (FFT) |
| Spatial Loss | — | Pixel-level similarity (L1) |

### Key Bug Fixes Applied

| Fix | Issue | Solution |
|---|---|---|
| FIX 1 | float16 overflow in VGG under AMP | Cast pred/target to float32 before VGG forward |
| FIX 2 | GAN-specific transforms causing overfitting | Removed; kept only paper's 6 transforms |
| FIX 3 | Bilinear JPEG proxy inaccurate | Real PIL JPEG encode/decode via BytesIO |
| FIX 4 | Decoder weight init saturation (33.6%) | Xavier uniform gain=0.1 + zero bias |
| FIX 5 | ProGAN nearest-neighbour coverage | Sampling bias: nearest=0.5, bilinear=0.3, bicubic=0.2 |
| FIX 6 | Stale GradScaler from broken run | Reset init_scale=2¹⁰ |

### Training

```
python final_elimination.py
```

Trained on: NVIDIA A6000 (48GB VRAM) · FFHQ 128×128 · 70,000 images

![Training Curves](samples/training_curves.png)
![Epoch 200 Visualization](samples/viz_ep200.png)

---

## Results

### Fingerprint Elimination (ASR against DNA-Det)

| GAN | ASR (↑) | Images Tested |
|---|---|---|
| MMDGAN | 100.0% | 200 |
| CramerGAN | 99.3% | 200 |
| SNGAN | 53.7% | 200 |
| StyleGAN | 43.3% | 200 |
| ProGAN | ~1.0% | 200 |
| **Overall** | **74.1%** | **1,200** |

**Image Quality:** PSNR 23.57 dB · SSIM 0.751

> **ProGAN Note:** Near-zero ASR attributed to insufficient nearest-neighbour sampling coverage and spectral loss not targeting FFT corner frequencies where ProGAN checkerboard artifacts concentrate. Documented as a future research direction.

<!-- INSERT IMAGE: samples/asr_bar_chart.png — bar chart of ASR per GAN class -->

---

---

## Limitations

### ProGAN — Hard Fingerprint Problem
The elimination model achieves ~1% ASR on ProGAN, compared to 99–100% on
MMDGAN and CramerGAN. Two root causes were identified:

**1. Sampling bias insufficient:**
ProGAN generates images using nearest-neighbour upsampling, which creates
checkerboard artifacts at specific spatial frequencies. The nearest-neighbour
sampling unit in the Data Synthesis Module was not firing frequently enough
(~16.5% of steps) to expose the model to these patterns during training.
Even after Fix 5 (increased to ~35%), coverage remained insufficient for
full fingerprint removal.

**2. Spectral loss blind spot:**
ProGAN checkerboard artifacts concentrate at FFT corner frequencies
(Nyquist region). The SpectralLoss function computes global FFT magnitude
difference but does not apply targeted weighting to these corner regions,
so the elimination model never receives a strong gradient signal to remove
ProGAN-specific spectral patterns.

### Other Known Limitations
- Model trained on FFHQ 128×128 only — may not generalise to other
  resolutions or non-face domains
- PSNR 23.57 dB is below broadcast-quality threshold (>30 dB) — visible
  smoothing artifacts present in some outputs
- Single shared model for all GAN classes — no architecture-specific
  specialisation

---

## Future Scope

**1. Architecture-specific training for hard fingerprints**
Train separate elimination heads per GAN architecture, or use a mixture-of-
experts approach where a routing network selects the appropriate elimination
path based on detected fingerprint type. Expected to resolve ProGAN's ~1% ASR.

**2. Targeted spectral loss at corner frequencies**
Modify SpectralLoss to apply higher weighting at FFT corner regions
(spatial frequencies above 0.8× Nyquist) where ProGAN and similar
nearest-neighbour upsampling artifacts concentrate.

**3. Higher resolution support**
Extend the framework to 256×256 and 512×512 by scaling the encoder-decoder
with additional residual blocks and progressive training similar to ProGAN
itself.

**4. Generalisation beyond faces**
Current training data (FFHQ) is face-only. Re-training on diverse image
domains (LSUN, ImageNet subsets) would test generalisation of the fingerprint
elimination approach.

**5. Extend ForensicNet to more GAN classes**
Currently 4-class (Real/ProGAN/StyleGAN/DCGAN). Extending
## Combined Gradio UI

`app.py` provides a 4-tab interactive interface:

 Function |

DNA-Det Attribution | 6-class GAN source prediction 
ForensicNet Attribution | 4-class custom model prediction 
Fingerprint Elimination | Remove GAN fingerprint from image 
Full Pipeline (ASR Demo) | Attribution → Eliminate → Attribution → ASR result 

### Run

```bash
pip install -r requirements.txt
python app.py
```

Place model checkpoints in `./checkpoints/` before running:
```
checkpoints/
├── dnadet_best.pth
├── forensicnet_best.pth
└── elimination_best.pth
```

![Gradio UI Tab 1](samples/final_UI.png)
![Gradio UI Tab 2](samples/final_UI2.png)

---

## Installation

```bash
git clone https://github.com/<your-username>/gan-fingerprint-elimination.git
cd gan-fingerprint-elimination
pip install -r requirements.txt
```

---

## Model Weights

Trained checkpoints are not included in this repository due to size.
Available on request from the supervisor (Dr. Sk. Subidh Ali, IIT Bhilai).

---

## Citation

```bibtex
@inproceedings{lai2026untraceable,
  title     = {Untraceable DeepFakes via Traceable Fingerprint Elimination},
  author    = {Lai, et al.},
  booktitle = {AAAI},
  year      = {2026}
}
```

---

## License

Research code for academic purposes only.
GAN-generated images are not redistributed.
Refer to respective model licenses (NVIDIA, etc.) for generated content usage.

---

*MIST Lab · IIT Bhilai · Internship Project · Jan–Jun 2025*
