"""
Untraceable DeepFakes via Traceable Fingerprint Elimination
============================================================
Paper  : Lai et al., AAAI 2026  (arXiv:2508.03067)
Target : ProGAN, StyleGAN, SNGAN, CramerGAN, MMDGAN
Dataset: FFHQ 128x128 thumbnails (70 000 faces)
Test   : DNA-Det attribution model

ALL FIXES APPLIED (vs previous broken version)
───────────────────────────────────────────────
FIX 1 — PerceptualLoss float32 cast  [ROOT CAUSE — kills ProGAN/StyleGAN]
         Added pred.float() + target.float() at start of forward().
         VGG16 was running in float16 inside autocast → overflow → loss
         exploded randomly (0.00003 ↔ 7.03) → model learned identity map.

FIX 2 — Remove GAN-specific transforms  [WRONG APPROACH — hurts SNGAN]
         Removed checkerboard, spectral_norm, fft_phase from transform_unit.
         Keeping only paper's 6: noise, blur, crop, jpeg, relight, combo.
         GAN-specific transforms caused overfitting to simulated patterns.

FIX 3 — Real PIL JPEG  [MINOR — 14% spectral budget was wasted]
         Replaced _jpeg_sim() bilinar proxy (0.78 spectral diff) with
         real PIL JPEG encode/decode via BytesIO.

FIX 4 — Decoder weight init  [33.6% pixels were saturating at init]
         Xavier uniform gain=0.1 + zero bias on decoder final conv.
         Keeps output small at init → clamp saturation < 2%.

FIX 5 — Sampling bias toward nearest-neighbour  [ProGAN coverage]
         p1: 0.5 → 0.7 (fires 70% of steps vs 50%)
         dm weights: nearest=0.5, bilinear=0.3, bicubic=0.2
         nearest absolute firing: 16.5% → 35% of all steps.
         bilinear absolute firing: 16.5% → 21% (StyleGAN still covered).
         bicubic absolute firing: 16.5% → 14% (CramerGAN already 99.3%).

FIX 6 — GradScaler reset  [was using stale scale from broken run]
         init_scale=2**10 instead of default 2**16.

Expected ASR after full retrain (150-200 epochs, batch=32 on A6000):
  ProGAN    : 80–97%   (was 1.3%)
  StyleGAN  : 75–95%   (was 19.3%)
  SNGAN     : 80–95%   (was 48%)
  CramerGAN : 90–99%   (was 99.3% — stays)
  MMDGAN    : 90–100%  (was 100% — stays)
  Overall   : 83–97%   (was 53.6%)
"""

# ══════════════════════════════════════════════════════════════════
# CELL 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════
import io
import os
import json
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free  = (torch.cuda.get_device_properties(0).total_memory
             - torch.cuda.memory_allocated()) / 1024**3
    print(f"VRAM    : {total:.1f} GB total  |  {free:.1f} GB free")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")


# ══════════════════════════════════════════════════════════════════
# CELL 2 — CONFIG
# ══════════════════════════════════════════════════════════════════
CONFIG = {
    # ── paths ──────────────────────────────────────────────────────
    "image_folder"   : "/data1/intern/thumbnails128x128",
    "checkpoint_dir" : "./checkpoints_fixed_v1",
    "output_dir"     : "./outputs_fixed_v1",

    # ── data ───────────────────────────────────────────────────────
    "max_samples"    : 70000,
    "image_size"     : 128,

    # ── training ───────────────────────────────────────────────────
    "epochs"         : 200,
    "batch_size"     : 32,       # A6000 has 47GB — use 32 for 1.8x speed
    "lr"             : 2e-4,
    "lr_min"         : 1e-6,
    "warmup_epochs"  : 5,
    "weight_decay"   : 1e-4,
    "grad_clip"      : 1.0,

    # ── logging ────────────────────────────────────────────────────
    "save_every"     : 10,
    "viz_every"      : 10,

    # ── architecture ───────────────────────────────────────────────
    "base_channels"  : 64,
    "residual_blocks": 5,

    # ── loss weights — paper exact ─────────────────────────────────
    "beta1"          : 0.5,      # perceptual
    "beta2"          : 0.1,      # spatial
    "beta3"          : 0.4,      # spectral

    # ── FIX 5: p1 raised 0.5 → 0.7 for ProGAN nearest-neighbour ───
    "p1"             : 0.7,      # was 0.5 — sampling unit fires 70% of steps
    "p2"             : 0.8,      # transform unit probability — unchanged
}

for d in [CONFIG["checkpoint_dir"], CONFIG["output_dir"]]:
    os.makedirs(d, exist_ok=True)

with open(os.path.join(CONFIG["checkpoint_dir"], "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

print("\nCONFIG saved")
print(f"  image_folder : {CONFIG['image_folder']}")
print(f"  epochs       : {CONFIG['epochs']}")
print(f"  batch_size   : {CONFIG['batch_size']}")
print(f"  lr           : {CONFIG['lr']}  (warm-up {CONFIG['warmup_epochs']} epochs)")
print(f"  p1           : {CONFIG['p1']}  (FIX 5: was 0.5)")
print(f"  beta         : {CONFIG['beta1']} / {CONFIG['beta2']} / {CONFIG['beta3']}")


# ══════════════════════════════════════════════════════════════════
# CELL 3 — DATASET
# ══════════════════════════════════════════════════════════════════
class RealImageDataset(Dataset):
    def __init__(self, folder, transform, max_samples=None):
        all_images = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if max_samples:
            all_images = all_images[:max_samples]
        self.images    = all_images
        self.transform = transform
        print(f"Dataset : {len(self.images):,} images from {folder!r}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset = RealImageDataset(
    CONFIG["image_folder"],
    transform,
    max_samples=CONFIG["max_samples"],
)

loader = DataLoader(
    dataset,
    batch_size      = CONFIG["batch_size"],
    shuffle         = True,
    num_workers     = 4,
    drop_last       = True,
    pin_memory      = True,
    prefetch_factor = 2,
)

print(f"Steps / epoch : {len(loader):,}")
print(f"Total steps   : {len(loader) * CONFIG['epochs']:,}")


# ══════════════════════════════════════════════════════════════════
# CELL 4 — DATA SYNTHESIS MODULE
#
# Paper transforms only (6) — no GAN-specific extras (FIX 2):
#   noise | blur | crop | jpeg | relight | combo
#
# FIX 3: Real PIL JPEG replaces bilinar proxy (_jpeg_real).
# FIX 5: sampling_unit biased toward nearest-neighbour.
# ══════════════════════════════════════════════════════════════════
class DataSynthesisModule(nn.Module):

    def __init__(self, p1=0.7, p2=0.8):
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    # ── FIX 5: sampling unit — biased toward nearest-neighbour ─────
    @torch.no_grad()
    def sampling_unit(self, x):
        """
        Down H→H/2 then up H/2→H.
        FIX 5: dm is biased — nearest=0.5, bilinear=0.3, bicubic=0.2
               nearest absolute firing: 16.5% → 35% of all steps.
               bilinear absolute firing: 16.5% → 21% (StyleGAN covered).
               bicubic absolute firing: 16.5% → 14% (CramerGAN fine).
        p1 raised 0.5→0.7 so sampling fires more often overall.
        um (up-mode) stays uniform — artifact type set by down-mode only.
        """
        if random.random() > self.p1:
            return x
        B, C, H, W = x.shape

        # FIX 5: biased down-mode weights
        dm = random.choices(
            ["nearest", "bilinear", "bicubic"],
            weights=[0.5, 0.3, 0.2]
        )[0]
        # up-mode stays uniform — unchanged from paper
        um = random.choice(["nearest", "bilinear", "bicubic"])

        kd = {} if dm == "nearest" else {"align_corners": False}
        ku = {} if um == "nearest" else {"align_corners": False}
        xd = F.interpolate(x, size=(H // 2, W // 2), mode=dm, **kd)
        return F.interpolate(xd, size=(H, W), mode=um, **ku)

    # ── helpers ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _gblur(self, x, k, sigma=1.0):
        C   = x.shape[1]
        pad = k // 2
        c   = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        g   = torch.exp(-c**2 / (2 * sigma**2))
        g   = g / g.sum()
        k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        return F.conv2d(F.pad(x, [pad] * 4, "reflect"), k2d, groups=C)

    # ── FIX 3: real PIL JPEG — replaces fake bilinar proxy ──────────
    @torch.no_grad()
    def _jpeg_real(self, x, quality):
        """
        FIX 3: Real PIL JPEG encode/decode via BytesIO.
        Previous _jpeg_sim() had 0.78 spectral diff from real JPEG,
        wasting 14% of spectral training signal on wrong artifacts.
        """
        res = []
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        for img_t in x:
            pil = to_pil(img_t.cpu().clamp(0, 1))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            out = to_tensor(Image.open(buf).convert("RGB"))
            res.append(out)
        return torch.stack(res).to(x.device)

    # ── transform unit — paper's 6 only (FIX 2) ─────────────────────
    @torch.no_grad()
    def transform_unit(self, x):
        if random.random() > self.p2:
            return x

        B, C, H, W = x.shape

        # FIX 2: paper's 6 transforms only — checkerboard/spectral_norm/
        # fft_phase removed. They caused overfitting to simulated patterns.
        t = random.choice([
            "noise", "blur", "crop", "jpeg", "relight", "combo"
        ])

        if t == "noise":
            sigma = random.uniform(5.0, 20.0) / 255.0
            x = (x + torch.randn_like(x) * sigma).clamp(0, 1)

        elif t == "blur":
            k = random.choice([1, 3, 5])
            if k > 1:
                x = self._gblur(x, k).clamp(0, 1)

        elif t == "crop":
            off  = max(int(H * random.uniform(0.05, 0.20)), 1)
            top  = random.randint(0, off)
            left = random.randint(0, off)
            hc   = H - random.randint(0, off)
            wc   = W - random.randint(0, off)
            top  = min(top, H - 1);   left = min(left, W - 1)
            hc   = max(hc, top + 1);  wc   = max(wc, left + 1)
            hc   = min(hc, H);        wc   = min(wc, W)
            x = F.interpolate(
                x[:, :, top:hc, left:wc],
                size=(H, W), mode="bilinear", align_corners=False)

        elif t == "jpeg":
            # FIX 3: use real PIL JPEG
            x = self._jpeg_real(x, random.randint(10, 75))

        elif t == "relight":
            x = (x * random.uniform(0.5, 1.5)).clamp(0, 1)
            mean = x.mean(dim=[2, 3], keepdim=True)
            x = ((x - mean) * random.uniform(0.5, 1.5) + mean).clamp(0, 1)

        elif t == "combo":
            # Paper order: relight → crop → blur → jpeg → noise
            x = (x * random.uniform(0.5, 1.5)).clamp(0, 1)
            off = int(H * random.uniform(0.05, 0.20))
            if off > 0:
                top  = random.randint(0, off)
                left = random.randint(0, off)
                hc   = max(H - random.randint(0, off), top + 1)
                wc   = max(W - random.randint(0, off), left + 1)
                hc   = min(hc, H); wc = min(wc, W)
                x = F.interpolate(
                    x[:, :, top:hc, left:wc],
                    size=(H, W), mode="bilinear", align_corners=False)
            x = self._gblur(x, random.choice([3, 5])).clamp(0, 1)
            # FIX 3: real PIL JPEG in combo too
            x = self._jpeg_real(x, random.randint(10, 75))
            x = (x + torch.randn_like(x) * random.uniform(5, 20) / 255.0).clamp(0, 1)

        return x

    @torch.no_grad()
    def forward(self, xr):
        xup = self.sampling_unit(xr)
        xs  = self.transform_unit(xup)
        return xr, xs


print("DataSynthesisModule ready")
print("  FIX 2: transforms = noise/blur/crop/jpeg/relight/combo only (paper exact)")
print("  FIX 3: _jpeg_real() via PIL BytesIO")
print(f"  FIX 5: sampling nearest={0.5} bilinear={0.3} bicubic={0.2}  p1={CONFIG['p1']}")


# ══════════════════════════════════════════════════════════════════
# CELL 5 — MODEL ARCHITECTURE  (paper exact)
# ══════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, bc=64, n_res=5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, bc,    7, 1, 3, bias=False),
            nn.InstanceNorm2d(bc,   affine=True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(bc,   bc*2,   3, 2, 1, bias=False),
            nn.InstanceNorm2d(bc*2, affine=True),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(bc*2, bc*4,   3, 2, 1, bias=False),
            nn.InstanceNorm2d(bc*4, affine=True),
            nn.ReLU(inplace=True))
        self.res = nn.Sequential(*[ResidualBlock(bc * 4) for _ in range(n_res)])

    def forward(self, x):
        return self.res(self.conv3(self.conv2(self.conv1(x))))


class Decoder(nn.Module):
    def __init__(self, bc=64, out_ch=3):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(bc*4, bc*2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(bc*2, affine=True),
            nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(bc*2, bc,   3, 1, 1, bias=False),
            nn.InstanceNorm2d(bc, affine=True),
            nn.ReLU(inplace=True))
        self.out = nn.Sequential(
            nn.Conv2d(bc, out_ch, 7, 1, 3),
            nn.Tanh())

    def _init_weights(self):
        # FIX 4: small init on final conv → clamp saturation < 2% at init
        # was 33.6% of pixels hitting clamp before this fix
        nn.init.xavier_uniform_(self.out[0].weight, gain=0.1)
        nn.init.zeros_(self.out[0].bias)

    def forward(self, x):
        return self.out(self.up2(self.up1(x)))


class EliminationModel(nn.Module):
    """Adversarial model Phi — paper equation: x' = G(Phi(x))"""

    def __init__(self, in_ch=3, bc=64, n_res=5):
        super().__init__()
        self.encoder = Encoder(in_ch, bc, n_res)
        self.decoder = Decoder(bc, in_ch)
        # FIX 4: apply small weight init immediately after construction
        self.decoder._init_weights()

    def forward(self, x):
        return (x + self.decoder(self.encoder(x))).clamp(-1, 1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


print("EliminationModel : Encoder(Conv×3 + Res×5) + Decoder(Up×2 + Conv)")
print("  FIX 4: decoder final conv init gain=0.1 → clamp sat < 2%")


# ══════════════════════════════════════════════════════════════════
# CELL 6 — GBMS SMOOTHER  (paper exact, unchanged)
# ══════════════════════════════════════════════════════════════════
class _GaussianBlur(nn.Module):
    def __init__(self, ks=5, sigma=1.0):
        super().__init__()
        c   = torch.arange(ks, dtype=torch.float32) - ks // 2
        k1d = torch.exp(-c**2 / (2 * sigma**2))
        k1d = k1d / k1d.sum()
        k2d = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", k2d)
        self.pad = ks // 2

    def forward(self, x):
        C = x.shape[1]
        return F.conv2d(
            F.pad(x, [self.pad] * 4, "reflect"),
            self.kernel.expand(C, 1, -1, -1),
            groups=C)


class _MeanShift(nn.Module):
    def __init__(self, ks=7, ss=3.0, sr=0.1, iters=1):
        super().__init__()
        self.ks    = ks
        self.sr    = sr
        self.iters = iters
        self.pad   = ks // 2
        c = torch.arange(ks, dtype=torch.float32) - ks // 2
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("sw", torch.exp(-(gx**2 + gy**2) / (2 * ss**2)))

    @torch.no_grad()
    def _pass(self, x):
        B, C, H, W = x.shape
        k       = self.ks
        patches = F.unfold(
            F.pad(x, [self.pad] * 4, "reflect"), k
        ).view(B, C, k * k, H * W)
        centre = x.view(B, C, 1, H * W)
        rw     = torch.exp(
            -((patches - centre) ** 2).sum(1, keepdim=True) / (2 * self.sr ** 2)
        )
        sw  = self.sw.view(1, 1, k * k, 1)
        wt  = (sw * rw) / ((sw * rw).sum(2, keepdim=True) + 1e-8)
        return (patches * wt).sum(2).view(B, C, H, W)

    def forward(self, x):
        for _ in range(self.iters):
            x = self._pass(x)
        return x


class GBMSSmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.gb = _GaussianBlur(5, 1.0)
        self.ms = _MeanShift(7, 3.0, 0.1, 1)

    @torch.no_grad()
    def forward(self, x):
        return self.ms(self.gb(x)).clamp(0, 1)


print("GBMSSmoother : GaussianBlur(ks=5,σ=1) + MeanShift(ks=7,σs=3,σr=0.1)")


# ══════════════════════════════════════════════════════════════════
# CELL 7 — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss on relu1_2, relu2_2, relu3_3.

    FIX 1 (ROOT CAUSE): pred.float() + target.float() at start of forward().
    VGG16 was running in float16 inside torch.amp.autocast → features
    overflow to inf/nan → loss explodes randomly (0.00003 ↔ 7.03) →
    model learned identity mapping → ProGAN/StyleGAN never fixed.
    Cast to float32 before VGG. The rest of the model keeps AMP.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        f   = list(vgg.features.children())
        self.s1 = nn.Sequential(*f[:4])    # relu1_2
        self.s2 = nn.Sequential(*f[4:9])   # relu2_2
        self.s3 = nn.Sequential(*f[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # FIX 1 (COMPLETE): escape autocast entirely with enabled=False.
        # pred.float() alone is not enough — self.mean/std buffers are
        # also cast to float16 inside autocast context, so the division
        # immediately promotes back to float16 before VGG sees the data.
        # torch.amp.autocast(enabled=False) creates a float32-only sub-scope
        # that fully overrides the outer autocast — VGG runs in float32.
        with torch.amp.autocast("cuda", enabled=False):
            pred   = pred.float()
            target = target.float()
            mean   = self.mean.float()
            std    = self.std.float()
            pred   = (pred   - mean) / std
            target = (target - mean) / std
            loss = 0.0
            p, t = pred, target
            for sl in [self.s1, self.s2, self.s3]:
                p = sl(p)
                t = sl(t)
                loss = loss + F.mse_loss(p, t.detach()) / 3.0
        return loss


class SpatialLoss(nn.Module):
    """Paper: L_spatial = ||Phi(xs) - xr||²"""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class SpectralLoss(nn.Module):
    """
    Paper exact:
      L(x, si) = log(|fft(x^si)| + eps)
      L_spectral = Σ_si wi * ||L(Phi(xs),si) - L(xr,si)||₁
      scales={1.0,0.5,0.25}, weights={0.5,0.3,0.2}

    NO fftshift — paper does not use it.
    fftshift was moving ProGAN corner-frequency peaks to centre,
    breaking gradient signal. Already correct in previous version.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.scales  = [1.0, 0.5, 0.25]
        self.weights = [0.5, 0.3, 0.2]
        self.eps     = eps

    def _fft_mag(self, x, s):
        if s != 1.0:
            h = max(int(x.shape[2] * s), 8)
            w = max(int(x.shape[3] * s), 8)
            x = F.interpolate(x, size=(h, w),
                              mode="bilinear", align_corners=False)
        # paper exact — fft2, log magnitude, NO fftshift
        fs  = torch.fft.fft2(x)
        mag = torch.log(torch.abs(fs) + self.eps)
        return mag

    def forward(self, pred, target):
        loss = 0.0
        for s, w in zip(self.scales, self.weights):
            pm = self._fft_mag(pred,   s)
            tm = self._fft_mag(target, s)
            loss = loss + w * F.l1_loss(pm, tm.detach())
        return loss


class TotalLoss(nn.Module):
    """
    Paper: L_total = beta1*L_perceptual + beta2*L_spatial + beta3*L_spectral
    beta1=0.5, beta2=0.1, beta3=0.4
    """

    def __init__(self, beta1, beta2, beta3):
        super().__init__()
        self.beta1      = beta1
        self.beta2      = beta2
        self.beta3      = beta3
        self.perceptual = PerceptualLoss()
        self.spatial    = SpatialLoss()
        self.spectral   = SpectralLoss()

    def forward(self, pred, target):
        lp    = self.perceptual(pred, target)
        ls    = self.spatial(pred, target)
        lf    = self.spectral(pred, target)
        total = self.beta1 * lp + self.beta2 * ls + self.beta3 * lf
        return total, {
            "perceptual": lp.item(),
            "spatial"   : ls.item(),
            "spectral"  : lf.item(),
        }


print("PerceptualLoss : VGG16 relu1_2/relu2_2/relu3_3  — FIX 1: float32 cast")
print("SpectralLoss   : paper-exact fft2, log|F|+eps, NO fftshift")
print(f"TotalLoss      : β1={CONFIG['beta1']} β2={CONFIG['beta2']} β3={CONFIG['beta3']}")


# ══════════════════════════════════════════════════════════════════
# CELL 8 — BUILD ALL COMPONENTS
# ══════════════════════════════════════════════════════════════════
model = EliminationModel(
    in_ch = 3,
    bc    = CONFIG["base_channels"],
    n_res = CONFIG["residual_blocks"],
).to(device)

smoother  = GBMSSmoother().to(device)
synthesis = DataSynthesisModule(p1=CONFIG["p1"], p2=CONFIG["p2"])
criterion = TotalLoss(
    CONFIG["beta1"], CONFIG["beta2"], CONFIG["beta3"]
).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr           = CONFIG["lr"],
    weight_decay = CONFIG["weight_decay"],
    betas        = (0.9, 0.999),
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max   = CONFIG["epochs"] - CONFIG["warmup_epochs"],
    eta_min = CONFIG["lr_min"],
)

# FIX 6: reset GradScaler — init_scale=2**10 (not default 2**16)
# Previous run had stale scale from broken float16 overflow run.
scaler = torch.amp.GradScaler("cuda",
    enabled    = torch.cuda.is_available(),
    init_scale = 2**10,
)

CKPT_DIR    = CONFIG["checkpoint_dir"]
CKPT_LATEST = os.path.join(CKPT_DIR, "latest.pth")
CKPT_BEST   = os.path.join(CKPT_DIR, "best_model.pth")
HIST_PATH   = os.path.join(CKPT_DIR, "history.json")

print(f"\nModel params : {model.count_parameters():,}")
print(f"Optimizer    : AdamW  lr={CONFIG['lr']}  wd={CONFIG['weight_decay']}")
print(f"Scheduler    : warm-up {CONFIG['warmup_epochs']} ep → CosineAnnealing")
print(f"GradScaler   : FIX 6: init_scale=2**10 (fresh start)")


# ══════════════════════════════════════════════════════════════════
# CELL 9 — LR WARM-UP HELPER
# ══════════════════════════════════════════════════════════════════
def get_warmup_lr(epoch, warmup_epochs, base_lr):
    """Linear warm-up: lr goes from base_lr/10 to base_lr over warmup_epochs."""
    if epoch < warmup_epochs:
        return base_lr * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
    return None


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ══════════════════════════════════════════════════════════════════
# CELL 10 — RESUME LOGIC
# ══════════════════════════════════════════════════════════════════
start_epoch = 0
best_loss   = float("inf")
history     = []

# NOTE: Do NOT resume from old broken checkpoints.
# New checkpoint_dir is ./checkpoints_fixed_v1 — always starts fresh.
if os.path.exists(CKPT_LATEST):
    print(f"\nResuming from {CKPT_LATEST} ...")
    ckpt = torch.load(CKPT_LATEST, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_loss   = ckpt.get("best_loss", float("inf"))
    history     = ckpt.get("history", [])
    remaining = max(CONFIG["epochs"] - max(start_epoch, CONFIG["warmup_epochs"]), 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=CONFIG["lr_min"])
    print(f"  Resumed   : epoch {start_epoch} / {CONFIG['epochs']}")
    print(f"  Best loss : {best_loss:.6f}")
else:
    print("\nNo checkpoint found — starting fresh (correct for first run)")


# ══════════════════════════════════════════════════════════════════
# CELL 11 — HELPERS
# ══════════════════════════════════════════════════════════════════
def save_checkpoint(path, epoch, avg_loss):
    torch.save({
        "epoch"    : epoch,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler"   : scaler.state_dict(),
        "best_loss": best_loss,
        "history"  : history,
        "config"   : CONFIG,
        "loss"     : avg_loss,
    }, path)


def save_viz_grid(epoch, xr, xs, pred_smooth):
    n    = min(8, xr.shape[0])
    grid = torch.cat([
        xr[:n].cpu(),
        xs[:n].cpu(),
        pred_smooth[:n].detach().cpu(),
    ], dim=0)
    grid = vutils.make_grid(grid, nrow=n, normalize=True, scale_each=True)
    path = os.path.join(CONFIG["output_dir"], f"viz_ep{epoch+1:03d}.png")
    vutils.save_image(grid, path)
    return path


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim_batch(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    ks, sigma = 11, 1.5
    c   = torch.arange(ks, dtype=torch.float32, device=pred.device) - ks // 2
    g   = torch.exp(-c**2 / (2 * sigma**2))
    g   = g / g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    pad = ks // 2

    def _filt(x):
        B, C, H, W = x.shape
        k = k2d.expand(C, 1, -1, -1)
        return F.conv2d(F.pad(x, [pad]*4, "reflect"), k, groups=C)

    mu1  = _filt(pred);   mu2  = _filt(target)
    mu1s = mu1 * mu1;     mu2s = mu2 * mu2;  mu12 = mu1 * mu2
    s1   = _filt(pred   * pred)   - mu1s
    s2   = _filt(target * target) - mu2s
    s12  = _filt(pred   * target) - mu12

    num  = (2*mu12 + C1) * (2*s12 + C2)
    den  = (mu1s + mu2s + C1) * (s1 + s2 + C2)
    ssim = (num / (den + 1e-8)).mean().item()
    return ssim


def save_curves():
    if len(history) < 2:
        return
    ep   = [h["epoch"]      for h in history]
    tot  = [h["loss"]       for h in history]
    perc = [h["perceptual"] for h in history]
    spat = [h["spatial"]    for h in history]
    spec = [h["spectral"]   for h in history]
    psnr = [h.get("psnr", 0) for h in history]
    ssim = [h.get("ssim", 0) for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(24, 4))

    axes[0].plot(ep, tot, "b-", lw=2)
    axes[0].set_title("Total Loss");    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, perc, label="Perceptual")
    axes[1].plot(ep, spat, label="Spatial")
    axes[1].plot(ep, spec, label="Spectral")
    axes[1].set_title("Loss Components"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, psnr, "g-", lw=2)
    axes[2].set_title("PSNR (dB)");     axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    axes[3].plot(ep, ssim, "r-", lw=2)
    axes[3].set_title("SSIM");          axes[3].set_xlabel("Epoch")
    axes[3].set_ylim(0, 1);             axes[3].grid(alpha=0.3)

    best_t = min(tot)
    plt.suptitle(
        f"Best loss={best_t:.5f}  |  "
        f"PSNR={psnr[-1]:.2f}dB  |  "
        f"SSIM={ssim[-1]:.4f}  |  "
        f"Epoch {ep[-1]}/{CONFIG['epochs']}",
        y=1.01)
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["output_dir"], "training_curves.png"),
        dpi=120, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# CELL 12 — TRAINING LOOP
#
# Per-batch flow:
#   1. Synthesize (xr, xs) from real batch             [no_grad]
#   2. model(xs*2-1) → pred  [-1,1]
#   3. pred → pred_01  [0,1]
#   4. loss(pred_01, xr)  ← gradients flow here
#      FIX 1: PerceptualLoss casts to float32 internally
#   5. backward + grad_clip + step
#   6. smoother(pred_01) → pred_smooth                 [no_grad, viz only]
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"TRAINING  epoch {start_epoch+1} → {CONFIG['epochs']}")
print(f"Dataset   : FFHQ 128×128  ({len(dataset):,} images)")
print(f"Steps/ep  : {len(loader):,}")
print(f"{'='*65}")
print(f"SMOKE TEST — epoch 1 Perc should stay 0.20–0.50 (no spikes)")
print(f"If Perc > 1.0 at epoch 1 → FIX 1 (float32 cast) was not applied")
print(f"{'='*65}\n")

for epoch in range(start_epoch, CONFIG["epochs"]):

    # ── LR schedule ──────────────────────────────────────────────
    wu_lr = get_warmup_lr(epoch, CONFIG["warmup_epochs"], CONFIG["lr"])
    if wu_lr is not None:
        set_lr(optimizer, wu_lr)
        current_lr = wu_lr
    else:
        if epoch == CONFIG["warmup_epochs"]:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = CONFIG["epochs"] - CONFIG["warmup_epochs"],
                eta_min = CONFIG["lr_min"])
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

    model.train()
    loss_sum  = 0.0
    psnr_sum  = 0.0
    ssim_sum  = 0.0
    last_comp = {}
    last_xr = last_xs = last_pred = None
    t0 = time.time()

    for xr_raw in loader:

        # Step 1 — synthesize (no_grad, on CPU)
        with torch.no_grad():
            xr, xs = synthesis(xr_raw)

        xr = xr.to(device)   # real image  [0,1]
        xs = xs.to(device)   # synthetic   [0,1]

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):

            # Step 2 — model input in [-1,1]
            pred    = model(xs * 2.0 - 1.0)

            # Step 3 — output back to [0,1]
            pred_01 = ((pred + 1.0) / 2.0).clamp(0, 1)

            # Step 4 — loss
            # FIX 1: PerceptualLoss.forward() casts pred/target to float32
            # internally — VGG never sees float16 — no overflow possible
            xr_01      = xr.clamp(0, 1)
            loss, comp = criterion(pred_01, xr_01)

        # Step 5 — backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        # Step 6 — smoother for viz/metrics only (no gradients)
        with torch.no_grad():
            pred_smooth = smoother(pred_01.clamp(0.001, 0.999))

        loss_sum += loss.item()
        psnr_sum += compute_psnr(pred_smooth, xr_01)
        ssim_sum += compute_ssim_batch(pred_smooth, xr_01)
        last_comp = comp
        last_xr, last_xs, last_pred = xr_01, xs, pred_smooth

    # ── epoch-level stats ────────────────────────────────────────
    n_steps  = len(loader)
    avg_loss = loss_sum / n_steps
    avg_psnr = psnr_sum / n_steps
    avg_ssim = ssim_sum / n_steps
    elapsed  = time.time() - t0
    is_best  = avg_loss < best_loss

    history.append({
        "epoch"      : epoch + 1,
        "loss"       : round(avg_loss,  6),
        "perceptual" : round(last_comp.get("perceptual", 0), 6),
        "spatial"    : round(last_comp.get("spatial",    0), 6),
        "spectral"   : round(last_comp.get("spectral",   0), 6),
        "psnr"       : round(avg_psnr,  4),
        "ssim"       : round(avg_ssim,  4),
        "lr"         : current_lr,
        "time_sec"   : round(elapsed,   1),
    })

    print(
        f"Ep {epoch+1:03d}/{CONFIG['epochs']} | "
        f"Loss={avg_loss:.5f} | "
        f"Perc={last_comp.get('perceptual',0):.4f} | "
        f"Spat={last_comp.get('spatial',0):.5f} | "
        f"Spec={last_comp.get('spectral',0):.4f} | "
        f"PSNR={avg_psnr:.2f}dB | "
        f"SSIM={avg_ssim:.4f} | "
        f"LR={current_lr:.2e} | "
        f"{elapsed:.0f}s"
        + ("  ✓ BEST" if is_best else "")
    )

    # Save latest every epoch
    save_checkpoint(CKPT_LATEST, epoch, avg_loss)

    # Save best
    if is_best:
        best_loss = avg_loss
        save_checkpoint(CKPT_BEST, epoch, avg_loss)
        print(f"   → best_model.pth  (loss={best_loss:.5f}  "
              f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f})")

    # Milestone checkpoint
    if (epoch + 1) % CONFIG["save_every"] == 0:
        mp = os.path.join(CKPT_DIR, f"epoch_{epoch+1:03d}.pth")
        save_checkpoint(mp, epoch, avg_loss)
        print(f"   → epoch_{epoch+1:03d}.pth")

    # Visualisation grid
    if (epoch + 1) % CONFIG["viz_every"] == 0 and last_xr is not None:
        vp = save_viz_grid(epoch, last_xr, last_xs, last_pred)
        print(f"   → {vp}")

    save_curves()
    with open(HIST_PATH, "w") as f:
        json.dump(history, f, indent=2)

print(f"\n{'='*65}")
print(f"Training complete!")
print(f"Best loss  : {best_loss:.5f}")
print(f"Checkpoints: {CKPT_DIR}/")
print(f"Outputs    : {CONFIG['output_dir']}/")
print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════
# CELL 13 — INFERENCE
# ══════════════════════════════════════════════════════════════════
def eliminate_fingerprint(img_input, checkpoint=CKPT_BEST):
    """
    Eliminate GAN fingerprint from a single image.
    Paper equation: x' = G(Phi(x))

    Parameters
    ----------
    img_input : str | PIL.Image | torch.Tensor [0,1]
    checkpoint: path to .pth file (default: best_model.pth)

    Returns
    -------
    PIL.Image — fingerprint-eliminated image
    """
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if isinstance(img_input, str):
        img = transforms.ToTensor()(Image.open(img_input).convert("RGB"))
    elif isinstance(img_input, Image.Image):
        img = transforms.ToTensor()(img_input.convert("RGB"))
    else:
        img = img_input

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred        = model(img * 2.0 - 1.0)
        pred_01     = ((pred + 1.0) / 2.0).clamp(0, 1)
        pred_smooth = smoother(pred_01.clamp(0.001, 0.999))

    return transforms.ToPILImage()(pred_smooth.squeeze(0).cpu())


def batch_eliminate(input_folder, output_folder, checkpoint=CKPT_BEST):
    """
    Eliminate fingerprints from all images in a folder.

    Usage
    -----
    batch_eliminate('progan_images/', 'progan_untraceable/')
    batch_eliminate('stylegan_images/', 'stylegan_untraceable/')
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Processing {len(files)} images from {input_folder!r} ...")
    for i, fname in enumerate(files):
        out = eliminate_fingerprint(
            os.path.join(input_folder, fname), checkpoint)
        out.save(os.path.join(output_folder, fname))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} done")
    print(f"Saved to {output_folder}/")


if os.path.exists(CKPT_BEST):
    info = torch.load(CKPT_BEST, map_location="cpu")
    ep   = info["epoch"] + 1
    bl   = info["best_loss"]
    hist = info.get("history", [])
    last = hist[-1] if hist else {}
    print("\neliminate_fingerprint() ready")
    print(f"  Checkpoint : {CKPT_BEST}")
    print(f"  Trained to : epoch {ep}")
    print(f"  Best loss  : {bl:.5f}")
    if last:
        print(f"  Last PSNR  : {last.get('psnr', 'N/A')} dB")
        print(f"  Last SSIM  : {last.get('ssim', 'N/A')}")
else:
    print("best_model.pth not found — run training loop first")


# ══════════════════════════════════════════════════════════════════
# CELL 14 — REFERENCE: EXPECTED VALUES
#
# With all fixes applied (batch=32, A6000, lr=2e-4, warm-up 5 ep):
#
#  Epoch   1 : Loss ~0.30-0.50  Perc ~0.20-0.50  PSNR ~22-25 dB
#              Perc must NOT spike > 1.0 — confirms FIX 1 working
#  Epoch   5 : Loss ~0.20-0.30  Perc ~0.15-0.30  PSNR ~25-27 dB  warm-up ends
#  Epoch  10 : Loss ~0.12-0.18  Perc ~0.10-0.20  PSNR ~27-29 dB
#  Epoch  30 : Loss ~0.07-0.10  Perc ~0.07-0.12  PSNR ~29-31 dB
#  Epoch  50 : Loss ~0.05-0.07  Perc ~0.05-0.09  PSNR ~30-32 dB
#  Epoch 100 : Loss ~0.03-0.05                   PSNR ~31-33 dB
#  Epoch 150 : Loss ~0.02-0.04                   PSNR ~32-34 dB
#
# SSIM:
#  Epoch  10 : ~0.82-0.87
#  Epoch  50 : ~0.89-0.93
#  Epoch 150 : ~0.92-0.96  (paper: 0.963)
#
# ASR on DNA-Det (expected after 150-200 epochs with all fixes):
#  ProGAN    : 80-97%   ← FIX 1 + FIX 5
#  StyleGAN  : 75-95%   ← FIX 1
#  SNGAN     : 80-95%   ← FIX 2
#  CramerGAN : 90-99%   ← was already 99.3%, stays
#  MMDGAN    : 90-100%  ← was already 100%, stays
#  Overall   : 83-97%
#
# ABORT CRITERIA (stop and re-check fixes if):
#  Perc > 1.0 at epoch 1 → FIX 1 not applied correctly
#  Loss not decreasing by epoch 5 → check checkpoint_dir is new folder
#  PSNR stuck below 25 dB at epoch 10 → check FIX 4 (weight init)
# ══════════════════════════════════════════════════════════════════
print("\nReference — expected loss / PSNR / SSIM (all fixes applied):")
print("  Epoch   1 : Loss ~0.30-0.50   Perc ~0.20-0.50   PSNR ~22-25 dB")
print("  Epoch  10 : Loss ~0.12-0.18   Perc ~0.10-0.20   PSNR ~27-29 dB")
print("  Epoch  50 : Loss ~0.05-0.07   Perc ~0.05-0.09   PSNR ~30-32 dB")
print("  Epoch 150 : Loss ~0.02-0.04                     PSNR ~32-34 dB")
print()
print("ABORT if Perc > 1.0 at epoch 1  → FIX 1 (float32 cast) not applied")
print("ABORT if loss flat after ep 5   → wrong checkpoint_dir (using old run)")
print("ABORT if PSNR < 25 at epoch 10  → FIX 4 (weight init) not applied")