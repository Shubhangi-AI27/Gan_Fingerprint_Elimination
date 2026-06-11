"""
Defines the forensic feature extractor and ForensicNet classifier.

ForensicNet uses EfficientNet-B0 as backbone, taking 3-channel
forensic residual images as input instead of raw RGB.

Forensic channels:
    R → Noise residual (high-frequency details)
    G → FFT magnitude spectrum (frequency fingerprint)
    B → Local standard deviation (texture noise map)
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def extract_forensic_residuals(img_pil):
    img  = img_pil.resize((128, 128), Image.LANCZOS)
    gray = img.convert('L')
    arr  = np.array(gray).astype(np.float32)

    # --- Channel R: Noise Residual ---
    b1 = np.array(gray.filter(ImageFilter.GaussianBlur(1))).astype(np.float32)
    b2 = np.array(gray.filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
    r  = np.clip((arr - b1) * 2.0 + (arr - b2) * 1.0 + 128, 0, 255).astype(np.uint8)

    # --- Channel G: FFT Magnitude Spectrum ---
    win     = np.outer(np.hanning(128), np.hanning(128))
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(arr * win)))
    log_mag = np.log1p(fft_mag)
    g       = ((log_mag - log_mag.min()) /
               (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)

    # --- Channel B: Local Standard Deviation ---
    mean    = uniform_filter(arr,    size=5)
    mean_sq = uniform_filter(arr**2, size=5)
    std     = np.sqrt(np.clip(mean_sq - mean**2, 0, None))
    b       = ((std / (std.max() + 1e-8)) * 255).astype(np.uint8)

    return Image.fromarray(np.stack([r, g, b], axis=2), 'RGB')


class ForensicNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Quick sanity check
    model = ForensicNet(num_classes=4)
    dummy = torch.randn(2, 3, 128, 128)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: torch.Size([2, 4])
    print("ForensicNet defined successfully")
