"""
DCGAN Architecture A — Two-Phase Training + Image Generation
--------------------------------------------------------------
Second DCGAN variant used for ForensicNet dataset generation.
Produces 128x128 face images using a deeper generator with label smoothing.

Key differences from dcgan_generate.py:
    LATENT_DIM : 100  (vs 128)
    IMG_SIZE   : 128  (vs 64)
    BATCH_SIZE : 64   (vs 32)
    Generator  : 6 layers with extra 32-channel layer (vs 5)
    Training   : Two phases (1-30, 31-60) with checkpoint resume
    Label smooth: 0.9 on real labels (stabilizes discriminator)
    Output name : ArchA_ep{epoch}_idx{i}.png

Dataset : CelebA — https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Usage:
    # Phase 1 (epochs 1-30)
    python dcgan_archA_generate.py --data_dir ./raw_inputs/celeba --output_dir ./raw_inputs/DCGAN_A --phase 1

    # Phase 2 (epochs 31-60, resumes from checkpoint)
    python dcgan_archA_generate.py --data_dir ./raw_inputs/celeba --output_dir ./raw_inputs/DCGAN_A --phase 2

    # Generate only using saved weights
    python dcgan_archA_generate.py --data_dir ./raw_inputs/celeba --output_dir ./raw_inputs/DCGAN_A \
                                   --generate_only --ckpt ./checkpoints/dcgan_archA_latest.pth
"""

import os
import gc
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
LATENT_DIM = 100
IMG_SIZE   = 128
BATCH_SIZE = 64


# ─── DATASET ──────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, img_dir, transform, max_images=50000):
        self.images  = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:max_images]
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        return self.transform(img)


# ─── GENERATOR ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,  3,   4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z.view(-1, LATENT_DIM, 1, 1))


# ─── DISCRIMINATOR ────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # inplace=False prevents RuntimeError during backward pass
        self.main = nn.Sequential(
            nn.Conv2d(3,   64,  4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(512, 1,   8, 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def run_training(args, start_ep, end_ep, device, dataloader):
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Phase 2 uses lower D lr for competitive balance
    lr_G = 0.0002
    lr_D = 0.0002 if start_ep == 1 else 0.0001

    optG      = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optD      = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Resume from checkpoint if available
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        optG.load_state_dict(ckpt['optG'])
        optD.load_state_dict(ckpt['optD'])
        print(f"Checkpoint loaded — resuming from epoch {start_ep}")

    print(f"\nTraining Phase: Epoch {start_ep} to {end_ep}  |  lr_G={lr_G}  lr_D={lr_D}")

    for epoch in range(start_ep, end_ep + 1):
        d_losses, g_losses = [], []
        G.train(); D.train()

        for real_imgs in tqdm(dataloader, desc=f"Epoch {epoch}/{end_ep}", leave=False):
            if isinstance(real_imgs, (list, tuple)):
                real_imgs = real_imgs[0]
            real_imgs  = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # ── Train Discriminator ──────────────────────────
            optD.zero_grad()
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # label smoothing
            fake_labels = torch.zeros((batch_size, 1), device=device)

            loss_real = criterion(D(real_imgs), real_labels)

            z         = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = G(z)
            loss_fake = criterion(D(fake_imgs.detach()), fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optD.step()

            # ── Train Generator ──────────────────────────────
            optG.zero_grad()
            gen_labels = torch.ones((batch_size, 1), device=device)
            loss_G     = criterion(D(fake_imgs), gen_labels)
            loss_G.backward()
            optG.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

        print(f"Epoch [{epoch}/{end_ep}]  D: {np.mean(d_losses):.4f}  G: {np.mean(g_losses):.4f}")

        # ── Save sample images at milestone epochs ───────────
        if epoch in [1, 10, 30, 60]:
            _save_samples(G, args.output_dir, epoch, device, n=500)

        # ── Save checkpoint every epoch ──────────────────────
        torch.save({
            'epoch': epoch,
            'G': G.state_dict(), 'D': D.state_dict(),
            'optG': optG.state_dict(), 'optD': optD.state_dict()
        }, args.ckpt)

    return G


# ─── SAMPLE SAVING ────────────────────────────────────────────────────────────
def _save_samples(G, output_dir, epoch, device, n=500):
    os.makedirs(output_dir, exist_ok=True)
    G.eval()
    with torch.no_grad():
        z    = torch.randn(n, LATENT_DIM, device=device)
        imgs = ((G(z) + 1) / 2 * 255).clamp(0, 255).cpu().byte()
        for i in range(imgs.size(0)):
            img_np = imgs[i].permute(1, 2, 0).numpy()
            Image.fromarray(img_np).save(
                os.path.join(output_dir, f"ArchA_ep{epoch}_idx{i}.png")
            )
    G.train()
    print(f"  Saved {n} samples for epoch {epoch} → {output_dir}")


# ─── GENERATE ONLY ────────────────────────────────────────────────────────────
def generate_only(args, device):
    G = Generator().to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location=device)['G'])
    print(f"Loaded generator from {args.ckpt}")
    _save_samples(G, args.output_dir, epoch="final", device=device, n=args.num_imgs)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN Architecture A — two-phase training")
    parser.add_argument("--data_dir",      required=True,                                    help="Folder of real face images (CelebA etc.)")
    parser.add_argument("--output_dir",    default="./raw_inputs/DCGAN_A",                   help="Where to save generated images")
    parser.add_argument("--ckpt",          default="./checkpoints/dcgan_archA_latest.pth",   help="Checkpoint path (save/resume)")
    parser.add_argument("--phase",         type=int, choices=[1, 2], default=1,              help="Training phase: 1 = epochs 1-30, 2 = epochs 31-60")
    parser.add_argument("--generate_only", action="store_true",                              help="Skip training, generate images from existing checkpoint")
    parser.add_argument("--num_imgs",      type=int, default=500,                            help="Images to generate in generate_only mode (default: 500)")
    args = parser.parse_args()

    os.makedirs(args.output_dir,                      exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt) or ".",    exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if args.generate_only:
        generate_only(args, device)
    else:
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        dataset    = FaceDataset(args.data_dir, transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        print(f"Dataset : {len(dataset)} images from {args.data_dir}")

        if args.phase == 1:
            run_training(args, start_ep=1,  end_ep=30, device=device, dataloader=dataloader)
        else:
            run_training(args, start_ep=31, end_ep=60, device=device, dataloader=dataloader)
