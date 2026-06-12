"""
DCGAN — Training + Image Generation
-------------------------------------
Trains a DCGAN on face images (CelebA or similar) and generates
synthetic face images for use as the DCGAN class in ForensicNet.

Dataset : CelebA — https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
          Any folder of face images works (JPG/PNG).

Usage:
    # Train and generate
    python dcgan_generate.py --data_dir ./raw_inputs/celeba --output_dir ./raw_inputs/DCGAN

    # Generate only (skip training, load existing weights)
    python dcgan_generate.py --data_dir ./raw_inputs/celeba --output_dir ./raw_inputs/DCGAN \
                             --ckpt ./checkpoints/dcgan_generator_final.pth --generate_only

Output:
    ./raw_inputs/DCGAN/dcgan_00001.png ... dcgan_02000.png
    ./checkpoints/dcgan_generator_final.pth
"""

import os
import argparse
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
LATENT_DIM = 128
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 50
NUM_IMGS   = 2000


# ─── DATASET ──────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, img_dir, transform, max_images=30000):
        self.img_dir = img_dir
        self.images  = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:max_images]
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
        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),        nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),        nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1),        nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  3,   4, 2, 1),        nn.Tanh()
        )

    def forward(self, z):
        return self.model(z.view(-1, LATENT_DIM, 1, 1))


# ─── DISCRIMINATOR ────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,   64,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1,   4, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train(args, device):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset    = FaceDataset(args.data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Dataset    : {len(dataset)} images from {args.data_dir}")

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G     = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D     = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        for real_imgs in dataloader:
            real_imgs   = real_imgs.to(device)
            b           = real_imgs.size(0)
            real_labels = torch.ones(b, 1).to(device)
            fake_labels = torch.zeros(b, 1).to(device)

            # Train D
            z        = torch.randn(b, LATENT_DIM).to(device)
            fake_imgs = G(z)
            loss_D   = (criterion(D(real_imgs), real_labels) +
                        criterion(D(fake_imgs.detach()), fake_labels)) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Train G
            z      = torch.randn(b, LATENT_DIM).to(device)
            loss_G = criterion(D(G(z)), real_labels)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        print(f"Epoch {epoch+1}/{EPOCHS}  D: {loss_D.item():.4f}  G: {loss_G.item():.4f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch, 'G_state': G.state_dict(), 'D_state': D.state_dict(),
                'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict()
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    final_path = os.path.join(args.ckpt_dir, "dcgan_generator_final.pth")
    torch.save(G.state_dict(), final_path)
    print(f"\nFinal generator saved → {final_path}")
    return G


# ─── GENERATION ───────────────────────────────────────────────────────────────
def generate(G, args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    G.eval()
    generated = 0

    print(f"\nGenerating {NUM_IMGS} images → {args.output_dir}")
    with torch.no_grad():
        with tqdm(total=NUM_IMGS) as pbar:
            while generated < NUM_IMGS:
                batch = min(64, NUM_IMGS - generated)
                z     = torch.randn(batch, LATENT_DIM).to(device)
                imgs  = G(z).cpu()

                for img in imgs:
                    img_np = ((img.numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(
                        os.path.join(args.output_dir, f"dcgan_{generated+1:05d}.png")
                    )
                    generated += 1
                    pbar.update(1)

    print(f"Done. {generated} images saved to {args.output_dir}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN trainer and image generator")
    parser.add_argument("--data_dir",      required=True,                              help="Folder of real face images for training (CelebA etc.)")
    parser.add_argument("--output_dir",    default="./raw_inputs/DCGAN",               help="Where to save generated images")
    parser.add_argument("--ckpt_dir",      default="./checkpoints",                    help="Where to save model checkpoints")
    parser.add_argument("--ckpt",          default=None,                               help="Path to existing generator .pth (for generate_only mode)")
    parser.add_argument("--generate_only", action="store_true",                        help="Skip training, just generate using existing checkpoint")
    parser.add_argument("--num_imgs",      type=int, default=NUM_IMGS,                 help="Number of images to generate (default: 2000)")
    args = parser.parse_args()

    NUM_IMGS = args.num_imgs
    os.makedirs(args.ckpt_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if args.generate_only:
        if args.ckpt is None:
            raise ValueError("--ckpt required when using --generate_only")
        G = Generator().to(device)
        G.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded generator from {args.ckpt}")
    else:
        G = train(args, device)

    generate(G, args, device)
