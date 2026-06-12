"""
Step 1 — Dataset Builder (ForensicNet)
Builds a 4-class dataset for the custom ForensicNet attribution model.

Classes : Real, ProGAN, StyleGAN, DCGAN
Images  : 5,000 per class (20,000 total)
Output  : dataset_v5/ with one subfolder per class

Image Sources:
  Real    : CelebA dataset — https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  ProGAN  : generation_scripts/progan_thub_5000images.py
  StyleGAN: 
  DCGAN   : Custom trained on CelebA (see generate/dcgan_generate.py)
"""

import os
import shutil
import random
from PIL import Image

BASE_INPUT_DIR = "./raw_inputs"

SOURCES = {
    'DCGAN'   : os.path.join(BASE_INPUT_DIR, "DCGAN"),
    'StyleGAN': os.path.join(BASE_INPUT_DIR, "StyleGAN"),
    'ProGAN'  : os.path.join(BASE_INPUT_DIR, "ProGAN"),
    'Real'    : os.path.join(BASE_INPUT_DIR, "Real"),
}

BASE_DIR   = "./dataset_v5"
TARGET_RES = 128
PER_CLASS  = 5000

random.seed(42)


def collect(folder):
    imgs = []
    if not os.path.exists(folder):
        print(f"  MISSING: {folder}")
        return imgs
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            imgs.append(os.path.join(folder, f))
    return imgs


if __name__ == "__main__":

    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    print("Building dataset_v5...")
    for label, folder in SOURCES.items():
        out_dir = os.path.join(BASE_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        all_imgs = collect(folder)
        random.shuffle(all_imgs)
        selected = all_imgs[:PER_CLASS]

        saved = 0
        for i, src in enumerate(selected):
            try:
                with Image.open(src).convert("RGB") as img:
                    img = img.resize((TARGET_RES, TARGET_RES), Image.LANCZOS)
                    img.save(os.path.join(out_dir, f"{label}_{i:05d}.png"))
                    saved += 1
            except Exception as e:
                print(f"  Skipped {src}: {e}")

        print(f"  {label}: {saved} images saved")

    print("\nDataset compilation complete. Final counts:")
    for label in SOURCES:
        d = os.path.join(BASE_DIR, label)
        if os.path.exists(d):
            print(f"  {label}: {len(os.listdir(d))}")
