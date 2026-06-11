# ProGAN Generator - Version 1
# Generated 5000 images
# Used in Attribution Model Dataset
# Works on Kaggle, Google Colab and Local Machine

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from tqdm import tqdm
import os, shutil

# ── 1. AUTO-DETECT PLATFORM ───────────────────────────────────────────
if os.path.exists('/kaggle/working'):
    WORKING_DIR = '/kaggle/working'        # Kaggle
elif os.path.exists('/content'):
    WORKING_DIR = '/content'               # Google Colab
else:
    WORKING_DIR = os.path.join(os.getcwd(), 'output')  # Local Machine

print(f"Platform detected. Working directory: {WORKING_DIR}")

# ── 2. DEFINE PATHS ───────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(WORKING_DIR, 'progan_generated_v2')
NUM_IMAGES = 5000
BATCH      = 16
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 3. LOAD MODEL ─────────────────────────────────────────────────────
print("Loading ProGAN from TF Hub...")
model    = hub.load('https://tfhub.dev/google/progan-128/1')
generate = model.signatures['default']
print("Model loaded!")

# ── 4. GENERATE IMAGES ────────────────────────────────────────────────
count = 0
print(f"Generating {NUM_IMAGES} ProGAN face images...")
for _ in tqdm(range(0, NUM_IMAGES, BATCH)):
    bs  = min(BATCH, NUM_IMAGES - count)
    if bs <= 0: break
    z   = tf.random.normal([bs, 512])
    out = generate(latent_vector=z)

    imgs = out['default'].numpy()
    imgs = ((imgs + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    for i in range(bs):
        Image.fromarray(imgs[i]).save(
            os.path.join(OUTPUT_DIR, f'progan_{count:05d}.png')
        )
        count += 1

# ── 5. ZIP GENERATED IMAGES ───────────────────────────────────────────
ZIP_PATH = os.path.join(WORKING_DIR, 'progan_5000')
print(f"\nZipping images...")
shutil.make_archive(ZIP_PATH, 'zip', OUTPUT_DIR)

print(f"\nDone! {count} images saved to {OUTPUT_DIR}")
print(f"ZIP file saved to {ZIP_PATH}.zip")
