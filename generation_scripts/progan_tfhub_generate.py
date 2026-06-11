import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tqdm import tqdm

# ── 1. DEFINE ABSOLUTE PATHS ──────────────────────────────────────────
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, 'dataset/GANs/ProGAN/celeba_align_png_cropped')
ZIP_OUT_PATH = os.path.join(BASE_DIR, 'progan_generated_samples')

os.makedirs(DATASET_DIR, exist_ok=True)

# ── 2. GENERATION LOGIC ───────────────────────────────────────────────
print("Loading ProGAN model...")
module = hub.load("https://tfhub.dev/google/progan-128/1")
infer = module.signatures["default"]

num_images = 3000
batch_size = 16
img_idx = 0

print(f"Generating images to: {DATASET_DIR}")
while img_idx < num_images:
    current_batch = min(batch_size, num_images - img_idx)
    latents = tf.random.normal([current_batch, 512])
    outputs = infer(latents)["default"]
    images = (outputs.numpy() * 255).astype(np.uint8)

    for i in range(current_batch):
        img = Image.fromarray(images[i])
        # Saving with absolute path
        img.save(os.path.join(DATASET_DIR, f"gen_progan_{img_idx:05d}.png"))
        img_idx += 1

# ── 3. ZIP WITH ABSOLUTE PATH ────────────────────────────────────────
print(f"Zipping images into {ZIP_OUT_PATH}.zip...")
# make_archive(output_filename, format, root_dir)
shutil.make_archive(ZIP_OUT_PATH, 'zip', DATASET_DIR)

print("\n--- DONE ---")
print(f"Check the sidebar for: {ZIP_OUT_PATH}.zip")
