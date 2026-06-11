"""
Step 1 — Annotation Generator
Run this first. Generates train/val/test annotation files. Resizes images to 128×128.

Dataset Sources :
MMDGAN : https://github.com/ICTMCG/DNA-Det
SNGAN  : https://github.com/ICTMCG/DNA-Det
CramerGAN : https://github.com/ICTMCG/DNA-Det
StyleGAN :  https://github.com/ksmolko/stylegan-detector

StyleGAN2-ADA (additional batch):
     Repo      : https://github.com/NVlabs/stylegan2-ada-pytorch
     Weights   : https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
     Generated : 2000 images at 1024×1024, truncation_psi=0.7
     Script    : generation_scripts/stylegan2_generate.py

ProGAN (Custom Generation Pipeline):
     Weights   : TensorFlow Hub CelebA-HQ (https://tfhub.dev/google/progan-128/1)
     Generated : 3000-5000 images at 128×128 resolution using pre-trained weights.
     Template  : PyTorch structural architecture implementation also provided.
     Scripts   : generation_scripts/progan_tfhub_generate.py & progan_tfhub_5000images.py
Only edit the paths in CLASS_DIR_MAP, ANN_DIR, and RESIZED_DIR.
"""

import os, random
from pathlib import Path
from PIL import Image

# ══════════════════════════════════════════════════════
# EDIT ONLY THESE PATHS
# ══════════════════════════════════════════════════════
BASE_DIR = os.getcwd()
CLASS_DIR_MAP = {
    "Real"      : [os.path.join(BASE_DIR, "dataset/Real")],
    "ProGAN"    : [os.path.join(BASE_DIR, "dataset/GANs/ProGAN/celeba_align_png_cropped")],
    "MMDGAN"    : [os.path.join(BASE_DIR, "dataset/GANs/MMDGAN")],
    "SNGAN"     : [os.path.join(BASE_DIR, "dataset/GANs/SNGAN")],
    "StyleGAN"  : [os.path.join(BASE_DIR, "dataset/GANs/StyleGAN")],
    "CramerGAN" : [os.path.join(BASE_DIR, "dataset/GANs/CramerGAN")],
}
ANN_DIR     = os.path.join(BASE_DIR, "annotations")
RESIZED_DIR = os.path.join(BASE_DIR, "resized_128")
IMAGES_PER_CLASS = 5000
SPLIT            = (0.70, 0.15, 0.15)
SEED             = 42
TARGET_SIZE      = (128, 128)


CLASS_NAMES = ["Real", "ProGAN", "MMDGAN", "SNGAN", "StyleGAN", "CramerGAN"]
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(SEED)
os.makedirs(ANN_DIR, exist_ok=True)
os.makedirs(RESIZED_DIR, exist_ok=True)

print("="*60)
print("STEP 1 — Annotation Generator (with 128×128 resize)")
print("="*60)

# ── Verify paths exist first ───────────────────────────
print("\nChecking paths...")
for cls, folders in CLASS_DIR_MAP.items():
    for folder in folders:
        exists = "✓" if os.path.exists(folder) else "✗ NOT FOUND"
        print(f"  {cls:12s} | {folder} | {exists}")

print()

# ── Resize helper ──────────────────────────────────────
def resize_and_save(src_path: Path, cls: str) -> Path:
    dest_dir = Path(RESIZED_DIR) / cls
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use parent folder name as prefix to prevent filename collisions
    unique_name = f"{src_path.parent.name}_{src_path.name}"
    dest_path   = dest_dir / unique_name

    if not dest_path.exists():
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")            # normalise palette/RGBA/grayscale
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                img.save(dest_path)
        except Exception as e:
            print(f"    Could not resize {src_path}: {e}")
            return None

    return dest_path

# ── Collect images per class ───────────────────────────
all_samples  = []
class_counts = {}

for label, cls in enumerate(CLASS_NAMES):
    folders = CLASS_DIR_MAP[cls]
    images  = []

    for folder in folders:
        p = Path(folder)
        if not p.exists():
            print(f"    SKIPPING (not found): {folder}")
            continue
        found = [f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS]
        images.extend(found)
        print(f"  {cls:12s} | {len(found):6d} images | {folder}")

    if len(images) == 0:
        print(f"   No images found for {cls} — fix your path!")
        continue

    random.shuffle(images)
    selected = images[:IMAGES_PER_CLASS]

    # ── Resize selected images ─────────────────────────
    print(f"  {cls:12s} | Resizing {len(selected)} images to {TARGET_SIZE[0]}px …")
    resized_paths = []
    skipped = 0
    for i, src in enumerate(selected):
        dest = resize_and_save(src, cls)
        if dest is not None:
            resized_paths.append(dest)
        else:
            skipped += 1
        if (i + 1) % 500 == 0:
            print(f"    … {i+1}/{len(selected)} done")

    if skipped:
        print(f"    {skipped} images skipped due to errors")

    class_counts[cls] = len(resized_paths)
    print(f"  {cls:12s} | Total: {len(images):6d} → Selected: {len(selected)} → Resized: {len(resized_paths)}\n")

    for img_path in resized_paths:
        all_samples.append((str(img_path), label))

# ── Check balance ──────────────────────────────────────
print("\nClass balance check:")
for cls, cnt in class_counts.items():
    bar = "█" * (cnt // 100)
    print(f"  {cls:12s} : {cnt:5d}  {bar}")

if len(set(class_counts.values())) > 1:
    print("\n    WARNING: Classes are not balanced!")
    print("     Check that all folders have enough images.")
else:
    print("\n   All classes balanced")

# ── Shuffle and split ──────────────────────────────────
random.shuffle(all_samples)
n     = len(all_samples)
n_tr  = int(n * SPLIT[0])
n_val = int(n * SPLIT[1])

train = all_samples[:n_tr]
val   = all_samples[n_tr : n_tr + n_val]
test  = all_samples[n_tr + n_val:]

# ── Write annotation files ─────────────────────────────
for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
    out_path = os.path.join(ANN_DIR, f"{split_name}.txt")
    with open(out_path, "w") as f:
        for path, label in split_data:
            f.write(f"{path}\t{label}\n")
    print(f"  ✓ {split_name:6s}.txt → {len(split_data):6d} samples → {out_path}")

# ── Final summary ──────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Total images : {n}")
print(f"  Train        : {len(train)}")
print(f"  Val          : {len(val)}")
print(f"  Test         : {len(test)}")
print(f"  Image size   : {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px (saved to {RESIZED_DIR})")
print(f"  Annotations  : {ANN_DIR}")
print("="*60)

# ── Sanity check — print 2 lines from each file ───────
for split_name in ["train", "val", "test"]:
    print(f"\n--- {split_name}.txt (first 2 lines) ---")
    with open(os.path.join(ANN_DIR, f"{split_name}.txt")) as f:
        for i, line in enumerate(f):
            if i == 2: break
            print(" ", line.strip())

print("\n Step 1 complete. Run step2_train_dnadet.py next.")
