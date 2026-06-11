import os
import shutil
import sys
import torch
import pickle
import requests
from PIL import Image
from tqdm import tqdm
from IPython.display import display, FileLink
# --- 1. ENVIRONMENT & REPOSITORY SETUP ---
REPO_NAME = 'stylegan2-ada-pytorch'
REPO_URL = f'https://github.com/NVlabs/{REPO_NAME}.git'
CHECKPOINT_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
CHECKPOINT_PATH = '/kaggle/working/ffhq.pkl'
# Clone Repo if missing
if not os.path.exists(REPO_NAME):
    print("Cloning StyleGAN2-ADA repository...")
    !git clone {REPO_URL}
# Download Weights if missing
if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading FFHQ model weights...")
    r = requests.get(CHECKPOINT_URL, allow_redirects=True)
    with open(CHECKPOINT_PATH, 'wb') as f:
        f.write(r.content)
# Move folders to main directory to prevent ModuleNotFoundError
for folder in ['dnnlib', 'torch_utils']:
    src = os.path.join('/kaggle/working', REPO_NAME, folder)
    dest = os.path.join('/kaggle/working', folder)
    if os.path.exists(src):
        if os.path.exists(dest): shutil.rmtree(dest)
        shutil.copytree(src, dest)
# Bypassing specialized CUDA kernels for Kaggle compatibility
import torch_utils.ops.bias_act as bias_act
import torch_utils.ops.upfirdn2d as upfirdn2d
bias_act._init = lambda: False
upfirdn2d._init = lambda: False
# --- 2. CONFIGURATION ---
TOTAL_IMAGES = 2000
START_INDEX = 4500  # Continues numbering from your previous 4500
OUTPUT_DIR = '/kaggle/working/batch_2000_highres'
ZIP_NAME = '/kaggle/working/shubhangi_dataset_2000'
device = torch.device('cuda')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- 3. LOAD MODEL ---
print("Loading model G...")
with open(CHECKPOINT_PATH, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
print("✓ Model ready.")
# --- 4. GENERATION LOOP ---
print(f" Starting generation of {TOTAL_IMAGES} high-quality faces...")
with torch.no_grad():
    for i in tqdm(range(TOTAL_IMAGES)):
        # Generate random latent vector
        z = torch.randn([1, G.z_dim]).to(device)

        # Generate the image (truncation 0.7 for best realism)
        img = G(z, None, truncation_psi=0.7, noise_mode='const')

        # Post-process: Convert Tensor to PIL Image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL_img = Image.fromarray(img[0].cpu().numpy(), 'RGB')

        # Save at full 1024x1024 resolution
        save_path = os.path.join(OUTPUT_DIR, f'face_{START_INDEX + i:05d}.png')
        PIL_img.save(save_path)

        # Show a preview of the very first image to confirm quality
        if i == 0:
            print(f"Preview of first generated image (Index {START_INDEX}):")
            display(PIL_img.resize((256, 256)))
# --- 5. COMPRESS & DOWNLOAD ---
print("\n Packaging images into ZIP... this may take 3-5 minutes.")
shutil.make_archive(ZIP_NAME, 'zip', OUTPUT_DIR)
shutil.rmtree(OUTPUT_DIR) # Delete loose images to save disk space
print(f"\n SUCCESS! Generated images {START_INDEX} to {START_INDEX + TOTAL_IMAGES - 1}")
display(FileLink(f'{ZIP_NAME.split("/")[-1]}.zip'))
