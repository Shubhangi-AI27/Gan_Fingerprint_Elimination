"""
Step 2 — DNA-Det Training (Fixed Version)
Fixes from original:
  1. Removed double-weighting (sampler + class_weights conflict)
  2. 60 epochs instead of 35
  3. Per-class accuracy logged every epoch
  4. Best model saved on min per-class accuracy not overall accuracy
  5. Early collapse detection with warnings
"""

import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ══════════════════════════════════════════════════════
# EDIT ONLY THESE TWO PATHS
# ══════════════════════════════════════════════════════
ANN_DIR  = "/data1/intern/dnadet_new/annotations"
SAVE_DIR = "/data1/intern/dnadet_new/saved_model"
# ══════════════════════════════════════════════════════

CONFIG = {
    "image_size"  : 128,
    "batch_size"  : 32,
    "num_epochs"  : 60,
    "lr"          : 1e-4,
    "weight_decay": 1e-4,
    "num_workers" : 4,
    "seed"        : 42,
    "class_names" : ["Real", "ProGAN", "MMDGAN", "SNGAN", "StyleGAN", "CramerGAN"],
    "ann_dir"     : ANN_DIR,
    "save_dir"    : SAVE_DIR,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*60)
print("STEP 2 — DNA-Det Training (Fixed)")
print("="*60)
print(f"  Device  : {device}")
print(f"  Epochs  : {CONFIG['num_epochs']}")
print(f"  Classes : {CONFIG['class_names']}")
print(f"  Saving  : {SAVE_DIR}")


# ══════════════════════════════════════════════════════
# FORENSIC PREPROCESSING
# ══════════════════════════════════════════════════════
def extract_forensic_residuals(img_pil, size=128):
    img  = img_pil.resize((size, size), Image.LANCZOS)
    gray = img.convert('L')
    arr  = np.array(gray).astype(np.float32)

    # Channel R — noise residuals
    b1 = np.array(gray.filter(ImageFilter.GaussianBlur(1))).astype(np.float32)
    b2 = np.array(gray.filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
    r  = np.clip((arr-b1)*2.0 + (arr-b2)*1.0 + 128, 0, 255).astype(np.uint8)

    # Channel G — FFT magnitude
    win     = np.outer(np.hanning(size), np.hanning(size))
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(arr * win)))
    log_mag = np.log1p(fft_mag)
    g = ((log_mag - log_mag.min()) /
         (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)

    # Channel B — local variance
    mean    = uniform_filter(arr,    size=5)
    mean_sq = uniform_filter(arr**2, size=5)
    std     = np.sqrt(np.clip(mean_sq - mean**2, 0, None))
    b = ((std / (std.max() + 1e-8)) * 255).astype(np.uint8)

    return Image.fromarray(np.stack([r, g, b], axis=2), 'RGB')


# ══════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════
class GANDataset(Dataset):
    def __init__(self, ann_file, is_train=False):
        with open(ann_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.samples  = [(l.split("\t")[0], int(l.split("\t")[1]))
                         for l in lines]
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = extract_forensic_residuals(img, CONFIG["image_size"])
        except Exception:
            img = Image.new("RGB",
                (CONFIG["image_size"], CONFIG["image_size"]), (128,128,128))
        return self.transform(img), label


# ══════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg     = nn.AdaptiveAvgPool2d(1)
        self.max     = nn.AdaptiveMaxPool2d(1)
        self.fc      = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.fc(self.avg(x).view(b, c))
        m = self.fc(self.max(x).view(b, c))
        return x * self.sigmoid(a + m).view(b, c, 1, 1)


class DNADetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.ca  = ChannelAttention(out_ch)
        self.act = nn.ReLU(True)
    def forward(self, x):
        return self.act(self.ca(self.conv(x)) + self.shortcut(x))


class DNADet(nn.Module):
    def __init__(self, class_num=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.stage1 = DNADetBlock(64,  128, stride=2)
        self.stage2 = DNADetBlock(128, 256, stride=2)
        self.stage3 = DNADetBlock(256, 512, stride=2)
        self.stage4 = DNADetBlock(512, 512, stride=2)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num),
        )
    def forward(self, x):
        f = self.stem(x)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)
        f = self.stage4(f)
        feat = self.gap(f)
        return self.classifier(feat), feat.flatten(1)


# ══════════════════════════════════════════════════════
# DATALOADERS
# FIX: WeightedRandomSampler ONLY — no class_weights in loss
# ══════════════════════════════════════════════════════
print("\nLoading datasets...")
train_ds = GANDataset(os.path.join(ANN_DIR, "train.txt"), is_train=True)
val_ds   = GANDataset(os.path.join(ANN_DIR, "val.txt"),   is_train=False)
test_ds  = GANDataset(os.path.join(ANN_DIR, "test.txt"),  is_train=False)

labels = [s[1] for s in train_ds.samples]
counts = np.bincount(labels, minlength=6)

print("\n  Per-class counts in train set:")
for name, cnt in zip(CONFIG["class_names"], counts):
    print(f"    {name:12s} : {cnt:5d}")

# FIX 1 — sampler only, no class weights in loss
weights = [1.0 / counts[l] for l in labels]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(
    train_ds, batch_size=CONFIG["batch_size"],
    sampler=sampler, num_workers=CONFIG["num_workers"], pin_memory=True)
val_loader = DataLoader(
    val_ds, batch_size=CONFIG["batch_size"],
    shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
test_loader = DataLoader(
    test_ds, batch_size=CONFIG["batch_size"],
    shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)


# ══════════════════════════════════════════════════════
# MODEL + LOSS + OPTIMIZER
# FIX 1 — plain CrossEntropy, no weight= parameter
# ══════════════════════════════════════════════════════
model     = DNADet(class_num=6).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

print(f"\n  Params : {sum(p.numel() for p in model.parameters()):,}")
print(f"  Loss   : CrossEntropy (label_smoothing=0.1, no class weights)")
print(f"  Sampler: WeightedRandomSampler (balances all 6 classes)")


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def train_epoch(loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(loader, desc="  Train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out, _ = model(imgs)
        loss   = criterion(out, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == lbls).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# FIX 3 — per-class accuracy every epoch
def per_class_accuracy(loader):
    model.eval()
    correct = np.zeros(6)
    total   = np.zeros(6)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out, _  = model(imgs)
            preds   = out.argmax(1)
            for c in range(6):
                mask       = (lbls == c)
                total[c]  += mask.sum().item()
                correct[c]+= (preds[mask] == c).sum().item()
    acc = correct / (total + 1e-8)
    return acc, correct.sum() / total.sum()


def save_checkpoint(epoch, val_acc, per_cls, tag=""):
    path = os.path.join(SAVE_DIR, f"checkpoint{tag}.pth")
    torch.save({
        "epoch"      : epoch,
        "model"      : model.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        "scheduler"  : scheduler.state_dict(),
        "val_acc"    : val_acc,
        "per_cls_acc": per_cls.tolist(),
        "config"     : CONFIG,
    }, path)
    return path


# ══════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════
history = {
    "train_loss"    : [],
    "train_acc"     : [],
    "val_acc"       : [],
    "per_class_val" : [],
    "min_class_acc" : [],
}
best_min_cls_acc = 0.0
best_path        = os.path.join(SAVE_DIR, "best_model.pth")

print("\n" + "="*95)
print(f"TRAINING — {CONFIG['num_epochs']} epochs | 128×128 | Fixed weighting")
print("="*95)
header = (f"{'Ep':>4} | {'Loss':>7} | {'Train':>7} | {'Val':>7} | "
          f"{'MinCls':>7} | "
          + "  ".join([f"{n[:6]:>6}" for n in CONFIG["class_names"]]))
print(header)
print("-"*95)

for epoch in range(1, CONFIG["num_epochs"] + 1):
    t0 = time.time()

    tr_loss, tr_acc  = train_epoch(train_loader)
    per_cls, vl_acc  = per_class_accuracy(val_loader)
    scheduler.step()

    min_cls = per_cls.min()
    history["train_loss"].append(round(tr_loss, 6))
    history["train_acc"].append(round(tr_acc, 6))
    history["val_acc"].append(round(vl_acc, 6))
    history["per_class_val"].append(per_cls.tolist())
    history["min_class_acc"].append(round(float(min_cls), 6))

    # FIX 3 — save best on min per-class acc not overall acc
    flag = ""
    if min_cls > best_min_cls_acc:
        best_min_cls_acc = min_cls
        torch.save(model.state_dict(), best_path)
        flag = " ✅"

    cls_str = "  ".join([f"{a*100:6.1f}" for a in per_cls])
    elapsed = time.time() - t0
    print(f"{epoch:4d} | {tr_loss:7.4f} | {tr_acc*100:6.2f}% | "
          f"{vl_acc*100:6.2f}% | {min_cls*100:6.2f}%  | "
          f"{cls_str}  {elapsed:.0f}s{flag}")

    # Early collapse detection
    if epoch >= 10:
        for i, (name, acc) in enumerate(zip(CONFIG["class_names"], per_cls)):
            if acc < 0.10:
                print(f"  ⚠️  COLLAPSE WARNING: {name} = {acc*100:.1f}% "
                      f"after epoch {epoch} — check your data!")

    # Checkpoint every 10 epochs
    if epoch % 10 == 0:
        p = save_checkpoint(epoch, vl_acc, per_cls, tag=f"_ep{epoch}")
        print(f"  💾 Saved → {p}")

    # Save history every epoch
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

# Save final model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

print(f"\nBest min-class val accuracy : {best_min_cls_acc*100:.2f}%")


# ══════════════════════════════════════════════════════
# TRAINING CURVES
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train_loss"], "b-", lw=2)
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3)

axes[1].plot([a*100 for a in history["train_acc"]], label="Train")
axes[1].plot([a*100 for a in history["val_acc"]],   label="Val")
axes[1].set_title("Overall Accuracy (%)")
axes[1].legend(); axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3)

per_cls_arr = np.array(history["per_class_val"]) * 100
for i, name in enumerate(CONFIG["class_names"]):
    axes[2].plot(per_cls_arr[:, i], label=name)
axes[2].axhline(y=70, color="red", linestyle="--", alpha=0.5, label="70% target")
axes[2].set_title("Per-Class Val Accuracy (%)")
axes[2].legend(fontsize=8); axes[2].set_xlabel("Epoch")
axes[2].grid(alpha=0.3)

plt.suptitle(f"DNA-Det Training | Best min-class: {best_min_cls_acc*100:.2f}%",
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150,
            bbox_inches="tight")
plt.show()
print(f"Curves saved → {SAVE_DIR}/training_curves.png")


# ══════════════════════════════════════════════════════
# TEST EVALUATION
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST EVALUATION")
print("="*60)

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Test"):
        out, _ = model(imgs.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(lbls.numpy())

print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=CONFIG["class_names"],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CONFIG["class_names"],
            yticklabels=CONFIG["class_names"])
plt.title("DNA-Det Confusion Matrix (Fixed Version)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print(f"Confusion matrix saved → {SAVE_DIR}/confusion_matrix.png")


# ══════════════════════════════════════════════════════
# FINAL DIAGNOSIS
# ══════════════════════════════════════════════════════
per_cls_test, overall_test = per_class_accuracy(test_loader)

print("\n" + "="*50)
print("  FINAL RESULT — Per-class test accuracy")
print("="*50)
all_pass = True
for name, acc in zip(CONFIG["class_names"], per_cls_test):
    status = "✓" if acc >= 0.70 else "✗"
    if acc < 0.70:
        all_pass = False
    print(f"  {name:12s} : {acc*100:6.2f}%  {status}")
print("="*50)
print(f"  Overall      : {overall_test*100:.2f}%")
print(f"  Min class    : {per_cls_test.min()*100:.2f}%")
print("="*50)
if all_pass:
    print("  ✅ READY — all classes > 70%, proceed to Fix 2")
else:
    worst = CONFIG["class_names"][per_cls_test.argmin()]
    print(f"  ❌ NOT READY — {worst} is below 70%")
    print(f"     Share the confusion matrix and I will diagnose")
print("="*50)


# ══════════════════════════════════════════════════════
# LIST ALL SAVED FILES
# ══════════════════════════════════════════════════════
print("\nSaved files:")
for fname in sorted(os.listdir(SAVE_DIR)):
    kb = os.path.getsize(os.path.join(SAVE_DIR, fname)) / 1024
    print(f"  {fname:45s} {kb:8.1f} KB")
print(f"\n✅ Done! All files in → {SAVE_DIR}")