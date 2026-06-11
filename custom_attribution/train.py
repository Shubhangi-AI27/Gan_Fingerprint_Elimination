"""
Trains ForensicNet (EfficientNet-B0) on the processed dataset.

Two-phase training:
    Phase 1 → head warmup (5 epochs,  classifier only)
    Phase 2 → full fine-tune (25 epochs, all layers)

Usage:
    python train.py --data_dir ./dataset_v5 --ckpt_path ./checkpoints/forensic_net.pth
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from models.forensic_net import ForensicNet, extract_forensic_residuals


def get_transforms(mode):
    base = [
        transforms.Lambda(extract_forensic_residuals),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
    if mode == 'train':
        base.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)


def get_class_weights(data_dir, categories, device):
    class_counts = []
    for label in categories:
        d = os.path.join(data_dir, label)
        class_counts.append(len(os.listdir(d)))
    total = sum(class_counts)
    n     = len(categories)
    weights = torch.tensor(
        [total / (n * c) for c in class_counts], dtype=torch.float
    ).to(device)
    return weights


def get_loaders(data_dir, batch_size=32):
    full_ds   = datasets.ImageFolder(root=data_dir)
    n         = len(full_ds)
    idx       = torch.randperm(n).tolist()
    train_idx = idx[:int(0.85 * n)]
    val_idx   = idx[int(0.85 * n):]

    train_ds = datasets.ImageFolder(root=data_dir, transform=get_transforms('train'))
    val_ds   = datasets.ImageFolder(root=data_dir, transform=get_transforms('val'))

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, full_ds.classes


def validate(model, val_loader, categories, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            labs.extend(labels.tolist())

    preds   = np.array(preds)
    labs    = np.array(labs)
    overall = 100 * (preds == labs).mean()

    print(f"  Overall val acc: {overall:.2f}%")
    for i, name in enumerate(categories):
        mask = labs == i
        if not mask.any():
            continue
        acc = 100 * (preds[mask] == i).mean()
        bar = chr(9608) * int(acc / 5)
        print(f"    {name:<12} {bar:<20} {acc:.1f}%")

    return overall


def train(args):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, categories = get_loaders(args.data_dir, args.batch_size)
    print(f"Classes: {categories}")

    class_weights = get_class_weights(args.data_dir, categories, device)
    print(f"Class weights: {dict(zip(categories, class_weights.cpu().tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    model     = ForensicNet(num_classes=len(categories)).to(device)

    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)

    # ── Phase 1: head warmup ──────────────────────────────────────────
    for name, p in model.named_parameters():
        if 'classifier' not in name:
            p.requires_grad = False

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

    print("\n=== Phase 1: head warmup (5 epochs) ===")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 | loss {total_loss/len(train_loader):.4f}")
        sch.step()

    validate(model, val_loader, categories, device)

    # ── Phase 2: full fine-tune ───────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = True

    opt = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    best = 0.0
    print("\n=== Phase 2: full fine-tune (25 epochs) ===")
    for epoch in range(25):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/25 | loss {total_loss/len(train_loader):.4f}")
        sch.step()

        acc = validate(model, val_loader, categories, device)
        if acc > best:
            best = acc
            torch.save({
                'model'     : model.state_dict(),
                'categories': categories
            }, args.ckpt_path)
            print(f"  --> saved (best: {best:.2f}%)")

    print(f"\nDone. Best val accuracy: {best:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./dataset_v5',                    help='Processed dataset folder')
    parser.add_argument('--ckpt_path',  default='./checkpoints/forensic_net.pth',  help='Where to save best checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,                      help='Batch size')
    args = parser.parse_args()

    train(args)
