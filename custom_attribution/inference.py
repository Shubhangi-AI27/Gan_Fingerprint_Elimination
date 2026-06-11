"""
Runs ForensicNet on a single image and prints prediction.

Usage:
    python inference.py --image ./test.jpg
    python inference.py --image ./test.jpg --ckpt ./checkpoints/forensic_net.pth
"""

import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from models.forensic_net import ForensicNet, extract_forensic_residuals


def load_model(ckpt_path, device):
    ckpt       = torch.load(ckpt_path, map_location=device)
    categories = ckpt['categories']
    model      = ForensicNet(num_classes=len(categories)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, categories


def get_transform():
    return transforms.Compose([
        transforms.Lambda(extract_forensic_residuals),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def predict(image_path, model, categories, device):
    img    = Image.open(image_path).convert('RGB')
    tensor = get_transform()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    pred       = int(np.argmax(probs))
    confidence = probs[pred]

    print(f"\nImage      : {image_path}")
    print(f"Prediction : {categories[pred]}")
    print(f"Confidence : {confidence*100:.1f}%")

    if confidence < 0.55:
        print("  Low confidence — model is uncertain.")

    print("\nClass probabilities:")
    for i, (name, p) in enumerate(zip(categories, probs)):
        bar    = chr(9608) * int(p * 30)
        marker = "  <--" if i == pred else ""
        print(f"  {name:<12} {bar:<30} {p*100:5.1f}%{marker}")

    return categories[pred], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',     required=True,                             help='Path to input image')
    parser.add_argument('--ckpt',      default='./checkpoints/forensic_net.pth',  help='Path to checkpoint')
    args = parser.parse_args()

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, categories = load_model(args.ckpt, device)

    predict(args.image, model, categories, device)
