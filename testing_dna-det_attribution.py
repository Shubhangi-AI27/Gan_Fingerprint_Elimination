"""
DNA-Det Inference UI
Run: python inference_ui.py
Opens a browser at http://localhost:7862
Upload any image → get GAN source prediction instantly.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter
from torchvision import transforms
import gradio as gr

# ══════════════════════════════════════════════════════
# EDIT ONLY THIS PATH
# ══════════════════════════════════════════════════════
# NOTE: Change this path to your saved model location before running
SAVE_DIR = "/data1/intern/dnadet_new/saved_model"
# ══════════════════════════════════════════════════════

CONFIG = {
    "image_size" : 128,
    "class_names": ["Real", "ProGAN", "MMDGAN", "SNGAN", "StyleGAN", "CramerGAN"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ══════════════════════════════════════════════════════
# FORENSIC PREPROCESSING  
# ══════════════════════════════════════════════════════
def extract_forensic_residuals(img_pil, size=128):
    img  = img_pil.resize((size, size), Image.LANCZOS)
    gray = img.convert('L')
    arr  = np.array(gray).astype(np.float32)

    b1 = np.array(gray.filter(ImageFilter.GaussianBlur(1))).astype(np.float32)
    b2 = np.array(gray.filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
    r  = np.clip((arr-b1)*2.0 + (arr-b2)*1.0 + 128, 0, 255).astype(np.uint8)

    win     = np.outer(np.hanning(size), np.hanning(size))
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(arr * win)))
    log_mag = np.log1p(fft_mag)
    g = ((log_mag - log_mag.min()) /
         (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)

    mean    = uniform_filter(arr,    size=5)
    mean_sq = uniform_filter(arr**2, size=5)
    std     = np.sqrt(np.clip(mean_sq - mean**2, 0, None))
    b = ((std / (std.max() + 1e-8)) * 255).astype(np.uint8)

    return Image.fromarray(np.stack([r, g, b], axis=2), 'RGB')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


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


# ── Load model ─────────────────────────────────────────
model_path = os.path.join(SAVE_DIR, "best_model.pth")
print(f"Loading model from: {model_path}")
model = DNADet(class_num=6).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")


# ══════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ══════════════════════════════════════════════════════
def predict(image):
    if image is None:
        return {}, "Please upload an image."

    try:
        img = Image.fromarray(image).convert("RGB")
    except Exception as e:
        return {}, f"Error loading image: {e}"

    forensic = extract_forensic_residuals(img, CONFIG["image_size"])
    tensor   = transform(forensic).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(tensor)
        probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = probs.argmax()
    pred_name  = CONFIG["class_names"][pred_idx]
    confidence = probs[pred_idx] * 100

    label_probs = {name: float(p) for name, p in zip(CONFIG["class_names"], probs)}

    if pred_name == "Real":
        verdict = f"  REAL IMAGE — {confidence:.1f}% confidence"
    else:
        verdict = f"  FAKE ({pred_name}) — {confidence:.1f}% confidence"

    return label_probs, verdict


# ══════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════
with gr.Blocks(
    title="DNA-Det GAN Detector",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    ),
    css="""
        body { background: #0a0f1e; }
        .gradio-container {
            background: #0a0f1e !important;
            font-family: 'Courier New', monospace;
        }
        h1 { color: #00e5ff; text-align: center; letter-spacing: 4px; font-size: 2rem; }
        .subtitle { color: #607d8b; text-align: center; margin-bottom: 20px; }
        #verdict-box textarea {
            font-size: 1.3rem !important;
            font-weight: bold;
            text-align: center;
            background: #0d1b2a !important;
            color: #00e5ff !important;
            border: 1px solid #00e5ff44 !important;
            border-radius: 8px;
        }
        .upload-box { border: 2px dashed #00e5ff44 !important; border-radius: 12px; }
    """
) as demo:

    gr.HTML("""
        <h1>⬡ DNA-DET</h1>
        <p class="subtitle">GAN Forensics · Deepfake Detection · 6-Class Classifier</p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Image",
                type="numpy",
                elem_classes=["upload-box"],
                height=300,
            )
            run_btn = gr.Button("🔍  Analyze Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict_box = gr.Textbox(
                label="Verdict",
                interactive=False,
                elem_id="verdict-box",
                lines=2,
            )
            prob_chart = gr.Label(
                label="Class Probabilities",
                num_top_classes=6,
            )

    run_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[prob_chart, verdict_box],
    )

    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[prob_chart, verdict_box],
    )

    gr.HTML("""
        <p style="text-align:center; color:#37474f; font-size:0.8rem; margin-top:20px;">
            Classes: Real · ProGAN · MMDGAN · SNGAN · StyleGAN · CramerGAN
        </p>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True,
    )
