"""
MIST Lab — DeepFake Forensics & Fingerprint Elimination
Combined Gradio UI
================================================
Tab 1 : DNA-Det Attribution     (6-class: Real/ProGAN/MMDGAN/SNGAN/StyleGAN/CramerGAN)
Tab 2 : ForensicNet Attribution  (4-class: Real/DCGAN/StyleGAN/ProGAN)
Tab 3 : Fingerprint Elimination  (Encoder-Decoder Φ + GBMS smoother)
Tab 4 : Full Pipeline            (Attribution → Elimination → Attribution again → ASR)

Run:
    pip install gradio torch torchvision pillow scipy numpy
    python app.py
"""

import os, io, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import gradio as gr

# ══════════════════════════════════════════════════════
# EDIT ONLY THESE PATHS
# ══════════════════════════════════════════════════════
DNA_DET_CKPT     = "./checkpoints/dnadet_best.pth"
FORENSICNET_CKPT = "./checkpoints/forensicnet_best.pth"
ELIMINATION_CKPT = "./checkpoints/elimination_best.pth"
# ══════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DNA_DET_CLASSES   = ["Real", "ProGAN", "MMDGAN", "SNGAN", "StyleGAN", "CramerGAN"]
FORENSIC_CLASSES  = ["Real", "DCGAN", "StyleGAN", "ProGAN"]
IMAGE_SIZE        = 128

# ══════════════════════════════════════════════════════
# SHARED — FORENSIC FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════
def extract_forensic_residuals(img_pil, size=128):
    img  = img_pil.resize((size, size), Image.LANCZOS)
    gray = img.convert('L')
    arr  = np.array(gray).astype(np.float32)
    b1 = np.array(gray.filter(ImageFilter.GaussianBlur(1))).astype(np.float32)
    b2 = np.array(gray.filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
    r  = np.clip((arr - b1) * 2.0 + (arr - b2) * 1.0 + 128, 0, 255).astype(np.uint8)
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


dnadet_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

forensic_tf = transforms.Compose([
    transforms.Lambda(extract_forensic_residuals),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════
# MODEL 1 — DNA-Det (6-class)
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
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num),
        )
    def forward(self, x):
        f = self.stem(x)
        f = self.stage4(self.stage3(self.stage2(self.stage1(f))))
        return self.classifier(self.gap(f)), self.gap(f).flatten(1)


# ══════════════════════════════════════════════════════
# MODEL 2 — ForensicNet (4-class)
# ══════════════════════════════════════════════════════
class ForensicNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        self.model = base
    def forward(self, x):
        return self.model(x)


# ══════════════════════════════════════════════════════
# MODEL 3 — Elimination Model Φ
# ══════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(ch, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
        )
    def forward(self, x): return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, bc=64, n_res=5):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, bc,   7,1,3,bias=False), nn.InstanceNorm2d(bc,  affine=True), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(bc,  bc*2,   3,2,1,bias=False), nn.InstanceNorm2d(bc*2,affine=True), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(bc*2,bc*4,   3,2,1,bias=False), nn.InstanceNorm2d(bc*4,affine=True), nn.ReLU(True))
        self.res   = nn.Sequential(*[ResidualBlock(bc*4) for _ in range(n_res)])
    def forward(self, x): return self.res(self.conv3(self.conv2(self.conv1(x))))


class Decoder(nn.Module):
    def __init__(self, bc=64, out_ch=3):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False), nn.Conv2d(bc*4,bc*2,3,1,1,bias=False), nn.InstanceNorm2d(bc*2,affine=True), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False), nn.Conv2d(bc*2,bc,  3,1,1,bias=False), nn.InstanceNorm2d(bc,  affine=True), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(bc, out_ch, 7,1,3), nn.Tanh())
        nn.init.xavier_uniform_(self.out[0].weight, gain=0.1)
        nn.init.zeros_(self.out[0].bias)
    def forward(self, x): return self.out(self.up2(self.up1(x)))


class EliminationModel(nn.Module):
    def __init__(self, bc=64, n_res=5):
        super().__init__()
        self.encoder = Encoder(3, bc, n_res)
        self.decoder = Decoder(bc, 3)
    def forward(self, x):
        return (x + self.decoder(self.encoder(x))).clamp(-1, 1)


# GBMS Smoother
class _GaussianBlur(nn.Module):
    def __init__(self, ks=5, sigma=1.0):
        super().__init__()
        c   = torch.arange(ks, dtype=torch.float32) - ks // 2
        k1d = torch.exp(-c**2 / (2 * sigma**2))
        k1d = k1d / k1d.sum()
        k2d = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", k2d)
        self.pad = ks // 2
    def forward(self, x):
        C = x.shape[1]
        return F.conv2d(F.pad(x, [self.pad]*4, "reflect"), self.kernel.expand(C,1,-1,-1), groups=C)


class _MeanShift(nn.Module):
    def __init__(self, ks=7, ss=3.0, sr=0.1, iters=1):
        super().__init__()
        self.ks=ks; self.sr=sr; self.iters=iters; self.pad=ks//2
        c = torch.arange(ks, dtype=torch.float32) - ks//2
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("sw", torch.exp(-(gx**2 + gy**2) / (2*ss**2)))
    @torch.no_grad()
    def _pass(self, x):
        B,C,H,W = x.shape
        k = self.ks
        patches = F.unfold(F.pad(x,[self.pad]*4,"reflect"),k).view(B,C,k*k,H*W)
        centre  = x.view(B,C,1,H*W)
        rw = torch.exp(-((patches-centre)**2).sum(1,keepdim=True)/(2*self.sr**2))
        sw = self.sw.view(1,1,k*k,1)
        wt = (sw*rw)/((sw*rw).sum(2,keepdim=True)+1e-8)
        return (patches*wt).sum(2).view(B,C,H,W)
    def forward(self, x):
        for _ in range(self.iters): x = self._pass(x)
        return x


class GBMSSmoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.gb = _GaussianBlur(5,1.0)
        self.ms = _MeanShift(7,3.0,0.1,1)
    @torch.no_grad()
    def forward(self, x): return self.ms(self.gb(x)).clamp(0,1)


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
def load_models():
    models = {}

    # DNA-Det
    if os.path.exists(DNA_DET_CKPT):
        try:
            m = DNADet(class_num=6).to(device)
            m.load_state_dict(torch.load(DNA_DET_CKPT, map_location=device, weights_only=False))
            m.eval()
            models["dnadet"] = m
            print("✓ DNA-Det loaded")
        except Exception as e:
            print(f"✗ DNA-Det load failed: {e}")
            models["dnadet"] = None
    else:
        models["dnadet"] = None
        print(f"✗ DNA-Det not found: {DNA_DET_CKPT}")

    # ForensicNet
    if os.path.exists(FORENSICNET_CKPT):
        try:
            ckpt = torch.load(FORENSICNET_CKPT, map_location=device, weights_only=False)
            cats = ckpt.get("categories", FORENSIC_CLASSES)
            m = ForensicNet(num_classes=len(cats)).to(device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            models["forensicnet"] = m
            models["forensic_classes"] = cats
            print("✓ ForensicNet loaded")
        except Exception as e:
            print(f"✗ ForensicNet load failed: {e}")
            models["forensicnet"] = None
            models["forensic_classes"] = FORENSIC_CLASSES
    else:
        models["forensicnet"] = None
        models["forensic_classes"] = FORENSIC_CLASSES
        print(f"✗ ForensicNet not found: {FORENSICNET_CKPT}")

    # Elimination
    if os.path.exists(ELIMINATION_CKPT):
        try:
            elim = EliminationModel().to(device)
            ckpt = torch.load(ELIMINATION_CKPT, map_location=device, weights_only=False)
            elim.load_state_dict(ckpt["model"])
            elim.eval()
            smoother = GBMSSmoother().to(device)
            models["elimination"] = elim
            models["smoother"]    = smoother
            print("✓ Elimination model loaded")
        except Exception as e:
            print(f"✗ Elimination load failed: {e}")
            models["elimination"] = None
            models["smoother"]    = None
    else:
        models["elimination"] = None
        models["smoother"]    = None
        print(f"✗ Elimination model not found: {ELIMINATION_CKPT}")

    return models


MODELS = load_models()

elim_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ══════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════
def run_dnadet(img_pil):
    if MODELS["dnadet"] is None:
        return {c: 0.0 for c in DNA_DET_CLASSES}, "  Model not loaded"
    forensic = extract_forensic_residuals(img_pil, IMAGE_SIZE)
    tensor   = dnadet_tf(forensic).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = MODELS["dnadet"](tensor)
        probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx  = probs.argmax()
    pred_name = DNA_DET_CLASSES[pred_idx]
    conf      = probs[pred_idx] * 100
    label_probs = {n: float(p) for n, p in zip(DNA_DET_CLASSES, probs)}
    verdict = f"  REAL — {conf:.1f}% confidence" if pred_name == "Real" \
              else f"  FAKE ({pred_name}) — {conf:.1f}% confidence"
    return label_probs, verdict


def run_forensicnet(img_pil):
    cats = MODELS["forensic_classes"]
    if MODELS["forensicnet"] is None:
        return {c: 0.0 for c in cats}, "  Model not loaded"
    tensor = forensic_tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(MODELS["forensicnet"](tensor), dim=1)[0].cpu().numpy()
    pred_idx  = probs.argmax()
    pred_name = cats[pred_idx]
    conf      = probs[pred_idx] * 100
    label_probs = {n: float(p) for n, p in zip(cats, probs)}
    verdict = f"  REAL — {conf:.1f}% confidence" if pred_name == "Real" \
              else f"  FAKE ({pred_name}) — {conf:.1f}% confidence"
    return label_probs, verdict


def run_elimination(img_pil):
    if MODELS["elimination"] is None:
        return img_pil, "  Elimination model not loaded"
    img_t = elim_tf(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred        = MODELS["elimination"](img_t * 2.0 - 1.0)
        pred_01     = ((pred + 1.0) / 2.0).clamp(0, 1)
        pred_smooth = MODELS["smoother"](pred_01.clamp(0.001, 0.999))
    out_np = (pred_smooth[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(out_np), "✓ Fingerprint elimination complete"


def pil_to_np(img): return np.array(img)


# ══════════════════════════════════════════════════════
# GRADIO TABS
# ══════════════════════════════════════════════════════

CSS = """
.gradio-container { font-family: 'Courier New', monospace; background: #080e1a !important; }
body { background: #080e1a; }
h1 { color: #00e5ff; text-align: center; letter-spacing: 3px; }
.tab-nav button { background: #0d1b2a !important; color: #607d8b !important; border-bottom: 2px solid #0d1b2a !important; }
.tab-nav button.selected { color: #00e5ff !important; border-bottom: 2px solid #00e5ff !important; }
.panel { background: #0d1b2a !important; border: 1px solid #00e5ff22 !important; border-radius: 12px !important; }
#verdict textarea, #verdict2 textarea, #pipeline-verdict textarea {
    font-size: 1.2rem !important; font-weight: bold; text-align: center;
    background: #060d18 !important; color: #00e5ff !important;
    border: 1px solid #00e5ff44 !important; border-radius: 8px !important;
}
.pill { display:inline-block; padding:2px 10px; border-radius:20px;
        background:#00e5ff22; color:#00e5ff; font-size:0.75rem; margin:2px; }
"""

with gr.Blocks(
    title="MIST Lab — DeepFake Forensics",
) as demo:
    gr.HTML("""
        <h1>⬡ MIST LAB — DEEPFAKE FORENSICS</h1>
        <p style="text-align:center;color:#607d8b;letter-spacing:2px;font-size:0.85rem;">
            IIT Bhilai · Machine Intelligence & Security of Things Lab
        </p>
        <p style="text-align:center;margin:8px 0 20px;">
            <span class="pill">DNA-Det (6-class)</span>
            <span class="pill">ForensicNet (4-class)</span>
            <span class="pill">Fingerprint Elimination</span>
            <span class="pill">Full Pipeline</span>
        </p>
    """)

    # ── TAB 1: DNA-Det ─────────────────────────────────
    with gr.Tab("🔬 DNA-Det Attribution"):
        gr.HTML("<p style='color:#607d8b;text-align:center;'>6-class GAN attribution — Real / ProGAN / MMDGAN / SNGAN / StyleGAN / CramerGAN</p>")
        with gr.Row():
            with gr.Column(scale=1):
                img1 = gr.Image(label="Upload Image", type="pil", height=280)
                btn1 = gr.Button("🔍 Analyze", variant="primary")
            with gr.Column(scale=1):
                verdict1  = gr.Textbox(label="Verdict", interactive=False, elem_id="verdict", lines=2)
                probs1    = gr.Label(label="Class Probabilities", num_top_classes=6)
                forensic1 = gr.Image(label="Forensic Feature Map (R=residual G=FFT B=noise)", height=160)

        def predict_dnadet(img):
            if img is None: return {}, "", None
            probs, verdict = run_dnadet(img)
            forensic_img   = extract_forensic_residuals(img, IMAGE_SIZE)
            return probs, verdict, forensic_img

        btn1.click(predict_dnadet, inputs=img1, outputs=[probs1, verdict1, forensic1])
        img1.change(predict_dnadet, inputs=img1, outputs=[probs1, verdict1, forensic1])

    # ── TAB 2: ForensicNet ─────────────────────────────
    with gr.Tab(" ForensicNet Attribution"):
        gr.HTML("<p style='color:#607d8b;text-align:center;'>Custom EfficientNet-B0 model — Real / DCGAN / StyleGAN / ProGAN</p>")
        with gr.Row():
            with gr.Column(scale=1):
                img2 = gr.Image(label="Upload Image", type="pil", height=280)
                btn2 = gr.Button("🔍 Analyze", variant="primary")
            with gr.Column(scale=1):
                verdict2  = gr.Textbox(label="Verdict", interactive=False, elem_id="verdict2", lines=2)
                probs2    = gr.Label(label="Class Probabilities", num_top_classes=4)
                forensic2 = gr.Image(label="Forensic Feature Map", height=160)

        def predict_forensicnet(img):
            if img is None: return {}, "", None
            probs, verdict = run_forensicnet(img)
            forensic_img   = extract_forensic_residuals(img, IMAGE_SIZE)
            return probs, verdict, forensic_img

        btn2.click(predict_forensicnet, inputs=img2, outputs=[probs2, verdict2, forensic2])
        img2.change(predict_forensicnet, inputs=img2, outputs=[probs2, verdict2, forensic2])

    # ── TAB 3: Fingerprint Elimination ─────────────────
    with gr.Tab(" Fingerprint Elimination"):
        gr.HTML("<p style='color:#607d8b;text-align:center;'>Encoder-Decoder Φ + GBMS Smoother — removes GAN fingerprint from deepfake</p>")
        with gr.Row():
            with gr.Column(scale=1):
                img3    = gr.Image(label="Input Deepfake Image", type="pil", height=256)
                btn3    = gr.Button(" Eliminate Fingerprint", variant="primary")
            with gr.Column(scale=1):
                out3    = gr.Image(label="Output — Untraceable Image x'", height=256)
                status3 = gr.Textbox(label="Status", interactive=False, lines=1)

        def eliminate(img):
            if img is None: return None, "Please upload an image"
            out, status = run_elimination(img)
            return out, status

        btn3.click(eliminate, inputs=img3, outputs=[out3, status3])

    # ── TAB 4: Full Pipeline ───────────────────────────
    with gr.Tab(" Full Pipeline (ASR Demo)"):
        gr.HTML("""
            <p style='color:#607d8b;text-align:center;'>
                Upload a deepfake → Attribution before → Eliminate fingerprint → Attribution after → See ASR
            </p>
        """)
        with gr.Row():
            img4 = gr.Image(label="Input Deepfake", type="pil", height=220)
            with gr.Column():
                btn4    = gr.Button("▶ Run Full Pipeline", variant="primary", size="lg")
                status4 = gr.Textbox(label="Pipeline status", interactive=False, lines=1)

        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='color:#00e5ff;text-align:center;font-size:0.85rem;'>BEFORE elimination</p>")
                before_probs = gr.Label(label="Attribution (DNA-Det) — BEFORE", num_top_classes=6)
                before_verdict = gr.Textbox(label="Verdict BEFORE", interactive=False, lines=1)
            with gr.Column():
                out4 = gr.Image(label="Untraceable output x'", height=180)
            with gr.Column():
                gr.HTML("<p style='color:#00e5ff;text-align:center;font-size:0.85rem;'>AFTER elimination</p>")
                after_probs = gr.Label(label="Attribution (DNA-Det) — AFTER", num_top_classes=6)
                after_verdict = gr.Textbox(label="Verdict AFTER", interactive=False, lines=1)

        asr_box = gr.Textbox(
            label="Attack Success Rate Result",
            interactive=False,
            elem_id="pipeline-verdict",
            lines=2
        )

        def full_pipeline(img):
            if img is None:
                return {}, "", None, {}, "", "Please upload an image", "–"

            # Step 1: Attribution BEFORE
            before_p, before_v = run_dnadet(img)

            # Step 2: Eliminate
            out_img, elim_status = run_elimination(img)

            # Step 3: Attribution AFTER
            after_p, after_v = run_dnadet(out_img)

            # Step 4: Compute ASR
            before_pred = max(before_p, key=before_p.get)
            after_pred  = max(after_p,  key=after_p.get)

            if before_pred != "Real":
                if after_pred == "Real" or after_pred != before_pred:
                    asr_result = f"  ATTACK SUCCESSFUL — Attribution changed: {before_pred} → {after_pred}\n" \
                                 f"Fingerprint successfully eliminated!"
                else:
                    asr_result = f"  ATTACK FAILED — Attribution unchanged: {before_pred} → {after_pred}\n" \
                                 f"Model still traces the image to the same GAN source."
            else:
                asr_result = f"  Input appears REAL — run with a deepfake image for meaningful ASR.\n" \
                              f"Attribution before: {before_pred} | after: {after_pred}"

            return before_p, before_v, out_img, after_p, after_v, elim_status, asr_result

        btn4.click(
            full_pipeline,
            inputs=img4,
            outputs=[before_probs, before_verdict, out4, after_probs, after_verdict, status4, asr_box]
        )

    gr.HTML("""
        <p style='text-align:center;color:#263238;font-size:0.75rem;margin-top:24px;'>
            MIST Lab · IIT Bhilai · Internship Project · Jan–Jun 2025
        </p>
    """)


if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True
    )
