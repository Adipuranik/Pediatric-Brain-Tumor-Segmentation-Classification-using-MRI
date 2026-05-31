"""
Gradio Web Interface — Pediatric Brain Tumor Segmentation & Classification
Author: Team 4VV22EC006/007/008/019 | VVCE, Mysuru
Guide : Dr. Jagadeesh B, Associate Professor, Dept. of ECE

Pipeline:
    1. Upload MRI → U-Net + ResNet34 segments tumor region
    2. ROI extracted from predicted mask
    3. ResNet18 classifies tumor type from ROI
    4. Result: overlay + tumor type + confidence

Usage:
    python src/gradio_app.py
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import os

from unet_resnet34 import get_segmentation_model, dice_score
from resnet_classifier import get_classifier, CLASSES
from utils import preprocess_mri_seg, preprocess_mri_cls, overlay_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# ─── Load Models ──────────────────────────────────────────────────────────────
seg_model = get_segmentation_model(in_channels=3, pretrained=False).to(DEVICE)
cls_model = get_classifier(backbone="resnet18", pretrained=False).to(DEVICE)

SEG_WEIGHTS = "models/unet_resnet34_weights.pth"
CLS_WEIGHTS = "models/resnet18_weights.pth"

if os.path.exists(SEG_WEIGHTS):
    seg_model.load_state_dict(torch.load(SEG_WEIGHTS, map_location=DEVICE))
    print("Segmentation weights loaded.")
else:
    print(f"[WARNING] {SEG_WEIGHTS} not found. Using untrained model.")

if os.path.exists(CLS_WEIGHTS):
    cls_model.load_state_dict(torch.load(CLS_WEIGHTS, map_location=DEVICE))
    print("Classification weights loaded.")
else:
    print(f"[WARNING] {CLS_WEIGHTS} not found. Using untrained model.")

seg_model.eval()
cls_model.eval()


# ─── Inference ────────────────────────────────────────────────────────────────

def analyze_mri(image: Image.Image):
    if image is None:
        return None, "Please upload an MRI image."

    original_np = np.array(image.convert("RGB"))

    # 1. Segmentation (128×128 input)
    seg_input = preprocess_mri_seg(image).to(DEVICE)
    with torch.no_grad():
        seg_logits = seg_model(seg_input)
        seg_mask = (torch.sigmoid(seg_logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # 2. Overlay
    overlay_img = overlay_mask(original_np, seg_mask)
    tumor_pixels = int(seg_mask.sum())
    size_pct = round((tumor_pixels / seg_mask.size) * 100, 2)

    # 3. ROI-based classification
    cls_input = preprocess_mri_cls(image, seg_mask).to(DEVICE)
    with torch.no_grad():
        logits = cls_model(cls_input)
        probs = torch.softmax(logits, dim=1).squeeze()
        confidence, predicted = torch.max(probs, dim=0)

    tumor_type  = CLASSES[predicted.item()]
    conf_pct    = round(confidence.item() * 100, 1)

    # 4. Build result
    if tumor_type == "No Tumor":
        result = (
            f"✅ **No Tumor Detected**\n"
            f"Confidence: {conf_pct}%\n"
            f"Segmentation coverage: {size_pct}% of scan"
        )
    else:
        prob_lines = "\n".join(
            [f"  {CLASSES[i]}: {round(probs[i].item()*100, 1)}%" for i in range(len(CLASSES))]
        )
        result = (
            f"⚠️ **Tumor Detected: {tumor_type}**\n"
            f"Confidence: {conf_pct}%\n"
            f"Estimated tumor area: {size_pct}% of scan\n\n"
            f"**All class probabilities:**\n{prob_lines}"
        )

    return Image.fromarray(overlay_img), result


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Pediatric Brain Tumor Analyzer") as demo:
    gr.Markdown(
        "# 🧠 Pediatric Brain Tumor Segmentation & Classification\n"
        "**U-Net + ResNet34** segmentation (Dice: 0.80) · **ResNet18** classification (Acc: 98%)\n\n"
        "_Dept. of ECE, VVCE, Mysuru | Guide: Dr. Jagadeesh B_\n\n"
        "> ⚠️ Research prototype only — not for clinical use."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Brain MRI Image")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Segmentation Overlay")
            output_text  = gr.Markdown(label="Result")

    analyze_btn.click(
        fn=analyze_mri,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

    gr.Examples(
        examples=[["data/sample/sample_mri.jpg"]],
        inputs=input_image,
        label="Try a sample image"
    )

if __name__ == "__main__":
    demo.launch(share=False)
