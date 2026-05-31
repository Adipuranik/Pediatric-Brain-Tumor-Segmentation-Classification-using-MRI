"""
Utility functions — preprocessing, ROI extraction, visualization
Author: Team 4VV22EC006/007/008/019 | VVCE, Mysuru
"""

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 128  # Project uses 128×128

# ─── Segmentation Preprocessing ───────────────────────────────────────────────

SEG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_mri_seg(image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image for segmentation model. Returns (1, 3, 128, 128)."""
    return SEG_TRANSFORM(image.convert("RGB")).unsqueeze(0)


# ─── Classification Preprocessing (with optional ROI crop) ───────────────────

CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_mri_cls(image: Image.Image, mask: np.ndarray = None) -> torch.Tensor:
    """
    Preprocess MRI for classification.
    If mask provided, crops to ROI bounding box (removes background) before resize.
    Returns (1, 3, 128, 128).
    """
    img_np = np.array(image.convert("RGB"))

    if mask is not None and mask.sum() > 0:
        h, w = img_np.shape[:2]
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = 15
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad); y2 = min(h, y + bh + pad)
            img_np = img_np[y1:y2, x1:x2]

    pil_img = Image.fromarray(img_np)
    return CLS_TRANSFORM(pil_img).unsqueeze(0)


# ─── Mask Overlay ─────────────────────────────────────────────────────────────

def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 alpha=0.4, color=(255, 50, 50)) -> np.ndarray:
    """Overlay binary segmentation mask on RGB image. Returns overlay numpy array."""
    h, w = image_rgb.shape[:2]
    mask_resized = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    overlay = image_rgb.copy()
    overlay[mask_resized == 1] = (
        np.array(overlay[mask_resized == 1], dtype=float) * (1 - alpha)
        + np.array(color) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


# ─── Threshold vs Dice Curve ──────────────────────────────────────────────────

def plot_threshold_vs_dice(logits_list, masks_list, save_path="results/threshold_vs_dice.png"):
    """
    Plots Dice score across different thresholds (0.1 to 0.9).
    Useful for picking the optimal binarization threshold.
    """
    import matplotlib.pyplot as plt

    thresholds = np.arange(0.1, 1.0, 0.05)
    dice_scores = []

    for thresh in thresholds:
        total_dice = 0.0
        for logits, masks in zip(logits_list, masks_list):
            probs = torch.sigmoid(logits)
            preds = (probs > thresh).float()
            smooth = 1e-6
            intersection = (preds * masks).sum()
            score = (2 * intersection + smooth) / (preds.sum() + masks.sum() + smooth)
            total_dice += score.item()
        dice_scores.append(total_dice / len(logits_list))

    best_thresh = thresholds[np.argmax(dice_scores)]
    best_dice   = max(dice_scores)

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, dice_scores, 'b-o', markersize=4)
    plt.axvline(best_thresh, color='r', linestyle='--',
                label=f'Best threshold: {best_thresh:.2f} (Dice: {best_dice:.3f})')
    plt.xlabel("Threshold")
    plt.ylabel("Dice Score")
    plt.title("Threshold vs Dice Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Threshold vs Dice graph saved to {save_path}")

    return best_thresh, best_dice
