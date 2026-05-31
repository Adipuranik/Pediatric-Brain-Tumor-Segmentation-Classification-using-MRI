# Pipeline Overview — Pediatric Brain Tumor Segmentation & Classification

## Project Summary

| Field | Details |
|-------|---------|
| Institution | Dept. of ECE, VVCE, Mysuru |
| Guide | Dr. Jagadeesh B, Associate Professor |
| Team | 4VV22EC006 Aditi K · 4VV22EC007 Aditi S Puranik · 4VV22EC008 Ajeya S R · 4VV22EC019 Anveesh S |
| Date | April 2026 |

---

## Dataset

- **BraTS (Brain Tumor Segmentation Challenge)** — standard benchmark for brain tumor MRI
- **Kaggle Pediatric Brain Tumor Dataset** — pediatric-specific MRI images
- **Modality**: 2D MRI slices only (not volumetric 3D)
- Manual annotation completed for ground truth masks

---

## Stage 1 — Preprocessing

| Step | Detail |
|------|--------|
| Resize | 128 × 128 pixels |
| Normalization | ImageNet mean/std (for pretrained backbone compatibility) |
| Augmentation | Horizontal flip, vertical flip, rotation (±15°) |

---

## Stage 2 — Segmentation: U-Net + ResNet34

| Hyperparameter | Value |
|----------------|-------|
| Architecture | U-Net with ResNet34 encoder |
| Encoder weights | ImageNet pretrained |
| Loss function | **Tversky Loss** (α=0.3, β=0.7) |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | **12** |
| Batch size | **8** |
| Training hardware | CPU |
| **Result** | **Mean Dice Score: 0.80** |

**Why Tversky Loss?**  
Tumor pixels are a small fraction of the total MRI. Standard cross-entropy treats foreground and background equally, causing the model to ignore small tumors. Tversky loss penalises False Negatives (missed tumors) more heavily via the β parameter — this improves sensitivity for small lesions.

**Why ResNet34 as encoder?**  
A plain U-Net encoder trained from scratch struggles to extract rich features from limited medical imaging data. Using ResNet34 pretrained on ImageNet provides strong low-level feature detectors (edges, textures) that transfer well to MRI analysis.

---

## Stage 3 — ROI Extraction

1. Predicted binary mask thresholded at optimal value (see Threshold vs Dice graph)
2. Bounding box computed around largest contour
3. Bounding box (+ 15px padding) cropped from original image
4. Background removed → classification model sees only the relevant tumor region

**Why ROI?**  
Without ROI cropping, the classifier processes the entire MRI including skull, ventricles, and healthy brain tissue — all of which are noise for tumor type prediction. Isolating the tumor region improves classification accuracy significantly.

---

## Stage 4 — Classification: ResNet18

| Hyperparameter | Value |
|----------------|-------|
| Architecture | ResNet18 (fine-tuned) |
| Encoder weights | ImageNet pretrained |
| Loss function | Cross-Entropy |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | **5** |
| Batch size | **16** |
| **Result** | **Mean Accuracy: 98%** |
| Alternative | DenseNet121 (also evaluated) |

**Classes**: Glioma · Meningioma · No Tumor · Pituitary

---

## Stage 5 — Output

- Tumor type label + confidence score
- Segmentation mask overlaid on original MRI (coloured highlight)
- Estimated tumor coverage (% of scan area)
- Gradio web interface for non-technical clinical users

---

## Results

| Metric | Value |
|--------|-------|
| Validation Dice Score | **0.80** |
| Classification Accuracy | **98%** |
| Optimal threshold | Determined from Threshold vs Dice graph |

Generated outputs (see `results/` folder):
- `training_curves.png` — Training loss + validation Dice per epoch
- `threshold_vs_dice.png` — Dice score across thresholds 0.1–0.9
- `confusion_matrix.png` — Per-class classification results

---

## Limitations & Future Work

- Currently 2D slice-based; 3D volumetric analysis would capture spatial relationships better
- CPU-only training limits model scale and experiment speed
- Multi-class tumor segmentation (not just binary) would improve clinical usefulness
- Real-time deployment in clinical PACS systems is a future goal
