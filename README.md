#  Deep Learning Framework for Segmentation and Classification of Pediatric Brain Tumors using MRI

An automated deep learning pipeline that combines **U-Net + ResNet34 segmentation** and **ResNet18 classification** to detect and classify pediatric brain tumors from MRI images — with ROI-based extraction, Tversky loss for class imbalance handling, and a unified inference pipeline.

> Final year project — Dept. of ECE, VVCE, Mysuru | April 2026  
> Guide: **Dr. Jagadeesh B**, Associate Professor, Dept. of ECE

---

##  Team

| USN | Name |
|-----|------|
| 4VV22EC006 | Aditi K |
| 4VV22EC007 | Aditi S Puranik |
| 4VV22EC008 | Ajeya S R |
| 4VV22EC019 | Anveesh S |

---

##  Results Summary

| Stage | Model | Metric | Value |
|-------|-------|--------|-------|
| Segmentation | U-Net + ResNet34 encoder | Mean Dice Score | **0.80** |
| Classification | ResNet18 | Mean Accuracy | **98%** |
| Training setup | CPU-based | Optimizer | Adam (LR: 0.0001) |

---

##  Repository Structure

```
pediatric-brain-tumor-segmentation/
├── src/
│   ├── unet_resnet34.py          # U-Net with ResNet34 encoder architecture
│   ├── resnet_classifier.py      # ResNet18 classification model
│   ├── train_segmentation.py     # Segmentation training script
│   ├── train_classifier.py       # Classification training script
│   ├── predict.py                # Run inference on new MRI images
│   ├── gradio_app.py             # Interactive web UI
│   └── utils.py                  # Preprocessing, metrics, visualization
├── models/
│   └── README.md                 # Instructions to download pretrained weights
├── data/
│   └── sample/                   # Sample MRI images for testing
├── docs/
│   └── pipeline_overview.md      # Architecture and design decisions
├── results/                      # Output masks and confusion matrix images
└── README.md
```

---

##  Pipeline Architecture

```
Input MRI (2D slice, 128×128)
        │
        ▼
┌───────────────────┐    Tversky Loss     Epoch: 12, BS: 8
│  U-Net            │ ──────────────────────────────────────
│  Encoder: ResNet34│    Binary mask output (tumor region)
│  Dice: 0.80       │
└───────────────────┘
        │
        ▼
   ROI Extraction
   (mask applied → isolate tumor region)
        │
        ▼
┌───────────────────┐    CrossEntropy     Epoch: 5, BS: 16
│  ResNet18         │ ──────────────────────────────────────
│  Classification   │    Tumor type prediction
│  Accuracy: 98%    │
└───────────────────┘
        │
        ▼
   Final Output:
   Tumor type label + Segmentation overlay + Visualization
```

---

##  Methodology

### 1. MRI Image Input
- Dataset: **BraTS** + **Kaggle Pediatric Brain Tumor** dataset
- Only **2D slices** used (not volumetric 3D data)

### 2. Preprocessing
- Resize to **128×128**
- Pixel value normalization
- Data augmentation for improved generalization
- Manual annotation completed for ground truth masks

### 3. Segmentation — U-Net + ResNet34
- U-Net architecture with **ResNet34 as the encoder backbone** for richer feature extraction
- **Tversky Loss** used to handle class imbalance (tumor pixels vs background)
- Trained for **12 epochs**, batch size **8**, Adam optimizer (LR: 0.0001)
- Outputs a **probability map** → thresholded to binary mask
- **Mean Dice Score: 0.80**

### 4. ROI Extraction
- Predicted mask used to isolate the tumor region
- Background removed — improves downstream classification accuracy

### 5. Classification — ResNet18
- Extracted ROI fed into **ResNet18** for tumor type prediction
- Trained for **5 epochs**, batch size **16**
- **Mean Accuracy: 98%**

### 6. Output
- Detected tumor type
- Segmented tumor region with visual overlay
- Threshold vs Dice graph and confusion matrix generated

---

##  Hardware & Software Requirements

| Category | Tool / Framework |
|----------|-----------------|
| Deep Learning | PyTorch |
| Image Processing | OpenCV, NumPy |
| Visualization | Matplotlib |
| Segmentation Model | U-Net + ResNet34 |
| Classification Model | ResNet18 |
| Optimizer | Adam (LR: 0.0001) |
| Training Hardware | CPU-based (GPU quota limited) |
| Dataset Storage | Google Drive |

---

##  requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
gradio>=4.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

##  Dataset

This project uses two datasets:

1. **BraTS (Brain Tumor Segmentation Challenge)** — [CBICA BraTS](https://www.med.upenn.edu/cbica/brats/)
2. **Kaggle Pediatric Brain Tumor Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

> Datasets are not included in this repo due to size. Download and place under `data/` as described in `docs/pipeline_overview.md`.

---

##  Results

> Add the following to `results/` folder:
> - Validation Dice Score curve
> - Training Loss graph
> - Threshold vs Dice graph
> - Confusion matrix
> - Final output overlay samples

---

##  Disclaimer

This is an academic research prototype developed for the Project Exhibition at VVCE, Mysuru. It is **not a certified medical diagnostic tool** and should not be used for clinical decision-making.

---

##  License

MIT License — free to use, modify, and distribute with attribution.

---

##  Authors

**Aditi S Puranik** (4VV22EC007) · [LinkedIn](https://linkedin.com/in/aditi-s-puranik-598629391) · [GitHub](https://github.com/Adipuranik)  
Aditi K (4VV22EC006) · Ajeya S R (4VV22EC008) · Anveesh S (4VV22EC019)  
Dept. of ECE, Vidyavardhaka College of Engineering, Mysuru
