"""
ResNet18 Classifier — Pediatric Brain Tumor Type Classification
Author: Team 4VV22EC006/007/008/019 | VVCE, Mysuru

Config (as per project):
    - Epochs    : 5
    - Batch size: 16
    - Optimizer : Adam (LR: 0.0001)
    - Input     : ROI-cropped MRI (from segmentation mask), 128×128
    - Accuracy  : 98%

Note: DenseNet121 is available as an alternative backbone (see get_classifier()).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
EPOCHS      = 5
BATCH_SIZE  = 16
LR          = 0.0001
IMG_SIZE    = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES     = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NUM_CLASSES = len(CLASSES)

# ─── Model Definitions ────────────────────────────────────────────────────────

def build_resnet18(pretrained=True):
    """ResNet18 fine-tuned for 4-class tumor classification."""
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    return model


def build_densenet121(pretrained=True):
    """DenseNet121 — alternative backbone used in project."""
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier.in_features, NUM_CLASSES)
    )
    return model


def get_classifier(backbone="resnet18", pretrained=True):
    """
    Returns selected classification model.
    Args:
        backbone: "resnet18" (default) or "densenet121"
    """
    if backbone == "densenet121":
        print("Using DenseNet121 classifier")
        return build_densenet121(pretrained=pretrained)
    else:
        print("Using ResNet18 classifier")
        return build_resnet18(pretrained=pretrained)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TumorROIDataset(Dataset):
    """
    Expects folder structure:
        data/classification/
            Glioma/       *.jpg
            Meningioma/   *.jpg
            No Tumor/     *.jpg
            Pituitary/    *.jpg
    ROI crops (from segmentation mask) should be pre-saved here.
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root_dir):
        self.samples = []
        for label, cls in enumerate(CLASSES):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.TRANSFORM(img), label


# ─── Training ─────────────────────────────────────────────────────────────────

def train_classifier(backbone="resnet18"):
    train_dataset = TumorROIDataset("data/classification/train")
    val_dataset   = TumorROIDataset("data/classification/val")
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model     = get_classifier(backbone=backbone, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        correct, total = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total * 100

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total * 100
        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{backbone}_weights.pth")
            print(f"  → Best model saved (Acc: {best_acc:.2f}%)")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {backbone} (Val Acc: {best_acc:.1f}%)")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    train_classifier(backbone="resnet18")
