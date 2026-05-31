"""
Segmentation Training Script — U-Net + ResNet34
Author: Team 4VV22EC006/007/008/019 | VVCE, Mysuru

Config (as per project):
    - Epochs    : 12
    - Batch size: 8
    - Optimizer : Adam (LR: 0.0001)
    - Loss      : Tversky Loss (alpha=0.3, beta=0.7)
    - Input size: 128×128
    - Training  : CPU-based
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet_resnet34 import get_segmentation_model, TverskyLoss, dice_score

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS      = 12
BATCH_SIZE  = 8
LR          = 0.0001
IMG_SIZE    = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = "models/unet_resnet34_weights.pth"

print(f"Training on: {DEVICE}")

# ─── Dataset ──────────────────────────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    """
    Expects folder structure:
        data/
          train/
            images/  *.jpg / *.png
            masks/   *.jpg / *.png  (binary, same filename as image)
    """

    IMG_TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    MASK_TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    AUG_TRANSFORM = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
    ])

    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.augment   = augment
        self.images    = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img  = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")

        # Joint augmentation (same transform for image and mask)
        if self.augment:
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed)
            img  = self.AUG_TRANSFORM(img)
            torch.manual_seed(seed)
            mask = self.AUG_TRANSFORM(mask)

        img  = self.IMG_TRANSFORM(img)
        mask = self.MASK_TRANSFORM(mask)
        mask = (mask > 0.5).float()  # Binarize

        return img, mask


# ─── Training Loop ────────────────────────────────────────────────────────────

def train():
    # Data
    train_dataset = BrainMRIDataset("data/train/images", "data/train/masks", augment=True)
    val_dataset   = BrainMRIDataset("data/val/images",   "data/val/masks",   augment=False)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, Loss, Optimizer
    model     = get_segmentation_model(in_channels=3, pretrained=True).to(DEVICE)
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_dices = [], []
    best_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ── Validate ──
        model.eval()
        epoch_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                epoch_dice += dice_score(logits, masks)

        avg_dice = epoch_dice / len(val_loader)
        val_dices.append(avg_dice)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f}")

        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Best model saved (Dice: {best_dice:.4f})")

    print(f"\nTraining complete. Best Dice Score: {best_dice:.4f}")

    # ── Plot training curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, EPOCHS + 1), train_losses, 'b-o')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Tversky Loss")

    ax2.plot(range(1, EPOCHS + 1), val_dices, 'g-o')
    ax2.set_title("Validation Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=150)
    print("Training curves saved to results/training_curves.png")


if __name__ == "__main__":
    train()
