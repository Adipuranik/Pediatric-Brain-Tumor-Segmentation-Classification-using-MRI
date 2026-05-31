"""
U-Net with ResNet34 Encoder — Pediatric Brain Tumor Segmentation
Author: Team 4VV22EC006/007/008/019 | VVCE, Mysuru
Guide : Dr. Jagadeesh B, Associate Professor, Dept. of ECE

Architecture:
    - Encoder : ResNet34 (pretrained on ImageNet) for rich feature extraction
    - Decoder : U-Net decoder with skip connections
    - Loss    : Tversky Loss (handles class imbalance — tumor vs background)
    - Input   : 128×128 grayscale/RGB MRI slices
    - Output  : Binary segmentation mask (tumor region)
    - Training: 12 epochs, batch size 8, Adam LR 0.0001
    - Result  : Mean Dice Score 0.80
"""

import torch
import torch.nn as nn

# Using segmentation-models-pytorch for U-Net + ResNet34
# Install: pip install segmentation-models-pytorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[WARNING] segmentation-models-pytorch not found. Using manual U-Net fallback.")


# ─── Option A: U-Net + ResNet34 via segmentation-models-pytorch (recommended) ─

def build_unet_resnet34(in_channels=3, out_channels=1, pretrained=True):
    """
    Build U-Net with ResNet34 encoder.
    Args:
        in_channels  : 1 for grayscale, 3 for RGB
        out_channels : 1 for binary segmentation
        pretrained   : Use ImageNet pretrained ResNet34 encoder
    Returns:
        model (nn.Module)
    """
    if not SMP_AVAILABLE:
        raise ImportError("Install segmentation-models-pytorch: pip install segmentation-models-pytorch")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,    # Raw logits — we apply sigmoid in loss/inference
    )
    return model


# ─── Option B: Manual U-Net fallback (if smp not available) ──────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetFallback(nn.Module):
    """Lightweight U-Net fallback — use build_unet_resnet34() when smp is available."""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2)
        ch = in_channels
        for f in features:
            self.downs.append(ConvBlock(ch, f))
            ch = f
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(ConvBlock(f * 2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat([x, skips[i // 2]], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


def get_segmentation_model(in_channels=3, pretrained=True):
    """Returns best available segmentation model."""
    if SMP_AVAILABLE:
        print("Using U-Net + ResNet34 encoder (segmentation-models-pytorch)")
        return build_unet_resnet34(in_channels=in_channels, pretrained=pretrained)
    else:
        print("Using fallback U-Net (no ResNet34 encoder)")
        return UNetFallback(in_channels=in_channels)


# ─── Tversky Loss ─────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky Loss — handles class imbalance between tumor (foreground) and background.
    alpha: weight for False Positives
    beta : weight for False Negatives (increase to penalise missed tumors more)
    When alpha=beta=0.5, reduces to Dice Loss.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


# ─── Dice Score Metric ────────────────────────────────────────────────────────

def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Compute mean Dice coefficient over a batch."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    score = (2 * intersection + smooth) / (
        preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
    )
    return score.mean().item()


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_segmentation_model(in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 128, 128)   # batch=2, RGB, 128×128
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")

    # Test Tversky loss
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss = criterion(out, target)
    print(f"Tversky loss (random): {loss.item():.4f}")
