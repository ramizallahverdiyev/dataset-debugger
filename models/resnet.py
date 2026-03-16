"""
resnet.py — ResNet-18 adapted for Tiny ImageNet (64x64 input, 200 classes).

Key modification: replace the original 7x7 conv (designed for 224x224)
with a 3x3 conv and remove maxpool, following common practice for small images.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 200, pretrained: bool = False) -> nn.Module:
    """
    Build ResNet-18 adapted for 64x64 Tiny ImageNet inputs.

    Args:
        num_classes: Number of output classes (200 for Tiny ImageNet).
        pretrained:  Load ImageNet pretrained weights (then fine-tune).

    Returns:
        Modified ResNet-18 model.
    """
    model = resnet18(pretrained=pretrained)

    # Adapt first conv: 7x7 stride 2 → 3x3 stride 1 (better for 64x64)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove maxpool (would over-downsample 64x64 input)
    model.maxpool = nn.Identity()

    # Replace final FC for 200 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_embeddings_model(model: nn.Module) -> nn.Module:
    """
    Strip the final FC layer to get a feature extractor.
    Returns embeddings of shape (batch, 512).
    """
    embedding_model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return embedding_model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_resnet18()
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print(f"Output shape: {out.shape}")           # (4, 200)
    print(f"Parameters: {count_parameters(model):,}")
