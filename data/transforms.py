"""
transforms.py — Normalization and augmentation pipelines for Tiny ImageNet.
"""

from torchvision import transforms

# Tiny ImageNet statistics (precomputed)
MEAN = [0.4802, 0.4481, 0.3975]
STD  = [0.2770, 0.2691, 0.2821]


def get_train_transforms():
    """Augmented transforms for training."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_val_transforms():
    """Clean transforms for validation and inference."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_embedding_transforms():
    """
    No augmentation for embedding extraction.
    Deterministic results are required for similarity comparisons.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(tensor):
    """Reverse normalization for visualization."""
    import torch
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)
