"""
dataset.py — PyTorch Dataset class and DataLoader factory for Tiny ImageNet.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .download import get_class_labels, get_val_annotations
from .transforms import get_train_transforms, get_val_transforms


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset with support for corrupted labels.

    Args:
        root:         Path to tiny-imagenet-200 directory.
        split:        'train' or 'val'.
        transform:    Torchvision transforms.
        labels_override: Optional dict {sample_idx: new_label} for corruption.
    """

    def __init__(self, root: str, split: str = "train", transform=None, labels_override: dict = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.labels_override = labels_override or {}

        self.class_to_idx = get_class_labels(self.root.parent)
        self.samples = []   # List of (image_path, label_idx)
        self._load_samples()

    def _load_samples(self):
        if self.split == "train":
            train_dir = self.root / "train"
            for class_id in sorted(os.listdir(train_dir)):
                class_idx = self.class_to_idx.get(class_id)
                if class_idx is None:
                    continue
                img_dir = train_dir / class_id / "images"
                for fname in sorted(os.listdir(img_dir)):
                    self.samples.append((str(img_dir / fname), class_idx))

        elif self.split == "val":
            val_dir = self.root / "val" / "images"
            annotations = get_val_annotations(self.root.parent)
            for fname in sorted(os.listdir(val_dir)):
                class_id = annotations.get(fname)
                class_idx = self.class_to_idx.get(class_id)
                if class_idx is None:
                    continue
                self.samples.append((str(val_dir / fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Apply corruption override if exists
        label = self.labels_override.get(idx, label)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, idx  # Return idx so we can track per-sample scores


def get_dataloaders(root: str, batch_size: int = 128, num_workers: int = 4,
                    labels_override: dict = None):
    """
    Build train and val DataLoaders.

    Returns:
        train_loader, val_loader
    """
    train_dataset = TinyImageNetDataset(
        root=root, split="train",
        transform=get_train_transforms(),
        labels_override=labels_override
    )
    val_dataset = TinyImageNetDataset(
        root=root, split="val",
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"[✓] Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")
    return train_loader, val_loader
