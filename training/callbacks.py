"""
callbacks.py — Early stopping and model checkpointing.
"""

import torch
from pathlib import Path


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Save best model weights based on validation accuracy."""

    def __init__(self, save_dir: str = "outputs/checkpoints",
                 model_name: str = "model", verbose: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.verbose = verbose
        self.best_acc = 0.0
        self.best_path = None

    def __call__(self, model: torch.nn.Module, val_acc: float, epoch: int):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc
            }, path)
            self.best_path = path
            if self.verbose:
                print(f"  [✓] Checkpoint saved: {path} (val_acc={val_acc:.4f})")

    def save_epoch(self, model: torch.nn.Module, epoch: int):
        """Save checkpoint at specific epoch."""
        path = self.save_dir / f"{self.model_name}_epoch{epoch}.pth"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)
