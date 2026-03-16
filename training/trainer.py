"""
trainer.py — Training loop with per-sample loss tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from .loss_tracker import LossTracker
from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """
    Trains a ResNet-18 on Tiny ImageNet with full per-sample loss tracking.

    Args:
        model:       PyTorch model.
        device:      'cuda' or 'cpu'.
        model_name:  Identifier used for checkpoint filenames (e.g., 'model_A').
        config:      Dict of hyperparameters (lr, epochs, weight_decay, etc.).
    """

    def __init__(self, model: nn.Module, device: str = "cuda",
                 model_name: str = "model", config: dict = None):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.config = config or {}

        lr           = self.config.get("lr", 0.1)
        weight_decay = self.config.get("weight_decay", 1e-4)
        momentum     = self.config.get("momentum", 0.9)
        self.epochs  = self.config.get("epochs", 30)

        self.criterion = nn.CrossEntropyLoss(reduction="none")  # Keep per-sample loss
        self.optimizer = SGD(model.parameters(), lr=lr,
                             momentum=momentum, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.loss_tracker = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              n_train_samples: int, save_dir: str = "outputs/checkpoints"):

        self.loss_tracker = LossTracker(n_samples=n_train_samples)
        early_stop  = EarlyStopping(patience=self.config.get("patience", 7))
        checkpoint  = ModelCheckpoint(save_dir=save_dir, model_name=self.model_name)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)
            self.scheduler.step()
            self.loss_tracker.end_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch:02d}/{self.epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            checkpoint(self.model, val_acc, epoch)

            if early_stop(val_loss):
                print(f"[!] Early stopping at epoch {epoch}")
                break

        # Save final loss tracker
        tracker_path = Path(save_dir) / f"{self.model_name}_loss_tracker.npy"
        self.loss_tracker.save(str(tracker_path))
        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for imgs, labels, indices in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            losses  = self.criterion(outputs, labels)  # shape: (batch,)

            loss = losses.mean()
            loss.backward()
            self.optimizer.step()

            # Track per-sample loss
            self.loss_tracker.update(
                indices.cpu().numpy(),
                losses.detach().cpu().numpy()
            )
            total_loss += loss.item()

        return total_loss / len(loader)

    def _val_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels, indices in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                losses  = self.criterion(outputs, labels)
                total_loss += losses.mean().item()
                preds   = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        return total_loss / len(loader), correct / total

    def get_loss_scores(self) -> dict:
        """Return per-sample loss arrays from tracker."""
        return {
            "avg_loss":   self.loss_tracker.get_average_loss(),
            "final_loss": self.loss_tracker.get_final_epoch_loss(),
            "variance":   self.loss_tracker.get_loss_variance(),
        }
