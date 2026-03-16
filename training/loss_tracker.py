"""
loss_tracker.py — Records per-sample loss every epoch.

Mislabeled samples tend to have persistently high loss throughout training,
while clean samples converge to low loss quickly.
"""

import numpy as np
from collections import defaultdict


class LossTracker:
    """
    Accumulates per-sample loss across all training epochs.

    Usage:
        tracker = LossTracker(n_samples=100000)
        # Inside training loop:
        tracker.update(sample_indices, losses)
        # After training:
        avg_loss = tracker.get_average_loss()
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        # Store list of losses per sample across epochs
        self._epoch_losses = defaultdict(list)
        self.epoch = 0

    def update(self, indices: np.ndarray, losses: np.ndarray):
        """
        Record losses for a batch of samples.

        Args:
            indices: Sample indices (from DataLoader, returned as 3rd item).
            losses:  Per-sample loss values (unreduced).
        """
        for idx, loss in zip(indices, losses):
            self._epoch_losses[int(idx)].append(float(loss))

    def end_epoch(self):
        self.epoch += 1

    def get_average_loss(self) -> np.ndarray:
        """
        Compute average loss per sample across all epochs.

        Returns:
            Array of shape (n_samples,) with mean loss per sample.
            Samples never seen get loss = 0.
        """
        avg = np.zeros(self.n_samples, dtype=np.float32)
        for idx, losses in self._epoch_losses.items():
            avg[idx] = np.mean(losses)
        return avg

    def get_loss_variance(self) -> np.ndarray:
        """
        High variance = sample the model is unsure about throughout training.
        """
        var = np.zeros(self.n_samples, dtype=np.float32)
        for idx, losses in self._epoch_losses.items():
            var[idx] = np.var(losses) if len(losses) > 1 else 0.0
        return var

    def get_final_epoch_loss(self) -> np.ndarray:
        """Loss in the most recent epoch only."""
        final = np.zeros(self.n_samples, dtype=np.float32)
        for idx, losses in self._epoch_losses.items():
            final[idx] = losses[-1] if losses else 0.0
        return final

    def save(self, path: str):
        np.save(path, dict(self._epoch_losses))
        print(f"[✓] Loss tracker saved to {path}")

    def load(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self._epoch_losses = defaultdict(list, data)
        print(f"[✓] Loss tracker loaded from {path}")
