"""
model_disagreement.py — Method 1: Multi-model vote vs dataset label.

Train 3 models with different seeds. If most models disagree with the
dataset label, the sample is likely mislabeled.

disagreement_score = num_models_predicting_differently / total_models
"""

import torch
import numpy as np
from torch.utils.data import DataLoader


class ModelDisagreement:
    """
    Compute per-sample disagreement score across an ensemble of models.

    Args:
        models: List of trained PyTorch models.
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, models: list, device: str = "cuda"):
        self.models = models
        self.device = device

    @torch.no_grad()
    def compute_scores(self, loader: DataLoader) -> np.ndarray:
        """
        Compute disagreement score for every sample.

        Returns:
            scores: np.ndarray of shape (n_samples,) in [0, 1].
                    1.0 = all models disagree with label (very suspicious).
                    0.0 = all models agree with label (clean).
        """
        n_samples  = len(loader.dataset)
        n_models   = len(self.models)
        all_preds  = np.zeros((n_models, n_samples), dtype=np.int64)

        for m_idx, model in enumerate(self.models):
            model.eval().to(self.device)
            for imgs, labels, indices in loader:
                imgs = imgs.to(self.device)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                for i, idx in enumerate(indices.numpy()):
                    all_preds[m_idx, idx] = preds[i]

        # Get dataset labels
        dataset_labels = np.array([label for _, label, _ in loader.dataset])

        # Disagreement: fraction of models that predict differently from dataset label
        scores = np.zeros(n_samples, dtype=np.float32)
        for sample_idx in range(n_samples):
            label = dataset_labels[sample_idx]
            disagreeing = np.sum(all_preds[:, sample_idx] != label)
            scores[sample_idx] = disagreeing / n_models

        return scores

    def get_prediction_table(self, loader: DataLoader) -> dict:
        """
        Return full prediction table for inspection.

        Returns:
            dict with keys: 'dataset_labels', 'model_predictions', 'scores'
        """
        n_samples = len(loader.dataset)
        n_models  = len(self.models)
        all_preds = np.zeros((n_models, n_samples), dtype=np.int64)

        for m_idx, model in enumerate(self.models):
            model.eval().to(self.device)
            with torch.no_grad():
                for imgs, labels, indices in loader:
                    imgs  = imgs.to(self.device)
                    preds = model(imgs).argmax(dim=1).cpu().numpy()
                    for i, idx in enumerate(indices.numpy()):
                        all_preds[m_idx, idx] = preds[i]

        dataset_labels = np.array([label for _, label, _ in loader.dataset])
        scores = self.compute_scores(loader)

        return {
            "dataset_labels":    dataset_labels,
            "model_predictions": all_preds,
            "scores":            scores
        }
