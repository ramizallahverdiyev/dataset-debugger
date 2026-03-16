"""
embedding_similarity.py — Method 2: Cosine similarity in feature space.

Extract penultimate-layer embeddings from the model.
For each sample, compute its average cosine similarity to other samples
with the SAME label. Low intra-class similarity → suspicious.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from models.resnet import get_embeddings_model


class EmbeddingSimilarity:
    """
    Detect mislabeled samples by measuring how well a sample
    fits its labeled class in embedding space.

    Args:
        model:  Trained ResNet-18 (full model; FC layer will be stripped).
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.device = device
        self.feature_extractor = get_embeddings_model(model).to(device).eval()

    @torch.no_grad()
    def extract_embeddings(self, loader: DataLoader) -> tuple:
        """
        Extract embeddings for all samples.

        Returns:
            embeddings: np.ndarray of shape (n_samples, 512).
            labels:     np.ndarray of shape (n_samples,).
            indices:    np.ndarray of shape (n_samples,).
        """
        all_embeds  = []
        all_labels  = []
        all_indices = []

        for imgs, labels, idx in loader:
            imgs = imgs.to(self.device)
            emb  = self.feature_extractor(imgs).cpu().numpy()
            all_embeds.append(emb)
            all_labels.append(labels.numpy())
            all_indices.append(idx.numpy())

        embeddings = np.vstack(all_embeds)
        labels     = np.concatenate(all_labels)
        indices    = np.concatenate(all_indices)

        # Reorder by original sample index
        order = np.argsort(indices)
        return embeddings[order], labels[order], indices[order]

    def compute_scores(self, loader: DataLoader, top_k: int = 50) -> np.ndarray:
        """
        Compute intra-class similarity score per sample.

        For each sample, find its top-K nearest neighbors within the
        same class and average their cosine similarity.

        Low score = sample does not fit its class → suspicious.
        We return (1 - avg_similarity) so higher score = more suspicious.

        Args:
            loader: DataLoader for the training set.
            top_k:  Number of same-class neighbors to compare against.

        Returns:
            scores: np.ndarray of shape (n_samples,) in [0, 1].
        """
        embeddings, labels, _ = self.extract_embeddings(loader)

        # L2-normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_norm = embeddings / norms

        n_samples   = len(embeddings)
        scores      = np.zeros(n_samples, dtype=np.float32)
        unique_classes = np.unique(labels)

        for cls in unique_classes:
            cls_mask    = labels == cls
            cls_indices = np.where(cls_mask)[0]
            cls_embeds  = embeddings_norm[cls_mask]

            if len(cls_indices) < 2:
                continue

            # Pairwise cosine similarity within class
            sim_matrix = cls_embeds @ cls_embeds.T  # (n_cls, n_cls)
            np.fill_diagonal(sim_matrix, -1)         # Exclude self-similarity

            k = min(top_k, len(cls_indices) - 1)
            top_k_sim = np.sort(sim_matrix, axis=1)[:, -k:]
            avg_sim    = top_k_sim.mean(axis=1)

            # Convert to suspicion: low similarity = high suspicion
            for i, orig_idx in enumerate(cls_indices):
                scores[orig_idx] = 1.0 - float(avg_sim[i])

        return scores

    def get_embeddings(self, loader: DataLoader) -> np.ndarray:
        """Public accessor for raw embeddings (used in t-SNE visualization)."""
        embeddings, labels, _ = self.extract_embeddings(loader)
        return embeddings, labels
