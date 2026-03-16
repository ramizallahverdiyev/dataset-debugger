"""
tsne_plot.py — t-SNE / UMAP embedding visualization.

Suspicious samples will appear far from their class cluster,
visually confirming the detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              suspicion_scores: np.ndarray = None,
              corrupted_indices: set = None,
              n_classes_to_show: int = 20,
              save_path: str = "outputs/figures/tsne_embedding.png",
              use_umap: bool = False):
    """
    Visualize embeddings with t-SNE or UMAP.

    Args:
        embeddings:         np.ndarray (n_samples, 512).
        labels:             np.ndarray (n_samples,) class indices.
        suspicion_scores:   Optional scores to highlight suspicious samples.
        corrupted_indices:  Optional ground truth for comparison.
        n_classes_to_show:  Limit to N classes for readability.
        save_path:          Output path.
        use_umap:           Use UMAP instead of t-SNE (faster for large sets).
    """
    # Subset to first N classes for cleaner visualization
    mask = labels < n_classes_to_show
    emb_sub    = embeddings[mask]
    labels_sub = labels[mask]
    orig_indices = np.where(mask)[0]

    print(f"[*] Running {'UMAP' if use_umap else 't-SNE'} on "
          f"{len(emb_sub):,} samples ({n_classes_to_show} classes) ...")

    if use_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            coords = reducer.fit_transform(emb_sub)
        except ImportError:
            print("[!] umap-learn not installed — falling back to t-SNE.")
            print("    To install: conda install -c conda-forge umap-learn")
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2, random_state=42,
                          perplexity=40, n_iter=1000).fit_transform(emb_sub)
    else:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42,
                      perplexity=40, n_iter=1000).fit_transform(emb_sub)

    fig, axes = plt.subplots(1, 2 if corrupted_indices else 1,
                              figsize=(20 if corrupted_indices else 10, 8))
    if not corrupted_indices:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab20", n_classes_to_show)

    # --- Plot 1: Colored by class ---
    ax = axes[0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=labels_sub, cmap=cmap, alpha=0.5, s=8, linewidths=0)
    ax.set_title("Embeddings Colored by Class Label", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Overlay suspicious samples if provided
    if suspicion_scores is not None:
        sus_sub = suspicion_scores[mask]
        high_sus = sus_sub > np.percentile(sus_sub, 90)
        ax.scatter(coords[high_sus, 0], coords[high_sus, 1],
                   c="red", s=25, alpha=0.8, marker="x",
                   label="Top 10% suspicious", linewidths=1.2)
        ax.legend(fontsize=10)

    # --- Plot 2: Ground truth corrupted vs clean (if available) ---
    if corrupted_indices:
        ax2 = axes[1]
        is_corrupted = np.array([orig_indices[i] in corrupted_indices
                                  for i in range(len(orig_indices))])
        colors = np.where(is_corrupted, "red", "steelblue")
        alphas = np.where(is_corrupted, 0.9, 0.3)

        ax2.scatter(coords[~is_corrupted, 0], coords[~is_corrupted, 1],
                    c="steelblue", alpha=0.3, s=6, label="Clean")
        ax2.scatter(coords[is_corrupted, 0], coords[is_corrupted, 1],
                    c="red", alpha=0.9, s=20, label="Corrupted (ground truth)")
        ax2.set_title("Ground Truth: Clean vs Corrupted", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.axis("off")

    plt.suptitle("Tiny ImageNet — Embedding Space Visualization", fontsize=16, y=1.01)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] t-SNE plot saved to {save_path}")
