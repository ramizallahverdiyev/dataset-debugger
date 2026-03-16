"""
sample_gallery.py — Display top-N suspicious samples as an image grid.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def plot_suspicious_gallery(dataset, suspicion_scores: np.ndarray,
                             corrupted_indices: set = None,
                             top_k: int = 20,
                             save_path: str = "outputs/figures/suspicious_gallery.png",
                             class_names: dict = None):
    """
    Show a grid of the most suspicious samples with their labels and scores.

    Args:
        dataset:           TinyImageNetDataset (untransformed, for raw images).
        suspicion_scores:  np.ndarray (n_samples,) final suspicion scores.
        corrupted_indices: Ground truth set (used to mark TP/FP with border color).
        top_k:             Number of suspicious samples to show.
        save_path:         Output path.
        class_names:       Optional {class_idx: class_name} for display.
    """
    top_indices = np.argsort(suspicion_scores)[::-1][:top_k]

    cols = 5
    rows = int(np.ceil(top_k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    for plot_idx, sample_idx in enumerate(top_indices):
        ax = axes[plot_idx]
        img, label, _ = dataset[sample_idx]

        # img is a tensor — convert to displayable format
        if hasattr(img, "numpy"):
            from data.transforms import denormalize
            img_disp = denormalize(img).permute(1, 2, 0).numpy()
            img_disp = np.clip(img_disp, 0, 1)
        else:
            img_disp = img

        ax.imshow(img_disp)

        score = suspicion_scores[sample_idx]
        label_name = class_names.get(label, str(label)) if class_names else str(label)

        # Border: green = true positive (actually corrupted), red = false positive
        if corrupted_indices is not None:
            is_tp = sample_idx in corrupted_indices
            border_color = "#00C853" if is_tp else "#FF1744"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

        ax.set_title(f"#{sample_idx}\nLabel: {label_name}\nScore: {score:.3f}",
                     fontsize=8, pad=3)
        ax.axis("off")

    # Hide unused subplots
    for i in range(top_k, len(axes)):
        axes[i].axis("off")

    # Legend
    if corrupted_indices:
        tp_patch = patches.Patch(color="#00C853", label="True Positive (actually corrupted)")
        fp_patch = patches.Patch(color="#FF1744", label="False Positive (clean sample)")
        fig.legend(handles=[tp_patch, fp_patch], loc="lower center",
                   ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Top-{top_k} Most Suspicious Samples", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Suspicious gallery saved to {save_path}")
