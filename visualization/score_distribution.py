"""
score_distribution.py — Suspicion score histograms and per-method comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_score_distribution(suspicion_scores: np.ndarray,
                             method_scores: dict = None,
                             corrupted_indices: set = None,
                             n_samples: int = None,
                             save_path: str = "outputs/figures/score_distribution.png"):
    """
    Plot suspicion score distributions.

    Args:
        suspicion_scores:  Final combined scores (n_samples,).
        method_scores:     Dict {method_name: scores} for individual breakdown.
        corrupted_indices: Ground truth set for overlaid distribution.
        n_samples:         Total sample count (for corrupt mask).
        save_path:         Output path.
    """
    n_plots = 1 + (1 if method_scores else 0) + (1 if corrupted_indices else 0)
    fig = plt.figure(figsize=(7 * n_plots, 6))
    gs  = gridspec.GridSpec(1, n_plots)
    plot_idx = 0

    # --- Plot 1: Combined suspicion score distribution ---
    ax1 = fig.add_subplot(gs[plot_idx]); plot_idx += 1
    ax1.hist(suspicion_scores, bins=80, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(np.percentile(suspicion_scores, 90), color="red",
                linestyle="--", label="90th percentile")
    ax1.axvline(0.5, color="orange", linestyle="--", label="Threshold = 0.5")
    ax1.set_xlabel("Suspicion Score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Combined Suspicion Score Distribution", fontsize=13, fontweight="bold")
    ax1.legend()

    # --- Plot 2: Per-method scores stacked ---
    if method_scores:
        ax2 = fig.add_subplot(gs[plot_idx]); plot_idx += 1
        colors = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"]
        for (method_name, scores), color in zip(method_scores.items(), colors):
            ax2.hist(scores, bins=60, alpha=0.5, label=method_name,
                     color=color, edgecolor="white")
        ax2.set_xlabel("Score", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Individual Method Score Distributions", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)

    # --- Plot 3: Corrupted vs Clean score distribution ---
    if corrupted_indices and n_samples:
        ax3 = fig.add_subplot(gs[plot_idx]); plot_idx += 1
        corrupt_mask = np.zeros(n_samples, dtype=bool)
        for idx in corrupted_indices:
            if idx < n_samples:
                corrupt_mask[idx] = True

        ax3.hist(suspicion_scores[~corrupt_mask], bins=60,
                 alpha=0.6, label="Clean samples", color="steelblue")
        ax3.hist(suspicion_scores[corrupt_mask], bins=60,
                 alpha=0.6, label="Corrupted samples", color="red")
        ax3.set_xlabel("Suspicion Score", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.set_title("Score: Clean vs Corrupted Samples", fontsize=13, fontweight="bold")
        ax3.legend(fontsize=10)

    plt.suptitle("Dataset Debugger — Score Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Score distribution plot saved to {save_path}")
