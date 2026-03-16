"""
metrics.py — Evaluate debugger performance against known corruption ground truth.

Since we intentionally corrupted labels, we know exactly which samples
are wrong. We can measure how well each detection method finds them.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)


def evaluate_debugger(suspicion_scores: np.ndarray,
                      corrupted_indices: set,
                      n_samples: int,
                      threshold: float = 0.5,
                      top_k_values: list = None,
                      save_dir: str = "outputs/reports") -> dict:
    """
    Evaluate debugger performance against ground truth corrupted indices.

    Args:
        suspicion_scores:  np.ndarray (n_samples,) — final suspicion scores.
        corrupted_indices: Set of sample indices that were actually corrupted.
        n_samples:         Total number of training samples.
        threshold:         Score threshold for binary classification.
        top_k_values:      List of K values for precision@K evaluation.
        save_dir:          Where to save metrics CSV.

    Returns:
        metrics: Dict with precision, recall, F1, AUROC, AP, precision@K.
    """
    if top_k_values is None:
        top_k_values = [100, 500, 1000, 2000, 5000]

    # Ground truth binary array
    y_true = np.zeros(n_samples, dtype=int)
    for idx in corrupted_indices:
        if idx < n_samples:
            y_true[idx] = 1

    # Binary predictions at threshold
    y_pred = (suspicion_scores >= threshold).astype(int)

    metrics = {
        "threshold":        threshold,
        "total_samples":    n_samples,
        "actual_corrupted": int(y_true.sum()),
        "flagged_samples":  int(y_pred.sum()),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":            float(roc_auc_score(y_true, suspicion_scores)),
        "avg_precision":    float(average_precision_score(y_true, suspicion_scores)),
    }

    # Precision@K: among top-K flagged, what fraction are truly corrupted?
    sorted_indices = np.argsort(suspicion_scores)[::-1]
    precision_at_k = {}
    for k in top_k_values:
        top_k_indices = sorted_indices[:k]
        hits = sum(1 for idx in top_k_indices if idx in corrupted_indices)
        precision_at_k[f"precision@{k}"] = round(hits / k, 4)

    metrics["precision_at_k"] = precision_at_k

    # Print summary
    print("\n" + "="*50)
    print("  DATASET DEBUGGER — EVALUATION RESULTS")
    print("="*50)
    print(f"  Actual corrupted samples : {metrics['actual_corrupted']:,}")
    print(f"  Flagged by debugger      : {metrics['flagged_samples']:,}")
    print(f"  Precision                : {metrics['precision']:.4f}")
    print(f"  Recall                   : {metrics['recall']:.4f}")
    print(f"  F1 Score                 : {metrics['f1']:.4f}")
    print(f"  AUROC                    : {metrics['auroc']:.4f}")
    print(f"  Avg Precision (AP)       : {metrics['avg_precision']:.4f}")
    print("  Precision@K:")
    for k_key, val in precision_at_k.items():
        print(f"    {k_key:20s}: {val:.4f}")
    print("="*50 + "\n")

    # Save to JSON
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to {save_path / 'metrics.json'}")

    return metrics


def compare_methods(method_scores: dict, corrupted_indices: set,
                    n_samples: int) -> dict:
    """
    Compare all 4 individual detection methods side by side.

    Args:
        method_scores:     Dict {method_name: np.ndarray of scores}.
        corrupted_indices: Ground truth corrupted sample indices.
        n_samples:         Total training samples.

    Returns:
        comparison: Dict {method_name: metrics_dict}
    """
    comparison = {}
    print("\n" + "="*60)
    print(f"  {'Method':<25} {'Precision':>10} {'Recall':>10} {'AUROC':>10}")
    print("="*60)

    for method_name, scores in method_scores.items():
        m = evaluate_debugger(
            scores, corrupted_indices, n_samples,
            save_dir=f"outputs/reports/{method_name}"
        )
        comparison[method_name] = m
        print(f"  {method_name:<25} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['auroc']:>10.4f}")

    print("="*60 + "\n")
    return comparison
