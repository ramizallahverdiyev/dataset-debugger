"""
suspicion_score.py — Combine all 4 detection signals into one ranked list.

Final formula:
    suspicion = 0.4 * loss_score
              + 0.3 * embedding_score
              + 0.2 * disagreement_score
              + 0.1 * anomaly_score
"""

import json
import numpy as np
from pathlib import Path


class SuspicionScorer:
    """
    Weighted combination of all detection method scores.

    Args:
        weights: Dict with keys matching score names. Must sum to 1.0.
    """

    DEFAULT_WEIGHTS = {
        "loss":        0.40,
        "embedding":   0.30,
        "disagreement": 0.20,
        "anomaly":     0.10,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total:.4f}"

    def compute(self, loss_scores: np.ndarray, embedding_scores: np.ndarray,
                disagreement_scores: np.ndarray, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Compute final suspicion scores.

        All input arrays must be in [0, 1] and of same shape (n_samples,).

        Returns:
            suspicion: np.ndarray of shape (n_samples,) in [0, 1].
        """
        suspicion = (
            self.weights["loss"]         * loss_scores +
            self.weights["embedding"]    * embedding_scores +
            self.weights["disagreement"] * disagreement_scores +
            self.weights["anomaly"]      * anomaly_scores
        )
        return suspicion.astype(np.float32)

    def get_top_suspicious(self, suspicion_scores: np.ndarray,
                           top_k: int = 500) -> np.ndarray:
        """Return indices of top-K most suspicious samples (sorted descending)."""
        return np.argsort(suspicion_scores)[::-1][:top_k]

    def get_flagged_mask(self, suspicion_scores: np.ndarray,
                         threshold: float = 0.5) -> np.ndarray:
        """Boolean mask of samples above suspicion threshold."""
        return suspicion_scores >= threshold

    def save_report(self, suspicion_scores: np.ndarray, labels: np.ndarray,
                    save_dir: str = "outputs/reports", top_k: int = 500):
        """Save full debugging report to JSON."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        top_indices = self.get_top_suspicious(suspicion_scores, top_k)
        flagged_mask = self.get_flagged_mask(suspicion_scores)

        report = {
            "summary": {
                "total_samples":            int(len(suspicion_scores)),
                "flagged_samples":          int(flagged_mask.sum()),
                "flagged_pct":              float(flagged_mask.mean() * 100),
                "mean_suspicion_score":     float(suspicion_scores.mean()),
                "median_suspicion_score":   float(np.median(suspicion_scores)),
                "p90_suspicion_score":      float(np.percentile(suspicion_scores, 90)),
            },
            "weights_used": self.weights,
            "top_suspicious_samples": [
                {
                    "sample_idx":      int(idx),
                    "suspicion_score": float(suspicion_scores[idx]),
                    "assigned_label":  int(labels[idx]),
                }
                for idx in top_indices
            ]
        }

        out_path = save_path / "debugger_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[✓] Report saved to {out_path}")
        return report
