"""
loss_analysis.py — Method 4: Persistent high training loss detection.

Clean samples → model learns quickly, loss drops.
Mislabeled samples → model keeps getting contradictory signal, loss stays high.

This method uses the LossTracker output from training.
"""

import numpy as np


class LossAnalysis:
    """
    Convert per-sample training loss history into suspicion scores.

    Args:
        avg_loss:   np.ndarray (n_samples,) — mean loss across all epochs.
        final_loss: np.ndarray (n_samples,) — loss in the final epoch.
        variance:   np.ndarray (n_samples,) — loss variance across epochs.
    """

    def __init__(self, avg_loss: np.ndarray, final_loss: np.ndarray,
                 variance: np.ndarray = None):
        self.avg_loss   = avg_loss
        self.final_loss = final_loss
        self.variance   = variance if variance is not None else np.zeros_like(avg_loss)

    def compute_scores(self, weight_avg: float = 0.5,
                       weight_final: float = 0.4,
                       weight_var: float = 0.1) -> np.ndarray:
        """
        Combine average loss, final loss, and variance into one suspicion score.

        All components are normalized to [0, 1] before combining.

        Args:
            weight_avg:   Weight for average training loss.
            weight_final: Weight for final epoch loss.
            weight_var:   Weight for loss variance.

        Returns:
            scores: np.ndarray of shape (n_samples,) in [0, 1].
        """
        avg_norm   = self._normalize(self.avg_loss)
        final_norm = self._normalize(self.final_loss)
        var_norm   = self._normalize(self.variance)

        scores = (weight_avg   * avg_norm +
                  weight_final * final_norm +
                  weight_var   * var_norm)

        return scores.astype(np.float32)

    def get_high_loss_mask(self, threshold_percentile: float = 90) -> np.ndarray:
        """
        Boolean mask of samples above a loss percentile threshold.

        Args:
            threshold_percentile: e.g., 90 = top 10% highest loss samples.

        Returns:
            Boolean np.ndarray of shape (n_samples,).
        """
        threshold = np.percentile(self.avg_loss, threshold_percentile)
        return self.avg_loss >= threshold

    def get_loss_statistics(self) -> dict:
        return {
            "mean_avg_loss":    float(self.avg_loss.mean()),
            "median_avg_loss":  float(np.median(self.avg_loss)),
            "p90_avg_loss":     float(np.percentile(self.avg_loss, 90)),
            "p95_avg_loss":     float(np.percentile(self.avg_loss, 95)),
            "max_avg_loss":     float(self.avg_loss.max()),
        }

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    @classmethod
    def from_tracker(cls, tracker) -> "LossAnalysis":
        """Construct from a LossTracker instance."""
        return cls(
            avg_loss=tracker.get_average_loss(),
            final_loss=tracker.get_final_epoch_loss(),
            variance=tracker.get_loss_variance()
        )
