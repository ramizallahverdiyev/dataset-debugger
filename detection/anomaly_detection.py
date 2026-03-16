"""
anomaly_detection.py — Method 3: Isolation Forest on embeddings.

Finds samples that don't fit the general data distribution.
Catches corrupted images, outliers, and domain mismatches —
a different kind of problem from simple mislabeling.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Unsupervised anomaly detection on feature embeddings using Isolation Forest.

    Args:
        contamination: Expected fraction of anomalies (matches corruption rate).
        n_estimators:  Number of trees in the forest.
        random_state:  Seed for reproducibility.
    """

    def __init__(self, contamination: float = 0.10,
                 n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit Isolation Forest and return anomaly scores.

        Args:
            embeddings: np.ndarray of shape (n_samples, embedding_dim).

        Returns:
            scores: np.ndarray of shape (n_samples,) in [0, 1].
                    Higher = more anomalous (suspicious).
        """
        print("[*] Fitting Isolation Forest ...")
        X = self.scaler.fit_transform(embeddings)
        self.model.fit(X)

        # score_samples returns negative anomaly scores (lower = more anomalous)
        raw_scores = self.model.score_samples(X)

        # Normalize to [0, 1] and flip so higher = more suspicious
        scores = self._normalize(raw_scores)
        scores = 1.0 - scores

        print(f"[✓] Anomaly detection complete. "
              f"High suspicion (>0.8): {(scores > 0.8).sum():,} samples")
        return scores.astype(np.float32)

    def fit_predict_per_class(self, embeddings: np.ndarray,
                               labels: np.ndarray) -> np.ndarray:
        """
        Run Isolation Forest independently per class.
        More sensitive to within-class anomalies (e.g., mislabeled samples
        that look like outliers within their assigned class).
        """
        n_samples = len(embeddings)
        scores    = np.zeros(n_samples, dtype=np.float32)
        unique_classes = np.unique(labels)

        for cls in unique_classes:
            cls_mask = labels == cls
            cls_emb  = embeddings[cls_mask]

            if len(cls_emb) < 10:
                continue

            X = StandardScaler().fit_transform(cls_emb)
            clf = IsolationForest(
                contamination=self.contamination,
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X)
            raw = clf.score_samples(X)
            norm = 1.0 - self._normalize(raw)
            scores[cls_mask] = norm

        return scores

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1]."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
