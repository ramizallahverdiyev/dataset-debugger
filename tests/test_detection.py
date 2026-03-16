"""
test_detection.py — Unit tests for each detection method.
"""

import pytest
import numpy as np
from detection.loss_analysis import LossAnalysis
from detection.anomaly_detection import AnomalyDetector


# ── LossAnalysis ──────────────────────────────────────────────────

def test_loss_analysis_scores_range():
    """Scores should be in [0, 1]."""
    avg   = np.random.rand(1000).astype(np.float32)
    final = np.random.rand(1000).astype(np.float32)
    var   = np.random.rand(1000).astype(np.float32)
    la    = LossAnalysis(avg, final, var)
    scores = la.compute_scores()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_loss_analysis_high_loss_flagged():
    """Samples with highest loss should get highest suspicion scores."""
    n = 1000
    avg_loss = np.random.rand(n).astype(np.float32)
    avg_loss[0] = 10.0   # Clearly highest loss
    avg_loss[1] = 9.5
    la = LossAnalysis(avg_loss, avg_loss, np.zeros(n))
    scores = la.compute_scores()
    top2 = np.argsort(scores)[-2:]
    assert 0 in top2 and 1 in top2


def test_loss_analysis_mask():
    """High loss mask should flag the correct percentile."""
    n = 1000
    avg = np.arange(n, dtype=np.float32)
    la  = LossAnalysis(avg, avg)
    mask = la.get_high_loss_mask(threshold_percentile=90)
    assert mask.sum() == pytest.approx(100, abs=2)


def test_loss_analysis_constant_scores():
    """Constant loss should return zeros (no variance to separate samples)."""
    avg = np.ones(500, dtype=np.float32)
    la  = LossAnalysis(avg, avg)
    scores = la.compute_scores()
    assert np.allclose(scores, 0.0)


# ── AnomalyDetector ───────────────────────────────────────────────

def test_anomaly_scores_range():
    """Anomaly scores should be in [0, 1]."""
    emb = np.random.randn(500, 64).astype(np.float32)
    det = AnomalyDetector(contamination=0.1)
    scores = det.fit_predict(emb)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_anomaly_detects_outliers():
    """Injected outliers should get higher anomaly scores than normal samples."""
    np.random.seed(42)
    normal  = np.random.randn(450, 64).astype(np.float32)
    outlier = (np.random.randn(50, 64) * 10 + 50).astype(np.float32)
    emb     = np.vstack([normal, outlier])

    det    = AnomalyDetector(contamination=0.1)
    scores = det.fit_predict(emb)

    mean_normal  = scores[:450].mean()
    mean_outlier = scores[450:].mean()
    assert mean_outlier > mean_normal, \
        f"Outliers not detected: normal={mean_normal:.3f}, outlier={mean_outlier:.3f}"


def test_anomaly_output_shape():
    """Output shape must match input sample count."""
    emb    = np.random.randn(300, 128).astype(np.float32)
    det    = AnomalyDetector(contamination=0.1)
    scores = det.fit_predict(emb)
    assert scores.shape == (300,)
