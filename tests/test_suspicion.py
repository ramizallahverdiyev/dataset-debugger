"""
test_suspicion.py — Unit tests for the combined suspicion scorer.
"""

import json
import pytest
import tempfile
import numpy as np
from detection.suspicion_score import SuspicionScorer


def make_scores(n=1000):
    return {
        "loss_scores":         np.random.rand(n).astype(np.float32),
        "embedding_scores":    np.random.rand(n).astype(np.float32),
        "disagreement_scores": np.random.rand(n).astype(np.float32),
        "anomaly_scores":      np.random.rand(n).astype(np.float32),
    }


def test_weights_must_sum_to_one():
    """Invalid weights should raise an assertion error."""
    with pytest.raises(AssertionError):
        SuspicionScorer(weights={"loss": 0.5, "embedding": 0.5,
                                  "disagreement": 0.5, "anomaly": 0.5})


def test_output_range():
    """Combined suspicion score must be in [0, 1]."""
    scorer = SuspicionScorer()
    out = scorer.compute(**make_scores())
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_shape():
    """Output shape must match input."""
    scorer = SuspicionScorer()
    out = scorer.compute(**make_scores(n=500))
    assert out.shape == (500,)


def test_top_suspicious_sorted():
    """Top suspicious indices should be sorted by score descending."""
    scorer = SuspicionScorer()
    combined = scorer.compute(**make_scores())
    top = scorer.get_top_suspicious(combined, top_k=100)
    scores_of_top = combined[top]
    assert np.all(scores_of_top[:-1] >= scores_of_top[1:])


def test_high_score_sample_flagged():
    """Sample with maximum score should appear in top suspicious."""
    scorer = SuspicionScorer()
    s = make_scores(n=1000)
    for key in s:
        s[key][42] = 1.0
    combined = scorer.compute(**s)
    top = scorer.get_top_suspicious(combined, top_k=10)
    assert 42 in top


def test_flagged_mask_threshold():
    """Flagged mask should respect the threshold."""
    scorer = SuspicionScorer()
    combined = scorer.compute(**make_scores())
    mask = scorer.get_flagged_mask(combined, threshold=0.7)
    assert np.all(combined[mask] >= 0.7)
    assert np.all(combined[~mask] < 0.7)


def test_save_report():
    """Report JSON should be saved with correct structure."""
    scorer = SuspicionScorer()
    combined = scorer.compute(**make_scores())
    labels = np.zeros(1000, dtype=int)

    with tempfile.TemporaryDirectory() as tmpdir:
        report = scorer.save_report(combined, labels, save_dir=tmpdir, top_k=50)
        assert "summary" in report
        assert "top_suspicious_samples" in report
        assert len(report["top_suspicious_samples"]) == 50
        assert report["summary"]["total_samples"] == 1000