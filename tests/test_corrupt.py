"""
test_corrupt.py — Unit tests for label corruption logic.
"""

import json
import pytest
import tempfile
import numpy as np
from unittest.mock import MagicMock
from data.corrupt import corrupt_labels, load_corruption_index, get_corrupted_set


def make_mock_dataset(n_samples: int = 1000, n_classes: int = 200):
    """Create a minimal mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__ = lambda self: n_samples
    dataset.samples = [(f"img_{i}.jpg", i % n_classes) for i in range(n_samples)]
    return dataset


def test_corruption_rate():
    """Corrupted sample count should match the requested rate."""
    dataset = make_mock_dataset(n_samples=1000, n_classes=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        overrides = corrupt_labels(dataset, corruption_rate=0.10,
                                   num_classes=200, save_dir=tmpdir)
        assert abs(len(overrides) - 100) <= 2  # Allow ±2 rounding


def test_no_self_corruption():
    """Corrupted label must differ from original label."""
    dataset = make_mock_dataset(n_samples=500, n_classes=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        overrides = corrupt_labels(dataset, corruption_rate=0.20,
                                   num_classes=200, save_dir=tmpdir)
        for idx, new_label in overrides.items():
            original_label = dataset.samples[idx][1]
            assert new_label != original_label, \
                f"Sample {idx}: corrupted label same as original ({original_label})"


def test_corruption_index_saved():
    """Corruption index JSON should be saved to disk."""
    dataset = make_mock_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_labels(dataset, save_dir=tmpdir)
        index_path = f"{tmpdir}/corruption_index.json"
        map_path   = f"{tmpdir}/corruption_map.json"
        assert open(index_path), "corruption_index.json not found"
        assert open(map_path),   "corruption_map.json not found"


def test_load_roundtrip():
    """Saved corruption index should load back with same content."""
    dataset = make_mock_dataset(n_samples=200, n_classes=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        original = corrupt_labels(dataset, corruption_rate=0.10,
                                   num_classes=200, save_dir=tmpdir)
        loaded, _ = load_corruption_index(tmpdir)
        assert set(original.keys()) == set(loaded.keys())
        for k in original:
            assert original[k] == loaded[k]


def test_reproducible_with_seed():
    """Same seed should produce identical corruption."""
    dataset = make_mock_dataset()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        o1 = corrupt_labels(dataset, seed=42, save_dir=d1)
        o2 = corrupt_labels(dataset, seed=42, save_dir=d2)
        assert o1 == o2


def test_different_seeds_differ():
    """Different seeds should produce different corruptions."""
    dataset = make_mock_dataset()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        o1 = corrupt_labels(dataset, seed=1, save_dir=d1)
        o2 = corrupt_labels(dataset, seed=2, save_dir=d2)
        assert o1 != o2
