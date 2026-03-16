"""
corrupt.py — Label corruption logic for simulating dataset noise.

Randomly flips a percentage of training labels and saves:
  - corruption_index.json  →  {sample_idx: {original: int, corrupted: int}}
  - corruption_map.json    →  summary stats
"""

import json
import random
import numpy as np
from pathlib import Path


def corrupt_labels(dataset, corruption_rate: float = 0.10,
                   num_classes: int = 200, seed: int = 42,
                   save_dir: str = "data/corrupted") -> dict:
    """
    Randomly flip labels for a fraction of the dataset.

    Args:
        dataset:         TinyImageNetDataset instance.
        corruption_rate: Fraction of samples to corrupt (default 10%).
        num_classes:     Total number of classes.
        seed:            Random seed for reproducibility.
        save_dir:        Where to save corruption index files.

    Returns:
        labels_override: dict {sample_idx: corrupted_label}
    """
    random.seed(seed)
    np.random.seed(seed)

    n_samples = len(dataset)
    n_corrupt = int(n_samples * corruption_rate)

    # Randomly select indices to corrupt
    corrupt_indices = random.sample(range(n_samples), n_corrupt)

    labels_override = {}
    corruption_index = {}

    for idx in corrupt_indices:
        _, original_label = dataset.samples[idx]

        # Flip to a different class (never the same)
        possible = list(range(num_classes))
        possible.remove(original_label)
        corrupted_label = random.choice(possible)

        labels_override[idx] = corrupted_label
        corruption_index[str(idx)] = {
            "original": original_label,
            "corrupted": corrupted_label
        }

    # Save to disk
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "corruption_index.json", "w") as f:
        json.dump(corruption_index, f, indent=2)

    corruption_map = {
        "total_samples": n_samples,
        "corrupted_samples": n_corrupt,
        "corruption_rate": corruption_rate,
        "seed": seed,
        "num_classes": num_classes
    }
    with open(save_path / "corruption_map.json", "w") as f:
        json.dump(corruption_map, f, indent=2)

    print(f"[✓] Corrupted {n_corrupt:,} / {n_samples:,} samples ({corruption_rate*100:.0f}%)")
    print(f"[✓] Corruption index saved to {save_path}")
    return labels_override


def load_corruption_index(save_dir: str = "data/corrupted") -> tuple:
    """
    Load saved corruption index from disk.

    Returns:
        labels_override: dict {int(idx): corrupted_label}
        corruption_map:  dict with summary stats
    """
    save_path = Path(save_dir)

    with open(save_path / "corruption_index.json") as f:
        raw = json.load(f)

    # Keys stored as strings in JSON, convert back to int
    labels_override = {int(k): v["corrupted"] for k, v in raw.items()}
    corrupted_set = set(labels_override.keys())

    with open(save_path / "corruption_map.json") as f:
        corruption_map = json.load(f)

    print(f"[✓] Loaded {len(corrupted_set):,} corrupted indices from {save_path}")
    return labels_override, corruption_map


def get_corrupted_set(save_dir: str = "data/corrupted") -> set:
    """Return the set of corrupted sample indices (ground truth for evaluation)."""
    labels_override, _ = load_corruption_index(save_dir)
    return set(labels_override.keys())
