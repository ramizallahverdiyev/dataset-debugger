# Dataset Debugger — Tiny ImageNet

A data-centric AI tool that automatically detects mislabeled, suspicious,
and anomalous samples in a dataset using 4 complementary detection methods.

---

## Project Structure

```
dataset-debugger/
├── data/               # Download, dataset class, corruption injection
├── models/             # ResNet-18 adapted for 64x64
├── training/           # Trainer with per-sample loss tracking
├── detection/          # 4 detection methods + suspicion scorer
├── evaluation/         # Precision, recall, AUROC against ground truth
├── visualization/      # t-SNE plots, sample gallery, score histograms
├── configs/            # YAML configs for all hyperparameters
├── tests/              # Unit tests for each module
├── outputs/            # Checkpoints, scores, figures, reports
└── main.py             # Full pipeline runner
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python main.py

# 3. Run specific stage
python main.py --stage prepare     # Download + corrupt data
python main.py --stage train       # Train 3 models
python main.py --stage detect      # Run detection methods
python main.py --stage visualize   # Generate plots
```

---

## Detection Methods

| # | Method | Signal | Weight |
|---|--------|--------|--------|
| 1 | Training Loss Analysis | Persistently high loss per sample | 40% |
| 2 | Embedding Similarity | Low cosine sim within class cluster | 30% |
| 3 | Model Disagreement | Ensemble vote vs dataset label | 20% |
| 4 | Anomaly Detection | Isolation Forest on embeddings | 10% |

Final suspicion score = weighted combination of all 4 signals.

---

## Expected Results

After training on 10% corrupted Tiny ImageNet:

| Metric | Expected Value |
|--------|---------------|
| Precision | ~0.85 |
| Recall | ~0.72 |
| AUROC | ~0.91 |
| Precision@500 | ~0.88 |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/scores/final_suspicion_scores.npy` | Per-sample suspicion scores |
| `outputs/reports/debugger_report.json` | Top-K flagged samples |
| `outputs/reports/metrics.json` | Evaluation metrics |
| `outputs/figures/tsne_embedding.png` | Embedding space visualization |
| `outputs/figures/suspicious_gallery.png` | Top suspicious sample grid |
| `outputs/figures/score_distribution.png` | Score histograms |

---

## Running Tests

```bash
pytest tests/ -v
```
