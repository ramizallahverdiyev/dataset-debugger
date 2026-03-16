"""
main.py — End-to-end Dataset Debugger pipeline.

Stages:
    1. Download & prepare Tiny ImageNet
    2. Inject label corruption (10%)
    3. Train 3 ResNet-18 models with per-sample loss tracking
    4. Run all 4 detection methods
    5. Combine into suspicion scores
    6. Evaluate against ground truth
    7. Generate visualizations & report

Usage:
    python main.py                     # Full pipeline
    python main.py --stage train       # Only training
    python main.py --stage detect      # Only detection (needs trained models)
    python main.py --stage visualize   # Only visualization
"""

import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from data.download import download_tiny_imagenet
from data.dataset import TinyImageNetDataset, get_dataloaders
from data.corrupt import corrupt_labels, load_corruption_index, get_corrupted_set
from data.transforms import get_embedding_transforms

from models.resnet import build_resnet18
from training.trainer import Trainer
from training.loss_tracker import LossTracker

from detection.model_disagreement import ModelDisagreement
from detection.embedding_similarity import EmbeddingSimilarity
from detection.anomaly_detection import AnomalyDetector
from detection.loss_analysis import LossAnalysis
from detection.suspicion_score import SuspicionScorer

from evaluation.metrics import evaluate_debugger, compare_methods
from visualization.tsne_plot import plot_tsne
from visualization.sample_gallery import plot_suspicious_gallery
from visualization.score_distribution import plot_score_distribution


def load_config(base: str = "configs/base_config.yaml",
                debugger: str = "configs/debugger_config.yaml") -> dict:
    with open(base) as f:
        cfg = yaml.safe_load(f)
    with open(debugger) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def stage_prepare(cfg: dict):
    """Stage 1 & 2: Download data + inject corruption."""
    print("\n[STAGE 1] Downloading Tiny ImageNet ...")
    extract_path = download_tiny_imagenet(cfg["data"]["raw_dir"])

    print("\n[STAGE 2] Injecting label corruption ...")
    dataset = TinyImageNetDataset(
        root=cfg["data"]["dataset_root"], split="train"
    )
    labels_override = corrupt_labels(
        dataset,
        corruption_rate=cfg["corruption"]["rate"],
        num_classes=cfg["data"]["num_classes"],
        seed=cfg["corruption"]["seed"],
        save_dir=cfg["data"]["corrupted_dir"]
    )
    print(f"  Corruption rate : {cfg['corruption']['rate']*100:.0f}%")
    print(f"  Corrupted count : {len(labels_override):,}")
    return labels_override


def stage_train(cfg: dict, labels_override: dict):
    """Stage 3: Train 3 models with per-sample loss tracking."""
    print("\n[STAGE 3] Training models ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    n_models = cfg["model"]["num_models"]
    trainers = []

    for model_id in range(n_models):
        seed = cfg["training"]["seed"] + model_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n  Training Model {model_id + 1}/{n_models} (seed={seed}) ...")
        train_loader, val_loader = get_dataloaders(
            root=cfg["data"]["dataset_root"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            labels_override=labels_override
        )

        model   = build_resnet18(num_classes=cfg["data"]["num_classes"])
        trainer = Trainer(
            model=model,
            device=device,
            model_name=f"model_{chr(65 + model_id)}",
            config=cfg["training"]
        )
        trainer.train(
            train_loader, val_loader,
            n_train_samples=len(train_loader.dataset),
            save_dir=cfg["outputs"]["checkpoints"]
        )
        trainers.append(trainer)

    return trainers


def load_trainers(cfg: dict, device: str) -> list:
    """Load trained models + loss trackers from checkpoints."""
    trainers = []
    for i in range(cfg["model"]["num_models"]):
        model_name = f"model_{chr(65 + i)}"

        # Load model weights
        model = build_resnet18(num_classes=cfg["data"]["num_classes"])
        ckpt_path = f"{cfg['outputs']['checkpoints']}/{model_name}_best.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[✓] Loaded {model_name} from {ckpt_path}")

        t = Trainer(model=model, device=device, model_name=model_name)

        # Load loss tracker
        tracker_path = f"{cfg['outputs']['checkpoints']}/{model_name}_loss_tracker.npy"
        if os.path.exists(tracker_path):
            t.loss_tracker = LossTracker(n_samples=cfg["data"].get("n_train_samples", 100000))
            t.loss_tracker.load(tracker_path)
            print(f"[✓] Loaded loss tracker for {model_name}")
        else:
            print(f"[!] Loss tracker not found for {model_name} at {tracker_path}")

        trainers.append(t)

    return trainers


def stage_detect(cfg: dict, trainers: list, labels_override: dict):
    """Stage 4: Run all detection methods."""
    print("\n[STAGE 4] Running detection methods ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build embedding-transform loader (no augmentation)
    embed_dataset = TinyImageNetDataset(
        root=cfg["data"]["dataset_root"], split="train",
        transform=get_embedding_transforms(),
        labels_override=labels_override
    )
    from torch.utils.data import DataLoader
    embed_loader = DataLoader(embed_dataset, batch_size=256,
                              shuffle=False, num_workers=cfg["data"]["num_workers"])

    n_samples = len(embed_dataset)

    # Method 1 — Model Disagreement
    print("\n  [Method 1] Model Disagreement ...")
    models = [t.model for t in trainers]
    disagree = ModelDisagreement(models=models, device=device)
    disagreement_scores = disagree.compute_scores(embed_loader)
    np.save(f"{cfg['outputs']['scores']}/disagreement_scores.npy", disagreement_scores)

    # Method 2 — Embedding Similarity
    print("\n  [Method 2] Embedding Similarity ...")
    emb_sim = EmbeddingSimilarity(model=trainers[0].model, device=device)
    embedding_scores = emb_sim.compute_scores(embed_loader,
                                              top_k=cfg["detection"]["embedding_top_k"])
    embeddings, labels = emb_sim.get_embeddings(embed_loader)
    np.save(f"{cfg['outputs']['scores']}/embedding_scores.npy", embedding_scores)
    np.save(f"{cfg['outputs']['scores']}/embeddings.npy", embeddings)
    np.save(f"{cfg['outputs']['scores']}/embed_labels.npy", labels)

    # Method 3 — Anomaly Detection
    print("\n  [Method 3] Anomaly Detection ...")
    anomaly_det = AnomalyDetector(contamination=cfg["anomaly"]["contamination"],
                                  n_estimators=cfg["anomaly"]["n_estimators"])
    if cfg["anomaly"]["per_class"]:
        anomaly_scores = anomaly_det.fit_predict_per_class(embeddings, labels)
    else:
        anomaly_scores = anomaly_det.fit_predict(embeddings)
    np.save(f"{cfg['outputs']['scores']}/anomaly_scores.npy", anomaly_scores)

    # Method 4 — Loss Analysis
    print("\n  [Method 4] Loss Analysis ...")
    loss_data = trainers[0].get_loss_scores()
    loss_analysis = LossAnalysis(
        avg_loss=loss_data["avg_loss"],
        final_loss=loss_data["final_loss"],
        variance=loss_data["variance"]
    )
    loss_scores = loss_analysis.compute_scores()
    np.save(f"{cfg['outputs']['scores']}/loss_scores.npy", loss_scores)

    return {
        "disagreement": disagreement_scores,
        "embedding":    embedding_scores,
        "anomaly":      anomaly_scores,
        "loss":         loss_scores,
        "embeddings":   embeddings,
        "labels":       labels,
        "n_samples":    n_samples,
    }


def stage_combine_and_evaluate(cfg: dict, detection_results: dict):
    """Stage 5 & 6: Combine scores + evaluate."""
    print("\n[STAGE 5] Computing combined suspicion scores ...")
    scorer = SuspicionScorer(weights=cfg["suspicion_weights"])
    suspicion_scores = scorer.compute(
        loss_scores=detection_results["loss"],
        embedding_scores=detection_results["embedding"],
        disagreement_scores=detection_results["disagreement"],
        anomaly_scores=detection_results["anomaly"],
    )
    np.save(f"{cfg['outputs']['scores']}/final_suspicion_scores.npy", suspicion_scores)

    labels = detection_results["labels"]
    scorer.save_report(suspicion_scores, labels,
                       save_dir=cfg["outputs"]["reports"],
                       top_k=cfg["detection"]["top_k"])

    print("\n[STAGE 6] Evaluating against ground truth ...")
    corrupted_indices = get_corrupted_set(cfg["data"]["corrupted_dir"])
    n_samples = detection_results["n_samples"]

    metrics = evaluate_debugger(
        suspicion_scores=suspicion_scores,
        corrupted_indices=corrupted_indices,
        n_samples=n_samples,
        threshold=cfg["detection"]["threshold"],
        top_k_values=cfg["evaluation"]["top_k_values"],
        save_dir=cfg["outputs"]["reports"]
    )

    # Compare individual methods
    method_scores = {
        "Loss Analysis":      detection_results["loss"],
        "Embedding Sim":      detection_results["embedding"],
        "Model Disagreement": detection_results["disagreement"],
        "Anomaly Detection":  detection_results["anomaly"],
    }
    compare_methods(method_scores, corrupted_indices, n_samples)

    return suspicion_scores, corrupted_indices, metrics


def stage_visualize(cfg: dict, detection_results: dict,
                    suspicion_scores: np.ndarray, corrupted_indices: set,
                    labels_override: dict):
    """Stage 7: Generate visualizations."""
    print("\n[STAGE 7] Generating visualizations ...")

    # t-SNE plot
    plot_tsne(
        embeddings=detection_results["embeddings"],
        labels=detection_results["labels"],
        suspicion_scores=suspicion_scores,
        corrupted_indices=corrupted_indices,
        n_classes_to_show=cfg["visualization"]["tsne_classes"],
        save_path=f"{cfg['outputs']['figures']}/tsne_embedding.png",
        use_umap=cfg["visualization"]["use_umap"]
    )

    # Score distribution
    method_scores = {
        "Loss":         detection_results["loss"],
        "Embedding":    detection_results["embedding"],
        "Disagreement": detection_results["disagreement"],
        "Anomaly":      detection_results["anomaly"],
    }
    plot_score_distribution(
        suspicion_scores=suspicion_scores,
        method_scores=method_scores,
        corrupted_indices=corrupted_indices,
        n_samples=detection_results["n_samples"],
        save_path=f"{cfg['outputs']['figures']}/score_distribution.png"
    )

    # Suspicious gallery
    from data.transforms import get_val_transforms
    raw_dataset = TinyImageNetDataset(
        root=cfg["data"]["dataset_root"], split="train",
        transform=get_val_transforms(),
        labels_override=labels_override
    )
    plot_suspicious_gallery(
        dataset=raw_dataset,
        suspicion_scores=suspicion_scores,
        corrupted_indices=corrupted_indices,
        top_k=cfg["visualization"]["gallery_top_k"],
        save_path=f"{cfg['outputs']['figures']}/suspicious_gallery.png"
    )

    print("\n[✓] All visualizations saved to", cfg["outputs"]["figures"])


def main():
    parser = argparse.ArgumentParser(description="Dataset Debugger — Tiny ImageNet")
    parser.add_argument("--stage", default="all",
                        choices=["all", "prepare", "train", "detect", "visualize"],
                        help="Which stage to run")
    parser.add_argument("--base-config",     default="configs/base_config.yaml")
    parser.add_argument("--debugger-config", default="configs/debugger_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.debugger_config)

    # Ensure output dirs exist
    for path in cfg["outputs"].values():
        Path(path).mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "prepare"):
        labels_override = stage_prepare(cfg)

    if args.stage in ("all", "train"):
        if args.stage == "train":
            labels_override, _ = load_corruption_index(cfg["data"]["corrupted_dir"])
        trainers = stage_train(cfg, labels_override)

    if args.stage in ("all", "detect"):
        if args.stage == "detect":
            labels_override, _ = load_corruption_index(cfg["data"]["corrupted_dir"])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainers = load_trainers(cfg, device)

        detection_results = stage_detect(cfg, trainers, labels_override)
        suspicion_scores, corrupted_indices, metrics = \
            stage_combine_and_evaluate(cfg, detection_results)

    if args.stage in ("all", "visualize"):
        if args.stage == "visualize":
            labels_override, _ = load_corruption_index(cfg["data"]["corrupted_dir"])
            suspicion_scores  = np.load(f"{cfg['outputs']['scores']}/final_suspicion_scores.npy")
            corrupted_indices = get_corrupted_set(cfg["data"]["corrupted_dir"])
            detection_results = {
                "embeddings":   np.load(f"{cfg['outputs']['scores']}/embeddings.npy"),
                "labels":       np.load(f"{cfg['outputs']['scores']}/embed_labels.npy"),
                "loss":         np.load(f"{cfg['outputs']['scores']}/loss_scores.npy"),
                "embedding":    np.load(f"{cfg['outputs']['scores']}/embedding_scores.npy"),
                "disagreement": np.load(f"{cfg['outputs']['scores']}/disagreement_scores.npy"),
                "anomaly":      np.load(f"{cfg['outputs']['scores']}/anomaly_scores.npy"),
                "n_samples":    len(suspicion_scores),
            }

        stage_visualize(cfg, detection_results, suspicion_scores,
                        corrupted_indices, labels_override)

    print("\n[✓] Dataset Debugger pipeline complete!")


if __name__ == "__main__":
    main()