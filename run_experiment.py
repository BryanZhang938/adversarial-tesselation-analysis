"""
Main experiment runner.
Trains models under standard and adversarial settings, computes tessellation
statistics at checkpoints, and generates comparison figures.

Usage:
    python run_experiment.py
    python run_experiment.py --dataset concentric_rings
    python run_experiment.py --epochs 300 --epsilon 0.15
"""

import os
import sys
import argparse
import json
import copy
import numpy as np
import torch

from configs.experiment_config import (
    ExperimentConfig, DataConfig, ModelConfig, TrainConfig,
    AdversarialConfig, TessellationConfig,
)
from src.datasets import get_dataset, get_dataloader
from src.models import make_relu_mlp, count_parameters
from src.train import train_model
from src.tessellation_analysis import analyze_checkpoint
from src.visualization import (
    plot_dataset,
    plot_training_comparison,
    plot_epoch_snapshots,
    plot_boundary_distance_histograms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tessellation dynamics experiment")
    parser.add_argument("--dataset", type=str, default="spirals",
                        choices=["spirals", "concentric_rings", "moons"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 32, 32])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--grid_resolution", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_config(args):
    """Build ExperimentConfig from command line arguments."""
    cfg = ExperimentConfig()
    cfg.data.dataset = args.dataset
    cfg.data.n_samples = args.n_samples
    cfg.data.noise = args.noise
    cfg.model.hidden_dims = args.hidden_dims
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    cfg.adv.epsilon = args.epsilon
    cfg.adv.num_steps = args.pgd_steps
    cfg.adv.step_size = args.epsilon / 4
    cfg.tess.resolution = args.grid_resolution
    cfg.seed = args.seed
    cfg.device = args.device

    # Adjust checkpoint epochs to be within training range
    cfg.tess.checkpoint_epochs = [
        e for e in [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
        if e <= args.epochs
    ]

    return cfg


def run_single_experiment(config, X_train, y_train, dataset, run_name, device):
    """Train one model and analyze checkpoints."""
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Build model
    model = make_relu_mlp(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.output_dim,
    )
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Model: {config.model.hidden_dims}, "
          f"params: {count_parameters(model)}")
    print(f"{'='*60}")

    # Train
    dataloader = get_dataloader(dataset, batch_size=config.train.batch_size)
    checkpoint_dir = os.path.join(config.checkpoint_dir, run_name)
    history = train_model(
        model, dataloader, config,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        device=device,
    )

    # Analyze checkpoints
    model_factory = lambda: make_relu_mlp(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.output_dim,
    )

    all_stats = []
    all_grid_data = []
    for epoch in config.tess.checkpoint_epochs:
        ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_epoch{epoch:04d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        print(f"  Analyzing epoch {epoch}...")
        stats, grid_data = analyze_checkpoint(
            model_factory, ckpt_path, X_train, config, device=device
        )
        all_stats.append(stats)
        all_grid_data.append(grid_data)

        # Print summary
        print(f"    Regions: {stats['num_regions_grid']} | "
              f"Local complexity: {stats.get('local_region_count_mean', 'N/A'):.1f} | "
              f"Boundary dist: {stats.get('boundary_dist_mean', 'N/A'):.4f}")

    return history, all_stats, all_grid_data


def main():
    args = parse_args()
    config = build_config(args)

    # Create output directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Generate dataset
    dataset, X_train, y_train = get_dataset(
        config.data.dataset,
        n_samples=config.data.n_samples,
        noise=config.data.noise,
        seed=config.data.seed,
    )
    print(f"Dataset: {config.data.dataset}, n={len(X_train)}")

    # Plot dataset
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_dataset(X_train, y_train, title=f"Dataset: {config.data.dataset}", ax=ax)
    fig.savefig(os.path.join(config.figure_dir, "dataset.pdf"))
    plt.close()

    # ---- Standard Training ----
    config_std = copy.deepcopy(config)
    config_std.adv.enabled = False

    hist_std, stats_std, grids_std = run_single_experiment(
        config_std, X_train, y_train, dataset,
        run_name="standard", device=config.device
    )

    # ---- Adversarial Training ----
    config_adv = copy.deepcopy(config)
    config_adv.adv.enabled = True

    hist_adv, stats_adv, grids_adv = run_single_experiment(
        config_adv, X_train, y_train, dataset,
        run_name="adversarial", device=config.device
    )

    # ---- Generate Figures ----
    print("\nGenerating figures...")

    # Main comparison plot
    plot_training_comparison(stats_std, stats_adv, figure_dir=config.figure_dir)

    # Epoch snapshots
    snapshot_epochs = [s["epoch"] for s in stats_std]
    plot_epoch_snapshots(
        grids_std, X_train, y_train, snapshot_epochs,
        run_name="standard", figure_dir=config.figure_dir
    )
    plot_epoch_snapshots(
        grids_adv, X_train, y_train, snapshot_epochs,
        run_name="adversarial", figure_dir=config.figure_dir
    )

    # Boundary distance histograms
    plot_boundary_distance_histograms(
        stats_std, stats_adv, epoch_idx=-1, figure_dir=config.figure_dir
    )

    # ---- Save raw results ----
    def serialize_stats(stats_list):
        """Convert stats to JSON-serializable format."""
        out = []
        for s in stats_list:
            d = {}
            for k, v in s.items():
                if isinstance(v, np.ndarray):
                    d[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    d[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    d[k] = int(v)
                else:
                    try:
                        json.dumps(v)
                        d[k] = v
                    except (TypeError, ValueError):
                        pass  # skip non-serializable
            out.append(d)
        return out

    results = {
        "config": {
            "dataset": config.data.dataset,
            "hidden_dims": config.model.hidden_dims,
            "epochs": config.train.epochs,
            "epsilon": config.adv.epsilon,
        },
        "standard": serialize_stats(stats_std),
        "adversarial": serialize_stats(stats_adv),
    }
    results_path = os.path.join(config.results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
