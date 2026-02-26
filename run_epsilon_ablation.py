"""
Epsilon ablation study: run adversarial training with different perturbation
budgets on the spirals dataset and compare final-epoch tessellation statistics.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import json
import copy
import numpy as np
import torch

from configs.experiment_config import ExperimentConfig
from src.datasets import get_dataset, get_dataloader
from src.models import make_relu_mlp, count_parameters
from src.train import train_model
from src.tessellation_analysis import analyze_checkpoint, compute_grid_statistics
from src.visualization import plot_decision_regions, plot_tessellation_regions


def run_ablation():
    epsilons = [0.05, 0.1, 0.2]
    dataset_name = "spirals"
    epochs = 200
    seed = 42

    base_cfg = ExperimentConfig()
    base_cfg.data.dataset = dataset_name
    base_cfg.data.n_samples = 500
    base_cfg.data.noise = 0.1
    base_cfg.train.epochs = epochs
    base_cfg.tess.resolution = 200
    base_cfg.seed = seed
    base_cfg.device = "cpu"
    base_cfg.tess.checkpoint_epochs = [
        e for e in [1, 5, 10, 20, 50, 100, 150, 200] if e <= epochs
    ]

    figure_dir = "figures/epsilon_ablation"
    os.makedirs(figure_dir, exist_ok=True)

    dataset, X_train, y_train = get_dataset(
        dataset_name, n_samples=500, noise=0.1, seed=seed
    )

    # Also train standard for reference
    all_results = {}

    # Standard training
    print("=" * 60)
    print("Training: standard (no adversarial)")
    print("=" * 60)
    cfg_std = copy.deepcopy(base_cfg)
    cfg_std.adv.enabled = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_std = make_relu_mlp(
        input_dim=cfg_std.model.input_dim,
        hidden_dims=cfg_std.model.hidden_dims,
        output_dim=cfg_std.model.output_dim,
    )
    dataloader = get_dataloader(dataset, batch_size=cfg_std.train.batch_size)
    ckpt_dir = os.path.join("checkpoints", "ablation_standard")
    hist_std = train_model(
        model_std, dataloader, cfg_std,
        checkpoint_dir=ckpt_dir,
        run_name="ablation_standard",
        device="cpu",
    )

    model_std.eval()
    stats_std, grid_std = compute_grid_statistics(
        model_std, domain_range=cfg_std.data.domain_range,
        resolution=cfg_std.tess.resolution,
        data_points=X_train, device="cpu"
    )
    stats_std["clean_acc"] = hist_std["train_acc"][-1]
    all_results["standard"] = stats_std
    all_results["standard_grid"] = grid_std

    # Adversarial training for each epsilon
    for eps in epsilons:
        run_name = f"eps_{eps:.2f}"
        print(f"\n{'=' * 60}")
        print(f"Training: adversarial, epsilon = {eps}")
        print(f"{'=' * 60}")

        cfg = copy.deepcopy(base_cfg)
        cfg.adv.enabled = True
        cfg.adv.epsilon = eps
        cfg.adv.step_size = eps / 4

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = make_relu_mlp(
            input_dim=cfg.model.input_dim,
            hidden_dims=cfg.model.hidden_dims,
            output_dim=cfg.model.output_dim,
        )
        ckpt_dir = os.path.join("checkpoints", f"ablation_{run_name}")
        hist = train_model(
            model, dataloader, cfg,
            checkpoint_dir=ckpt_dir,
            run_name=f"ablation_{run_name}",
            device="cpu",
        )

        model.eval()
        stats, grid_data = compute_grid_statistics(
            model, domain_range=cfg.data.domain_range,
            resolution=cfg.tess.resolution,
            data_points=X_train, device="cpu"
        )
        stats["clean_acc"] = hist["train_acc"][-1]
        stats["adv_acc"] = hist["adv_acc"][-1]
        stats["epsilon"] = eps

        all_results[run_name] = stats
        all_results[f"{run_name}_grid"] = grid_data

    # ===== Generate comparison figures =====

    eps_labels = ["Standard"] + [f"ε={e}" for e in epsilons]
    eps_keys = ["standard"] + [f"eps_{e:.2f}" for e in epsilons]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    # --- Figure 1: 4-panel bar chart of key metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Number of linear regions
    vals = [all_results[k]["num_regions_grid"] for k in eps_keys]
    axes[0, 0].bar(eps_labels, vals, color=colors)
    axes[0, 0].set_ylabel("# Linear Regions")
    axes[0, 0].set_title("(a) Region Count")
    for i, v in enumerate(vals):
        axes[0, 0].text(i, v + 20, str(v), ha='center', fontsize=9)

    # (b) Local complexity
    vals = [all_results[k].get("local_region_count_mean", 0) for k in eps_keys]
    axes[0, 1].bar(eps_labels, vals, color=colors)
    axes[0, 1].set_ylabel("Mean Local Region Count")
    axes[0, 1].set_title("(b) Local Complexity Near Data")
    for i, v in enumerate(vals):
        axes[0, 1].text(i, v + 0.3, f"{v:.1f}", ha='center', fontsize=9)

    # (c) Boundary distance
    vals = [all_results[k].get("boundary_dist_mean", 0) for k in eps_keys]
    axes[1, 0].bar(eps_labels, vals, color=colors)
    axes[1, 0].set_ylabel("Mean Boundary Distance")
    axes[1, 0].set_title("(c) Data-to-Boundary Distance")
    for i, v in enumerate(vals):
        axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

    # (d) Clean accuracy
    vals = [all_results[k].get("clean_acc", 0) for k in eps_keys]
    axes[1, 1].bar(eps_labels, vals, color=colors)
    axes[1, 1].set_ylabel("Clean Accuracy")
    axes[1, 1].set_title("(d) Clean Accuracy")
    axes[1, 1].set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)

    fig.suptitle("Epsilon Ablation: Tessellation Statistics (Spirals, Epoch 200)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "epsilon_ablation_bars.pdf"))
    plt.savefig(os.path.join(figure_dir, "epsilon_ablation_bars.png"))
    plt.close()
    print(f"\nSaved epsilon_ablation_bars to {figure_dir}/")

    # --- Figure 2: Side-by-side decision regions ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (key, label) in enumerate(zip(eps_keys, eps_labels)):
        gd = all_results[f"{key}_grid"]
        plot_decision_regions(gd, X_train, y_train,
                              title=f"{label}", ax=axes[0, i])
        plot_tessellation_regions(gd, X_train, y_train,
                                   title=f"{label}", ax=axes[1, i])

    axes[0, 0].set_ylabel("Decision Regions", fontsize=12)
    axes[1, 0].set_ylabel("Linear Regions", fontsize=12)
    fig.suptitle("Epsilon Ablation: Decision Regions & Tessellation (Spirals, Epoch 200)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "epsilon_ablation_regions.pdf"))
    plt.savefig(os.path.join(figure_dir, "epsilon_ablation_regions.png"))
    plt.close()
    print(f"Saved epsilon_ablation_regions to {figure_dir}/")

    # --- Figure 3: Boundary distance distributions ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label) in enumerate(zip(eps_keys, eps_labels)):
        dists = all_results[key].get("boundary_distances", [])
        if len(dists) > 0:
            ax.hist(dists, bins=30, alpha=0.4, label=label,
                    color=colors[i], density=True)
    ax.set_xlabel("Distance to Decision Boundary")
    ax.set_ylabel("Density")
    ax.set_title("Boundary Distance Distribution by Epsilon")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "epsilon_ablation_boundary_hist.pdf"))
    plt.savefig(os.path.join(figure_dir, "epsilon_ablation_boundary_hist.png"))
    plt.close()
    print(f"Saved epsilon_ablation_boundary_hist to {figure_dir}/")

    # Print summary table
    print("\n" + "=" * 70)
    print("EPSILON ABLATION SUMMARY (Spirals, Epoch 200)")
    print("=" * 70)
    print(f"{'Setting':<15} {'Regions':>10} {'Local Cmplx':>12} {'Bdry Dist':>12} {'Clean Acc':>10}")
    print("-" * 70)
    for key, label in zip(eps_keys, eps_labels):
        s = all_results[key]
        print(f"{label:<15} {s['num_regions_grid']:>10} "
              f"{s.get('local_region_count_mean', 0):>12.1f} "
              f"{s.get('boundary_dist_mean', 0):>12.4f} "
              f"{s.get('clean_acc', 0):>10.4f}")
    print("=" * 70)

    # Save results JSON
    serializable = {}
    for key in eps_keys:
        s = all_results[key]
        serializable[key] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.float32, np.float64)) else v)
            for k, v in s.items()
        }
    with open(os.path.join("results", "epsilon_ablation.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to results/epsilon_ablation.json")


if __name__ == "__main__":
    run_ablation()
