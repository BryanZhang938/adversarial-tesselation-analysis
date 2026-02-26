"""
Visualization functions for tessellation dynamics paper.
Generates publication-quality figures comparing standard vs adversarial training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Use a clean style for paper figures
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_dataset(X, y, title="Dataset", ax=None):
    """Plot a 2D classification dataset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['#2196F3', '#FF5722']
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=10,
                   alpha=0.6, label=f"Class {cls}")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    return ax


def plot_decision_regions(grid_data, X=None, y=None, title="", ax=None):
    """Plot decision regions from grid predictions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    cmap = mcolors.ListedColormap(['#BBDEFB', '#FFCCBC'])
    ax.contourf(grid_data["grid_x"], grid_data["grid_y"],
                grid_data["preds_grid"], levels=[-0.5, 0.5, 1.5],
                cmap=cmap, alpha=0.7)

    # Decision boundary
    ax.contour(grid_data["grid_x"], grid_data["grid_y"],
               grid_data["preds_grid"], levels=[0.5],
               colors='red', linewidths=2)

    if X is not None and y is not None:
        colors = ['#1565C0', '#D84315']
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=15,
                       edgecolors='white', linewidth=0.5, zorder=5)

    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def plot_tessellation_regions(grid_data, X=None, y=None, title="", ax=None):
    """Plot the activation-pattern-based tessellation (linear regions)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    region_ids = grid_data["region_ids_grid"]
    n_regions = region_ids.max() + 1

    # Use a randomized colormap so adjacent regions are visually distinct
    rng = np.random.RandomState(42)
    colors_arr = rng.rand(n_regions, 3) * 0.6 + 0.2
    cmap = mcolors.ListedColormap(colors_arr)

    ax.imshow(region_ids, extent=[
        grid_data["grid_x"].min(), grid_data["grid_x"].max(),
        grid_data["grid_y"].min(), grid_data["grid_y"].max()
    ], origin='lower', cmap=cmap, alpha=0.5, aspect='equal')

    if X is not None and y is not None:
        colors = ['#1565C0', '#D84315']
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=15,
                       edgecolors='white', linewidth=0.5, zorder=5)

    ax.set_title(title)
    return ax


def plot_training_comparison(stats_standard, stats_adversarial, figure_dir="figures"):
    """
    Generate the main comparison plots for the paper:
    1. Tessellation statistics over training epochs
    2. Side-by-side decision regions at key epochs
    """
    os.makedirs(figure_dir, exist_ok=True)
    epochs_s = [s["epoch"] for s in stats_standard]
    epochs_a = [s["epoch"] for s in stats_adversarial]

    # ---- Figure 1: Metrics over training ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Number of linear regions
    ax = axes[0, 0]
    ax.plot(epochs_s, [s["num_regions_grid"] for s in stats_standard],
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, [s["num_regions_grid"] for s in stats_adversarial],
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("# Linear Regions (grid)")
    ax.set_title("(a) Region Count")
    ax.legend()

    # (b) Local region density
    ax = axes[0, 1]
    ax.plot(epochs_s, [s.get("local_region_count_mean", 0) for s in stats_standard],
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, [s.get("local_region_count_mean", 0) for s in stats_adversarial],
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Local Region Count")
    ax.set_title("(b) Local Complexity Near Data")
    ax.legend()

    # (c) Mean distance to decision boundary
    ax = axes[1, 0]
    ax.plot(epochs_s, [s.get("boundary_dist_mean", 0) for s in stats_standard],
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, [s.get("boundary_dist_mean", 0) for s in stats_adversarial],
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Boundary Distance")
    ax.set_title("(c) Data-to-Boundary Distance")
    ax.legend()

    # (d) Decision boundary density
    ax = axes[1, 1]
    ax.plot(epochs_s, [s.get("boundary_density", 0) for s in stats_standard],
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, [s.get("boundary_density", 0) for s in stats_adversarial],
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Boundary Density")
    ax.set_title("(d) Decision Boundary Density")
    ax.legend()

    fig.suptitle("Tessellation Statistics: Standard vs. Adversarial Training",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "metrics_comparison.pdf"))
    plt.savefig(os.path.join(figure_dir, "metrics_comparison.png"))
    plt.close()
    print(f"Saved metrics_comparison to {figure_dir}/")


def plot_epoch_snapshots(grid_data_list, X, y, epochs, run_name="standard",
                          figure_dir="figures"):
    """
    Plot decision regions and tessellation at selected epochs.
    Creates a 2-row figure: top row = decision regions, bottom = tessellation.
    """
    os.makedirs(figure_dir, exist_ok=True)
    n = len(epochs)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (gd, ep) in enumerate(zip(grid_data_list, epochs)):
        plot_decision_regions(gd, X, y, title=f"Epoch {ep}", ax=axes[0, i])
        plot_tessellation_regions(gd, X, y, title=f"Epoch {ep}", ax=axes[1, i])

    axes[0, 0].set_ylabel("Decision Regions")
    axes[1, 0].set_ylabel("Linear Regions")
    fig.suptitle(f"Tessellation Evolution ({run_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"snapshots_{run_name}.pdf"))
    plt.savefig(os.path.join(figure_dir, f"snapshots_{run_name}.png"))
    plt.close()
    print(f"Saved snapshots_{run_name} to {figure_dir}/")


def plot_boundary_distance_histograms(stats_standard, stats_adversarial,
                                       epoch_idx=-1, figure_dir="figures"):
    """
    Compare distributions of data-to-boundary distances at a given epoch.
    """
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    dists_s = stats_standard[epoch_idx].get("boundary_distances", [])
    dists_a = stats_adversarial[epoch_idx].get("boundary_distances", [])
    ep = stats_standard[epoch_idx]["epoch"]

    if len(dists_s) > 0:
        ax.hist(dists_s, bins=30, alpha=0.5, label="Standard",
                color='#2196F3', density=True)
    if len(dists_a) > 0:
        ax.hist(dists_a, bins=30, alpha=0.5, label="Adversarial",
                color='#FF5722', density=True)

    ax.set_xlabel("Distance to Decision Boundary")
    ax.set_ylabel("Density")
    ax.set_title(f"Boundary Distance Distribution (Epoch {ep})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"boundary_dist_hist_ep{ep}.pdf"))
    plt.savefig(os.path.join(figure_dir, f"boundary_dist_hist_ep{ep}.png"))
    plt.close()
