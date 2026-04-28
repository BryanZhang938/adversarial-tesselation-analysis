"""
Visualization functions for tessellation dynamics paper.
Generates publication-quality figures comparing standard vs adversarial training.
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

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


# ============================================================================
# Density-colored tessellation snapshots and shattering comparison
# ============================================================================

# Color used for empty (zero-point) tiles. Distinct from the viridis range so
# empty tiles read as "no data" rather than "lowest density".
_EMPTY_TILE_COLOR = "#dddddd"


def _density_cmap():
    """Viridis with `set_bad` set to the empty-tile gray."""
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color=_EMPTY_TILE_COLOR)
    return cmap


def plot_density_tessellation(grid_data, point_counts, X=None, y=None,
                              title="", ax=None, vmax=None, cmap=None):
    """
    Plot the tessellation colored by per-tile training-point count.

    If grid_data has SplineCam polygons, draw each one as a filled patch
    (vector). Otherwise, build a per-pixel density raster from the
    region-id grid and use imshow. Tiles with 0 points are colored with
    `_EMPTY_TILE_COLOR` so they read as "empty" rather than "least dense".

    Args:
        grid_data: dict from analyze_checkpoint. Must contain either
            "splinecam_regions" (preferred) or "region_ids_grid".
        point_counts: 1-D int array, one entry per tile, indexed in the
            same order as the tile/region list.
        X, y: optional training points to overlay.
        title: subplot title.
        ax: matplotlib Axes (one is created if None).
        vmax: upper end of the colormap. If None, defaults to
            max(1, point_counts.max()). Pass an explicit value to share
            normalization across multiple subplots.
        cmap: optional Colormap. Defaults to `_density_cmap()`.

    Returns:
        (ax, mappable) where `mappable` can be passed to fig.colorbar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    if cmap is None:
        cmap = _density_cmap()
    point_counts = np.asarray(point_counts)
    if vmax is None:
        vmax = max(1, int(point_counts.max()) if point_counts.size else 1)
    norm = mcolors.Normalize(vmin=1, vmax=vmax)

    if "splinecam_regions" in grid_data:
        # Vector path: each polygon as a filled patch.
        from src.tessellation_analysis import _region_to_vertices
        patches = []
        face_colors = []
        for region, count in zip(grid_data["splinecam_regions"], point_counts):
            verts = _region_to_vertices(region)
            if verts.shape[0] < 3:
                continue
            patches.append(MplPolygon(verts, closed=True))
            if count == 0:
                face_colors.append(_EMPTY_TILE_COLOR)
            else:
                face_colors.append(cmap(norm(count)))
        coll = PatchCollection(patches, facecolors=face_colors,
                               edgecolors="black", linewidths=0.2)
        ax.add_collection(coll)
        if "grid_x" in grid_data:
            gx = grid_data["grid_x"]
            gy = grid_data["grid_y"]
            ax.set_xlim(gx.min(), gx.max())
            ax.set_ylim(gy.min(), gy.max())
        # ScalarMappable used as the colorbar handle (the PatchCollection's
        # facecolors are baked in, so we need a separate mappable).
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        # Raster path: lookup table from region id -> count.
        region_ids = np.asarray(grid_data["region_ids_grid"])
        # point_counts is indexed by region id (dense in [0, n_tiles)).
        # Out-of-range region ids should never happen, but guard anyway.
        n = max(int(region_ids.max()) + 1, point_counts.size)
        lut = np.zeros(n, dtype=float)
        lut[: point_counts.size] = point_counts
        density = lut[region_ids]
        # Mask zero-count pixels so cmap.set_bad colors them gray.
        density = np.where(density == 0, np.nan, density)
        gx = grid_data["grid_x"]
        gy = grid_data["grid_y"]
        mappable = ax.imshow(
            density,
            extent=[gx.min(), gx.max(), gy.min(), gy.max()],
            origin="lower", cmap=cmap, norm=norm, aspect="equal",
        )

    # Overlay training points (small markers, colored by class).
    if X is not None and y is not None:
        colors = ["#1565C0", "#D84315"]
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], s=10,
                       edgecolors="white", linewidth=0.4, zorder=5)

    ax.set_title(title)
    ax.set_aspect("equal")
    return ax, mappable


def plot_density_snapshots(grid_data_list, point_counts_list, X, y, epochs,
                           run_name="standard", figure_dir="figures"):
    """
    One-row figure of density-colored tessellations across training.

    Mirrors `plot_epoch_snapshots` (same epoch list passed in by the caller),
    but each panel is colored by per-tile training-point count instead of a
    random region color. All panels share a single colormap normalization
    (vmax = global max across the panels) and a single colorbar.

    Args:
        grid_data_list: list of grid_data dicts (one per checkpoint).
        point_counts_list: list of per-tile count arrays (one per checkpoint),
            aligned with grid_data_list.
        X, y: training points (overlaid on each panel).
        epochs: list of epoch indices for panel titles.
        run_name: "standard" or "adversarial".
        figure_dir: where to write `density_snapshots_{run_name}.{png,pdf}`.
    """
    os.makedirs(figure_dir, exist_ok=True)
    n = len(epochs)
    if n == 0:
        return

    # Shared upper bound across panels in this figure (per spec).
    vmax = 1
    for pc in point_counts_list:
        if len(pc) > 0:
            vmax = max(vmax, int(np.asarray(pc).max()))

    # Width per panel matches plot_epoch_snapshots (4 inches), plus a slim
    # extra column for the shared colorbar so the panels stay square.
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 1.0, 4.5),
                             squeeze=False)
    cmap = _density_cmap()
    last_mappable = None
    for i, (gd, pc, ep) in enumerate(zip(grid_data_list, point_counts_list, epochs)):
        ax = axes[0, i]
        _, mappable = plot_density_tessellation(
            gd, pc, X=X, y=y, title=f"Epoch {ep}", ax=ax,
            vmax=vmax, cmap=cmap,
        )
        last_mappable = mappable

    # Shared colorbar to the right of the figure.
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(last_mappable, cax=cbar_ax)
    cbar.set_label("Training points per tile")

    fig.suptitle(f"Density-Colored Tessellation Evolution ({run_name})",
                 fontsize=14, y=1.02)
    out_png = os.path.join(figure_dir, f"density_snapshots_{run_name}.png")
    out_pdf = os.path.join(figure_dir, f"density_snapshots_{run_name}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved density_snapshots_{run_name} to {figure_dir}/")


def plot_shattering_comparison(stats_standard, stats_adversarial,
                               figure_dir="figures"):
    """
    Four-panel figure comparing shattering metrics between ERM and PGD-AT.

    Mirrors `plot_training_comparison`:
        (a) Fraction of empty tiles vs epoch
        (b) Fraction of single-point tiles vs epoch
        (c) Mean points per non-empty tile vs epoch
        (d) Histogram of final-epoch per-tile point counts (both runs, log-y)

    The histogram pulls `point_counts` from the in-memory final-epoch stats
    dict (run_experiment.py passes the live stats objects to plotters before
    JSON serialization strips ndarrays). If `point_counts` is missing for
    some reason the histogram panel renders empty rather than crashing.
    """
    os.makedirs(figure_dir, exist_ok=True)

    epochs_s = [s["epoch"] for s in stats_standard]
    epochs_a = [s["epoch"] for s in stats_adversarial]

    def _frac(stats_list, key):
        # Defensive: tolerate missing tiles count (e.g. older results dict).
        return [
            (s.get(key, 0) / s["n_tiles"]) if s.get("n_tiles", 0) > 0 else 0.0
            for s in stats_list
        ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Fraction of empty tiles
    ax = axes[0, 0]
    ax.plot(epochs_s, _frac(stats_standard, "n_empty_tiles"),
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, _frac(stats_adversarial, "n_empty_tiles"),
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of Empty Tiles")
    ax.set_title("(a) Empty-Tile Fraction")
    ax.set_ylim(0, 1)
    ax.legend()

    # (b) Fraction of single-point tiles
    ax = axes[0, 1]
    ax.plot(epochs_s, _frac(stats_standard, "n_single_tiles"),
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, _frac(stats_adversarial, "n_single_tiles"),
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of Single-Point Tiles")
    ax.set_title("(b) Shattered-Tile Fraction")
    ax.set_ylim(0, 1)
    ax.legend()

    # (c) Mean points per non-empty tile
    ax = axes[1, 0]
    ax.plot(epochs_s, [s.get("mean_points_per_nonempty_tile", 0)
                       for s in stats_standard],
            'o-', label="Standard", color='#2196F3')
    ax.plot(epochs_a, [s.get("mean_points_per_nonempty_tile", 0)
                       for s in stats_adversarial],
            's-', label="Adversarial", color='#FF5722')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Points / Non-Empty Tile")
    ax.set_title("(c) Mean Density of Occupied Tiles")
    ax.legend()

    # (d) Histogram of final-epoch per-tile point counts (log y).
    ax = axes[1, 1]
    pc_s = np.asarray(stats_standard[-1].get("point_counts", []))
    pc_a = np.asarray(stats_adversarial[-1].get("point_counts", []))
    final_ep = stats_standard[-1].get("epoch", "?")
    if pc_s.size or pc_a.size:
        max_count = int(max(pc_s.max() if pc_s.size else 0,
                            pc_a.max() if pc_a.size else 0))
        bins = np.arange(0, max_count + 2) - 0.5  # integer-centered bins
        if pc_s.size:
            ax.hist(pc_s, bins=bins, alpha=0.5, label="Standard",
                    color='#2196F3')
        if pc_a.size:
            ax.hist(pc_a, bins=bins, alpha=0.5, label="Adversarial",
                    color='#FF5722')
        ax.set_yscale("log")
        ax.set_xlabel("Points per Tile")
        ax.set_ylabel("Tile Count (log)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no point_counts in final stats",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"(d) Per-Tile Count Histogram (Epoch {final_ep})")

    fig.suptitle("Neural Shattering: Standard vs. Adversarial Training",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "shattering_comparison.pdf"))
    plt.savefig(os.path.join(figure_dir, "shattering_comparison.png"))
    plt.close()
    print(f"Saved shattering_comparison to {figure_dir}/")
