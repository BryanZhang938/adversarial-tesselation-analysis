"""
Tessellation analysis: SplineCam wrapper and geometric statistics.

This module provides two levels of analysis:
1. SplineCam-based exact tessellation computation (requires graph-tool)
2. Activation-pattern-based statistics (no extra dependencies)

If SplineCam is not installed, the code falls back to activation-based methods.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import splinecam
    import splinecam.compute
    import splinecam.plot
    SPLINECAM_AVAILABLE = True
except ImportError:
    SPLINECAM_AVAILABLE = False
    print("SplineCam not available. Using activation-pattern-based analysis only.")


# ============================================================================
# SplineCam-based exact tessellation analysis
# ============================================================================

def compute_splinecam_tessellation(model, domain_range=(-1.5, 1.5), device="cpu"):
    """
    Use SplineCam to compute the exact tessellation of a 2D ReLU network.

    Args:
        model: nn.Sequential ReLU MLP (must have 2D input)
        domain_range: (min, max) for the square visualization domain

    Returns:
        regions: list of polygonal regions (from SplineCam)
        db_edges: decision boundary edges
        T: SplineCam transformer object
    """
    if not SPLINECAM_AVAILABLE:
        raise RuntimeError("SplineCam is not installed.")

    model.eval().to(device)

    lo, hi = domain_range
    # Define square domain as vertices
    domain = np.array([
        [lo, lo],
        [hi, lo],
        [hi, hi],
        [lo, hi],
    ], dtype=np.float64)

    # Wrap the model with SplineCam
    T = splinecam.model.SplineCam(model)
    NN = None  # no projection needed for 2D input

    regions, db_edges = splinecam.compute.get_partitions_with_db(domain, T, NN)

    return regions, db_edges, T


def compute_splinecam_statistics(regions, db_edges, data_points=None):
    """
    Compute geometric statistics from SplineCam tessellation output.

    Args:
        regions: list of polygonal regions
        db_edges: decision boundary edges
        data_points: optional (N, 2) array of training points

    Returns:
        stats: dict of tessellation statistics
    """
    stats = {}

    # Number of linear regions
    stats["num_regions"] = len(regions)

    # Cell area distribution
    areas = []
    for region in regions:
        if hasattr(region, 'area'):
            areas.append(region.area)
        else:
            # Compute area via shoelace formula if region is array of vertices
            try:
                verts = np.array(region)
                n = len(verts)
                area = 0.5 * abs(
                    sum(verts[i, 0] * verts[(i+1) % n, 1] -
                        verts[(i+1) % n, 0] * verts[i, 1]
                        for i in range(n))
                )
                areas.append(area)
            except Exception:
                pass

    if areas:
        areas = np.array(areas)
        stats["cell_area_mean"] = float(np.mean(areas))
        stats["cell_area_std"] = float(np.std(areas))
        stats["cell_area_min"] = float(np.min(areas))
        stats["cell_area_max"] = float(np.max(areas))
        stats["cell_area_median"] = float(np.median(areas))
        stats["cell_areas"] = areas

    # Total decision boundary length
    total_db_length = 0.0
    if db_edges is not None:
        for edge in db_edges:
            edge = np.array(edge)
            if len(edge) >= 2:
                total_db_length += np.sqrt(
                    np.sum((edge[1] - edge[0]) ** 2)
                )
    stats["total_db_length"] = total_db_length

    return stats


# ============================================================================
# Activation-pattern-based analysis (no SplineCam needed)
# ============================================================================

def get_activation_pattern(model, x):
    """Get binary activation pattern for input x."""
    patterns = []
    h = x
    for layer in model:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            patterns.append((h > 0).int())
    return torch.cat(patterns, dim=-1)


def compute_grid_statistics(model, domain_range=(-1.5, 1.5), resolution=200,
                            data_points=None, device="cpu"):
    """
    Compute tessellation statistics by evaluating the network on a dense grid.
    This works without SplineCam.

    Args:
        model: nn.Sequential ReLU MLP
        domain_range: (min, max) for grid
        resolution: grid points per dimension
        data_points: optional (N, 2) array of data points
        device: "cpu" or "cuda"

    Returns:
        stats: dict of tessellation statistics
        grid_data: dict with grid predictions, patterns, etc.
    """
    model.eval().to(device)

    lo, hi = domain_range
    xx = np.linspace(lo, hi, resolution)
    yy = np.linspace(lo, hi, resolution)
    grid_x, grid_y = np.meshgrid(xx, yy)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_tensor = torch.from_numpy(grid_points.astype(np.float32)).to(device)

    stats = {}

    with torch.no_grad():
        # Get predictions
        logits = model(grid_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()

        # Get activation patterns
        patterns = get_activation_pattern(model, grid_tensor).cpu()

    # Count unique activation patterns (= number of distinct linear regions
    # visited by the grid)
    pattern_set = set()
    for row in patterns:
        pattern_set.add(tuple(row.tolist()))
    stats["num_regions_grid"] = len(pattern_set)

    # Assign region ID to each grid point
    pattern_to_id = {}
    region_ids = np.zeros(len(grid_points), dtype=int)
    for i, row in enumerate(patterns):
        key = tuple(row.tolist())
        if key not in pattern_to_id:
            pattern_to_id[key] = len(pattern_to_id)
        region_ids[i] = pattern_to_id[key]

    region_ids_grid = region_ids.reshape(resolution, resolution)
    preds_grid = preds.reshape(resolution, resolution)

    pixel_size = (hi - lo) / resolution

    # Vectorized boundary density computation
    boundary_v = np.sum(preds_grid[:-1, :] != preds_grid[1:, :])
    boundary_h = np.sum(preds_grid[:, :-1] != preds_grid[:, 1:])
    stats["boundary_density"] = int(boundary_v + boundary_h) * pixel_size

    # Region boundary density (activation pattern changes)
    region_v = np.sum(region_ids_grid[:-1, :] != region_ids_grid[1:, :])
    region_h = np.sum(region_ids_grid[:, :-1] != region_ids_grid[:, 1:])
    stats["region_boundary_density"] = int(region_v + region_h) * pixel_size

    # Local region density near data points
    if data_points is not None:
        local_counts = compute_local_region_count(
            model, data_points, radius=0.2, n_samples=200, device=device
        )
        stats["local_region_count_mean"] = float(np.mean(local_counts))
        stats["local_region_count_std"] = float(np.std(local_counts))
        stats["local_region_counts"] = local_counts

    # Distance from data points to decision boundary
    if data_points is not None:
        distances = estimate_boundary_distances_grid(
            preds_grid, data_points, domain_range, resolution
        )
        stats["boundary_dist_mean"] = float(np.mean(distances))
        stats["boundary_dist_std"] = float(np.std(distances))
        stats["boundary_dist_min"] = float(np.min(distances))
        stats["boundary_distances"] = distances

    grid_data = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "preds_grid": preds_grid,
        "region_ids_grid": region_ids_grid,
        "grid_points": grid_points,
    }

    return stats, grid_data


def compute_local_region_count(model, data_points, radius=0.2, n_samples=200,
                                device="cpu"):
    """
    Count the number of distinct linear regions within a ball of given radius
    around each data point. This is the "local complexity" metric from
    Humayun et al. (2023).
    """
    model.eval()
    counts = []

    for pt in data_points:
        # Sample uniformly in a ball around the point
        angles = np.random.uniform(0, 2 * np.pi, n_samples)
        radii = np.sqrt(np.random.uniform(0, 1, n_samples)) * radius
        offsets = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        neighbors = pt + offsets
        neighbors_tensor = torch.from_numpy(
            neighbors.astype(np.float32)
        ).to(device)

        with torch.no_grad():
            patterns = get_activation_pattern(model, neighbors_tensor).cpu()

        unique = set(tuple(row.tolist()) for row in patterns)
        counts.append(len(unique))

    return np.array(counts)


def estimate_boundary_distances_grid(preds_grid, data_points, domain_range,
                                      resolution):
    """
    For each data point, find the distance to the nearest grid cell where
    the predicted class changes (decision boundary pixel).
    """
    lo, hi = domain_range
    pixel_size = (hi - lo) / resolution

    # Compute boundary mask once (vectorized)
    diff_vertical = preds_grid[:-1, :] != preds_grid[1:, :]
    diff_horizontal = preds_grid[:, :-1] != preds_grid[:, 1:]
    boundary_mask = np.zeros_like(preds_grid, dtype=bool)
    boundary_mask[:-1, :] |= diff_vertical
    boundary_mask[:, :-1] |= diff_horizontal

    boundary_indices = np.argwhere(boundary_mask)
    if len(boundary_indices) == 0:
        return np.full(len(data_points), float('inf'))

    # row index -> y coord, col index -> x coord
    boundary_coords = np.column_stack([
        boundary_indices[:, 1] * pixel_size + lo,
        boundary_indices[:, 0] * pixel_size + lo,
    ])

    distances = []
    for pt in data_points:
        dists = np.sqrt(np.sum((boundary_coords - pt) ** 2, axis=1))
        distances.append(float(np.min(dists)))

    return np.array(distances)


def analyze_checkpoint(model_class, checkpoint_path, X_train, config, device="cpu"):
    """
    Load a checkpoint and compute all tessellation statistics.

    Args:
        model_class: callable that returns a fresh model
        checkpoint_path: path to .pt checkpoint
        X_train: training data points (N, 2) numpy array
        config: ExperimentConfig
        device: "cpu" or "cuda"

    Returns:
        stats: dict of statistics
        grid_data: dict with grid-based data for plotting
    """
    model = model_class()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    stats, grid_data = compute_grid_statistics(
        model,
        domain_range=config.data.domain_range,
        resolution=config.tess.resolution,
        data_points=X_train,
        device=device,
    )
    stats["epoch"] = ckpt["epoch"]

    # SplineCam analysis if available
    if SPLINECAM_AVAILABLE:
        try:
            regions, db_edges, T = compute_splinecam_tessellation(
                model, domain_range=config.data.domain_range, device=device
            )
            sc_stats = compute_splinecam_statistics(
                regions, db_edges, data_points=X_train
            )
            stats.update({f"sc_{k}": v for k, v in sc_stats.items()})
            grid_data["splinecam_regions"] = regions
            grid_data["splinecam_db_edges"] = db_edges
        except Exception as e:
            print(f"SplineCam failed for epoch {stats['epoch']}: {e}")

    return stats, grid_data
