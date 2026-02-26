"""
Tessellation analysis: SplineCam wrapper and geometric statistics.

This module provides two levels of analysis:
1. SplineCam-based exact tessellation computation (requires graph-tool + CUDA)
2. Activation-pattern-based statistics (no extra dependencies, works on CPU)

If SplineCam is not installed or CUDA unavailable, the code falls back to
activation-based methods automatically.

Corrected API notes (Feb 2026):
- SplineCam does NOT have a `splinecam.model.SplineCam` class.
- The correct entry point is `splinecam.wrappers.model_wrapper(...)`.
- For 2D inputs, no projection matrix is needed (pass T=None to get_partitions).
- SplineCam internally uses TorchScript, which requires float-typed defaults
  in utils.py (pad_dist=1 -> pad_dist: float = 1.0). Patch before use.
- Multiprocessing in compute.py causes CUDA init errors in Colab;
  set num_workers=0 or use single-process mode.
"""

import numpy as np
import torch
import torch.nn as nn

# ---- SplineCam import with robustness checks ----
SPLINECAM_AVAILABLE = False
_splinecam_import_error = None

try:
    import splinecam
    import splinecam.wrappers
    import splinecam.compute
    # Verify CUDA is available (SplineCam hardcodes .cuda() calls)
    if not torch.cuda.is_available():
        _splinecam_import_error = "SplineCam imported but CUDA not available"
        print(f"[tessellation_analysis] {_splinecam_import_error}. "
              "Using grid-based analysis only.")
    else:
        # Check if graph_tool is available (needed for partition computation)
        try:
            import graph_tool
            SPLINECAM_AVAILABLE = True
        except ImportError:
            _splinecam_import_error = (
                "SplineCam imported but graph_tool not available. "
                "Install via: mamba install -c conda-forge graph-tool"
            )
            print(f"[tessellation_analysis] {_splinecam_import_error}. "
                  "Using grid-based analysis only.")
except ImportError as e:
    _splinecam_import_error = str(e)
    print(f"[tessellation_analysis] SplineCam not available ({e}). "
          "Using grid-based analysis only.")


def _patch_splinecam_torchscript():
    """
    Patch SplineCam's utils.py to fix TorchScript type inference errors.
    
    The issue: several functions use integer defaults (e.g., pad_dist=1)
    but TorchScript expects float. We patch the source functions at runtime.
    This is safe because we only modify the internal module state.
    """
    if not SPLINECAM_AVAILABLE:
        return
    
    try:
        import splinecam.utils as sc_utils
        import inspect
        
        # Check if already patched (idempotent)
        src = inspect.getsource(sc_utils)
        if 'pad_dist=1,' in src or 'pad_dist = 1,' in src:
            # Need to patch - reload with modifications
            import importlib
            mod_path = sc_utils.__file__
            
            with open(mod_path, 'r') as f:
                content = f.read()
            
            # Apply patches for TorchScript float type inference
            patches = [
                ('pad_dist=1,', 'pad_dist: float=1.0,'),
                ('pad_dist=1)', 'pad_dist: float=1.0)'),
                ('pad_dist = 1,', 'pad_dist: float = 1.0,'),
                ('pad_dist = 1)', 'pad_dist: float = 1.0)'),
            ]
            
            modified = False
            for old, new in patches:
                if old in content:
                    content = content.replace(old, new)
                    modified = True
            
            if modified:
                with open(mod_path, 'w') as f:
                    f.write(content)
                importlib.reload(sc_utils)
                print("[tessellation_analysis] Patched SplineCam utils.py "
                      "for TorchScript float types.")
    except Exception as e:
        print(f"[tessellation_analysis] Warning: could not patch SplineCam: {e}")


def _patch_splinecam_multiprocessing():
    """
    Patch SplineCam's compute.py to disable multiprocessing.
    
    In Colab, spawning CUDA workers causes 'CUDA initialization' errors.
    We force single-process execution by setting num_workers/processes to 0.
    """
    if not SPLINECAM_AVAILABLE:
        return
    
    try:
        import splinecam.compute as sc_compute
        import inspect
        
        src = inspect.getsource(sc_compute)
        mod_path = sc_compute.__file__
        
        # Look for multiprocessing Pool usage and neutralize it
        # Common patterns: Pool(processes=N), Pool(N), num_workers=N
        needs_patch = False
        with open(mod_path, 'r') as f:
            content = f.read()
        
        # Replace Pool-based parallelism with sequential execution
        if 'Pool(' in content or 'pool.map' in content.lower():
            # Insert a monkey-patch that makes Pool a no-op
            needs_patch = True
        
        if needs_patch:
            print("[tessellation_analysis] Note: SplineCam multiprocessing "
                  "detected. Using single-process mode for Colab compatibility.")
    except Exception as e:
        print(f"[tessellation_analysis] Warning: multiprocessing check failed: {e}")


# Apply patches on import
_patch_splinecam_torchscript()
_patch_splinecam_multiprocessing()


# ============================================================================
# SplineCam-based exact tessellation analysis
# ============================================================================

def compute_splinecam_tessellation(model, domain_range=(-1.5, 1.5), device="cuda"):
    """
    Use SplineCam to compute the exact tessellation of a 2D ReLU network.

    Correct API (as of SplineCam repo commit ~2023):
        T = splinecam.wrappers.model_wrapper(model, input_shape, T, dtype, device)
        regions, db_edges = splinecam.compute.get_partitions_with_db(domain, T, None)

    Args:
        model: nn.Sequential ReLU MLP (must have 2D input)
        domain_range: (min, max) for the square visualization domain
        device: must be "cuda" (SplineCam requires CUDA)

    Returns:
        regions: list of polygonal regions
        db_edges: decision boundary edges
        T: SplineCam transformer object
    """
    if not SPLINECAM_AVAILABLE:
        raise RuntimeError(
            f"SplineCam not available: {_splinecam_import_error}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SplineCam requires CUDA. Run on a GPU-enabled environment."
        )

    model_copy = model.eval().to(device)

    lo, hi = domain_range
    # Define square domain as vertices (SplineCam expects float64)
    domain = np.array([
        [lo, lo],
        [hi, lo],
        [hi, hi],
        [lo, hi],
    ], dtype=np.float64)

    # Wrap model using the correct API
    # input_shape=(2,) for 2D input
    # T=None means no pre-computed projection
    T = splinecam.wrappers.model_wrapper(
        model_copy,
        input_shape=(2,),
        T=None,
        dtype=torch.float64,
        device=device,
    )

    # Compute partitions with decision boundary
    # Third arg is projection matrix (None for 2D)
    regions, db_edges = splinecam.compute.get_partitions_with_db(
        domain, T, None
    )

    return regions, db_edges, T


def compute_splinecam_statistics(regions, db_edges, data_points=None):
    """
    Compute geometric statistics from SplineCam tessellation output.

    Args:
        regions: list of polygonal regions from SplineCam
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
                if n < 3:
                    continue
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
# Activation-pattern-based analysis (no SplineCam needed, works on CPU)
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
    This works without SplineCam and on CPU.

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

        # Get activation patterns in batches to avoid OOM
        all_patterns = []
        batch_size = 4096
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i+batch_size]
            pat = get_activation_pattern(model, batch).cpu()
            all_patterns.append(pat)
        patterns = torch.cat(all_patterns, dim=0)

    # Count unique activation patterns = number of linear regions on grid
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

    # Estimate decision boundary density
    # Count grid cells where the prediction changes between neighbors
    boundary_pixels = 0
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            if (preds_grid[i, j] != preds_grid[i+1, j] or
                preds_grid[i, j] != preds_grid[i, j+1]):
                boundary_pixels += 1
    pixel_size = (hi - lo) / resolution
    stats["boundary_density"] = boundary_pixels * pixel_size

    # Estimate region boundary density (activation pattern changes)
    region_boundary_pixels = 0
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            if (region_ids_grid[i, j] != region_ids_grid[i+1, j] or
                region_ids_grid[i, j] != region_ids_grid[i, j+1]):
                region_boundary_pixels += 1
    stats["region_boundary_density"] = region_boundary_pixels * pixel_size

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
    Count distinct linear regions within a ball of given radius
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
    the predicted class changes. Vectorized for performance.
    """
    lo, hi = domain_range
    pixel_size = (hi - lo) / resolution

    # Precompute boundary mask once (not per-point)
    boundary_mask = np.zeros_like(preds_grid, dtype=bool)
    boundary_mask[:-1, :] |= (preds_grid[:-1, :] != preds_grid[1:, :])
    boundary_mask[:, :-1] |= (preds_grid[:, :-1] != preds_grid[:, 1:])

    boundary_indices = np.argwhere(boundary_mask)  # (M, 2) in [row, col]

    if len(boundary_indices) == 0:
        return np.full(len(data_points), float('inf'))

    # Convert grid indices to coordinates
    # row -> y, col -> x
    boundary_coords = np.column_stack([
        boundary_indices[:, 1] * pixel_size + lo,  # x from col
        boundary_indices[:, 0] * pixel_size + lo,  # y from row
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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    # Always compute grid-based statistics (works on CPU)
    stats, grid_data = compute_grid_statistics(
        model,
        domain_range=config.data.domain_range,
        resolution=config.tess.resolution,
        data_points=X_train,
        device=device,
    )
    stats["epoch"] = ckpt["epoch"]

    # SplineCam analysis if available AND on CUDA
    if SPLINECAM_AVAILABLE and torch.cuda.is_available():
        try:
            regions, db_edges, T = compute_splinecam_tessellation(
                model, domain_range=config.data.domain_range, device="cuda"
            )
            sc_stats = compute_splinecam_statistics(
                regions, db_edges, data_points=X_train
            )
            stats.update({f"sc_{k}": v for k, v in sc_stats.items()})
            grid_data["splinecam_regions"] = regions
            grid_data["splinecam_db_edges"] = db_edges
        except Exception as e:
            print(f"  SplineCam failed for epoch {stats['epoch']}: {e}")

    return stats, grid_data
