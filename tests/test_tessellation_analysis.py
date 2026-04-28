"""
Tests for the shattering metrics in src/tessellation_analysis.py.

Run with:
    python tests/test_tessellation_analysis.py

No pytest dependency: each test_* function uses plain asserts and is
called from __main__. If a test fails, the AssertionError surfaces with
a useful message.
"""

import os
import sys

import numpy as np

# Allow `python tests/test_tessellation_analysis.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tessellation_analysis import (
    compute_per_tile_point_counts_grid,
    compute_per_tile_point_counts_polygons,
    compute_shattering_stats,
)


# ---------------------------------------------------------------------------
# Polygon-based counter: hand-built unit square partitioned into 4 quadrants.
# ---------------------------------------------------------------------------

def _unit_square_quadrants():
    """Return 4 quadrant polygons (counter-clockwise) covering [0, 1]^2."""
    return [
        np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]),  # 0: BL
        np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5]]),  # 1: BR
        np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]),  # 2: TR
        np.array([[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]]),  # 3: TL
    ]


def test_polygons_basic_quadrant_assignment():
    """Each quadrant gets its expected points; no orphans inside the unit square."""
    regions = _unit_square_quadrants()
    # 1 in BL, 2 in BR, 3 in TR, 4 in TL  -> 10 total
    X = np.array([
        [0.25, 0.25],                                     # BL
        [0.75, 0.10], [0.60, 0.40],                       # BR (2)
        [0.70, 0.70], [0.80, 0.60], [0.90, 0.90],         # TR (3)
        [0.10, 0.60], [0.20, 0.70], [0.30, 0.80], [0.05, 0.95],  # TL (4)
    ])
    counts, n_orphans = compute_per_tile_point_counts_polygons(regions, X)
    assert list(counts) == [1, 2, 3, 4], f"got {counts}"
    assert n_orphans == 0, f"unexpected orphans: {n_orphans}"
    assert counts.sum() == len(X)


def test_polygons_orphan_outside_domain():
    """Points outside every polygon become orphans, not silently dropped."""
    regions = _unit_square_quadrants()
    X = np.array([
        [0.25, 0.25],     # BL
        [5.0, 5.0],       # orphan (way outside)
        [-0.1, 0.5],      # orphan (just outside left edge)
    ])
    counts, n_orphans = compute_per_tile_point_counts_polygons(regions, X)
    assert counts.sum() == 1, f"only the BL point should land: counts={counts}"
    assert n_orphans == 2, f"expected 2 orphans, got {n_orphans}"


def test_polygons_dedupe_on_shared_edge():
    """A point that lies on a shared edge gets assigned to exactly one tile,
    not double-counted across both adjacent polygons."""
    regions = _unit_square_quadrants()
    # The point (0.5, 0.25) sits on the BL/BR boundary. With float
    # arithmetic Path.contains_points may report it as inside both. Our
    # assignment loop must dedupe so the total count equals N.
    X = np.array([[0.5, 0.25]])
    counts, n_orphans = compute_per_tile_point_counts_polygons(regions, X)
    assert counts.sum() == 1, (
        f"shared-edge point counted {counts.sum()} times (expected 1): "
        f"counts={counts}"
    )
    assert n_orphans + counts.sum() == len(X)


# ---------------------------------------------------------------------------
# Grid-based counter: synthetic region_ids_grid with known structure.
# ---------------------------------------------------------------------------

def test_grid_basic_assignment():
    """4-region grid: left half is region 0, right half is region 1, with
    a small region 2 patch and region 3 patch in the corners."""
    R = 21  # odd so there's a clear midpoint
    grid = np.zeros((R, R), dtype=int)
    grid[:, R // 2:] = 1                                  # right half
    grid[:3, :3] = 2                                      # bottom-left 3x3
    grid[-3:, -3:] = 3                                    # top-right 3x3
    domain_range = (-1.0, 1.0)
    # Pick points whose nearest pixel is unambiguously in each region.
    X = np.array([
        [-0.5, 0.0],   # region 0 (left half, away from corner)
        [0.5, 0.0],    # region 1 (right half, away from corner)
        [-0.95, -0.95],  # region 2 (bottom-left corner patch)
        [-0.85, -0.85],  # region 2 again
        [0.95, 0.95],  # region 3 (top-right corner patch)
    ])
    counts, n_orphans = compute_per_tile_point_counts_grid(grid, X, domain_range)
    assert n_orphans == 0
    assert counts[0] == 1, f"region 0 expected 1 point, got {counts[0]}"
    assert counts[1] == 1, f"region 1 expected 1 point, got {counts[1]}"
    assert counts[2] == 2, f"region 2 expected 2 points, got {counts[2]}"
    assert counts[3] == 1, f"region 3 expected 1 point, got {counts[3]}"
    assert counts.sum() == len(X)


def test_grid_orphan_outside_domain():
    """Points outside [lo, hi]^2 should be flagged as orphans."""
    R = 11
    grid = np.zeros((R, R), dtype=int)
    grid[:, R // 2:] = 1
    domain_range = (-1.0, 1.0)
    X = np.array([
        [0.0, 0.0],   # in-bounds, region 0
        [2.0, 0.0],   # out-of-bounds (right of domain)
        [0.0, -3.0],  # out-of-bounds (below domain)
    ])
    counts, n_orphans = compute_per_tile_point_counts_grid(grid, X, domain_range)
    assert n_orphans == 2, f"expected 2 orphans, got {n_orphans}"
    assert counts.sum() == 1, f"only one point should land in any tile"


# ---------------------------------------------------------------------------
# Shattering stats: definitions and edge cases.
# ---------------------------------------------------------------------------

def test_shattering_stats_basic():
    """Hand-computed counts and fractions from a known point_counts array."""
    # 10 tiles: 4 empty, 3 single, 2 with 2pts, 1 with 5pts
    # placed points = 3*1 + 2*2 + 1*5 = 12
    # nonempty tiles = 6, sum of nonempty counts = 12, mean = 2.0
    # shattering_fraction = 3 / 12 = 0.25
    counts = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 5])
    s = compute_shattering_stats(counts, n_orphans=0)
    assert s["n_empty_tiles"] == 4
    assert s["n_single_tiles"] == 3
    assert s["n_multi_tiles"] == 3  # tiles with 2+ points: three of them (2, 2, 5)
    assert s["mean_points_per_nonempty_tile"] == 2.0
    assert abs(s["shattering_fraction"] - 0.25) < 1e-12
    assert s["n_orphans"] == 0
    assert (s["point_counts"] == counts).all()


def test_shattering_stats_excludes_orphans_from_denominator():
    """Per the spec: shattering_fraction's denominator is points-in-tiles, not n_train."""
    # 5 placed points: all in single-point tiles -> shattering_fraction = 1.0
    # 3 orphans should not enter the denominator.
    counts = np.array([1, 1, 1, 1, 1])
    s = compute_shattering_stats(counts, n_orphans=3)
    assert abs(s["shattering_fraction"] - 1.0) < 1e-12
    assert s["n_orphans"] == 3


def test_shattering_stats_all_orphans():
    """If no point lands in any tile, fraction is undefined -> 0.0, not NaN."""
    counts = np.zeros(10, dtype=int)
    s = compute_shattering_stats(counts, n_orphans=7)
    assert s["shattering_fraction"] == 0.0
    assert s["mean_points_per_nonempty_tile"] == 0.0
    assert s["n_empty_tiles"] == 10
    assert s["n_orphans"] == 7


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ok     {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL   {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"  ERROR  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
