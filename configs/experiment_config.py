"""
Experiment configuration for tessellation dynamics study.
All hyperparameters in one place for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DataConfig:
    """Synthetic dataset parameters."""
    dataset: str = "concentric_rings"  # "spirals", "concentric_rings", "moons"
    n_samples: int = 500
    noise: float = 0.05          # reduced noise for rings (was 0.1 for spirals)
    seed: int = 42
    # Concentric rings geometry: gap must be > 2*epsilon post-normalization
    # With inner=0.3, outer=1.0, noise=0.05: worst-case gap ≈ 0.30 >> 2*0.1 = 0.20
    inner_radius: float = 0.3
    outer_radius: float = 1.0
    # Domain for tessellation visualization
    domain_range: Tuple[float, float] = (-1.5, 1.5)

@dataclass
class ModelConfig:
    """Network architecture parameters."""
    input_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [50])  # 1 hidden layer, width 50
    output_dim: int = 2  # binary classification
    activation: str = "relu"

@dataclass
class TrainConfig:
    """Training parameters."""
    epochs: int = 500          # T=500 for spirals (override to 300 for rings)
    batch_size: int = 64
    lr: float = 0.01           # SGD learning rate
    weight_decay: float = 0.0  # no weight decay per paper
    optimizer: str = "sgd"     # SGD per paper
    momentum: float = 0.9      # SGD momentum per paper
    scheduler: str = "none"    # no scheduler per paper

@dataclass
class AdversarialConfig:
    """PGD adversarial training parameters."""
    enabled: bool = False
    epsilon: float = 0.03      # perturbation radius (l2); small enough for spirals
    step_size: float = 0.0075  # alpha = epsilon / 4
    num_steps: int = 7         # K=7 PGD steps per paper
    norm: str = "l2"           # l2 perturbation per paper

@dataclass
class TessellationConfig:
    """SplineCam tessellation analysis parameters."""
    num_checkpoints: int = 50  # M=50 evenly spaced checkpoints
    checkpoint_epochs: List[int] = field(default_factory=list)  # computed from epochs
    # 2D domain vertices for SplineCam (square region)
    domain_vertices: int = 4  # corners of a square
    resolution: int = 200     # grid resolution for boundary distance computation
    # Local complexity parameters
    local_complexity_radius: float = 0.1  # r=0.1 per paper
    # Metrics to compute
    compute_region_count: bool = True
    compute_cell_areas: bool = True
    compute_boundary_distance: bool = True
    compute_slope_norms: bool = True
    # Toggle SplineCam exact-tessellation analysis. The grid-based path
    # (region count, local complexity, boundary density, shattering
    # metrics) runs unconditionally; SplineCam adds vector polygons and
    # `sc_*` stats. SplineCam has Colab-specific reliability issues —
    # set this to False to skip it deterministically and cut noise.
    use_splinecam: bool = True

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    adv: AdversarialConfig = field(default_factory=AdversarialConfig)
    tess: TessellationConfig = field(default_factory=TessellationConfig)
    # Output paths
    checkpoint_dir: str = "checkpoints"
    figure_dir: str = "figures"
    results_dir: str = "results"
    device: str = "cpu"
    seed: int = 42


def _compute_checkpoint_epochs(total_epochs, num_checkpoints=50):
    """Compute M evenly spaced checkpoint epochs."""
    if num_checkpoints >= total_epochs:
        return list(range(1, total_epochs + 1))
    step = total_epochs / num_checkpoints
    return sorted(set(max(1, round(step * i)) for i in range(1, num_checkpoints + 1)))


def get_standard_config(dataset="spirals"):
    """Config for standard (ERM) training."""
    cfg = ExperimentConfig()
    cfg.adv.enabled = False
    if dataset == "concentric_rings":
        cfg.train.epochs = 300
    cfg.tess.checkpoint_epochs = _compute_checkpoint_epochs(
        cfg.train.epochs, cfg.tess.num_checkpoints
    )
    return cfg


def get_adversarial_config(dataset="spirals"):
    """Config for PGD adversarial training."""
    cfg = ExperimentConfig()
    cfg.adv.enabled = True
    if dataset == "concentric_rings":
        cfg.train.epochs = 300
    cfg.tess.checkpoint_epochs = _compute_checkpoint_epochs(
        cfg.train.epochs, cfg.tess.num_checkpoints
    )
    return cfg
