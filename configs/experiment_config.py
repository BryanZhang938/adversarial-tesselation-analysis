"""
Experiment configuration for tessellation dynamics study.
All hyperparameters in one place for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DataConfig:
    """Synthetic dataset parameters."""
    dataset: str = "spirals"  # "spirals", "concentric_rings", "moons"
    n_samples: int = 500
    noise: float = 0.1
    seed: int = 42
    # Domain for tessellation visualization
    domain_range: Tuple[float, float] = (-1.5, 1.5)

@dataclass
class ModelConfig:
    """Network architecture parameters."""
    input_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    output_dim: int = 2  # binary classification
    activation: str = "relu"

@dataclass
class TrainConfig:
    """Training parameters."""
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"  # "cosine", "step", "none"

@dataclass
class AdversarialConfig:
    """PGD adversarial training parameters."""
    enabled: bool = False
    epsilon: float = 0.1       # perturbation radius (l_inf)
    step_size: float = 0.025   # PGD step size
    num_steps: int = 10        # PGD iterations
    norm: str = "linf"         # "linf" or "l2"

@dataclass
class TessellationConfig:
    """SplineCam tessellation analysis parameters."""
    checkpoint_epochs: List[int] = field(default_factory=lambda: [
        1, 5, 10, 20, 50, 100, 150, 200
    ])
    # 2D domain vertices for SplineCam (square region)
    domain_vertices: int = 4  # corners of a square
    resolution: int = 200     # grid resolution for boundary distance computation
    # Metrics to compute
    compute_region_count: bool = True
    compute_cell_areas: bool = True
    compute_boundary_distance: bool = True
    compute_slope_norms: bool = True

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


def get_standard_config():
    """Config for standard (ERM) training."""
    cfg = ExperimentConfig()
    cfg.adv.enabled = False
    return cfg


def get_adversarial_config():
    """Config for PGD adversarial training."""
    cfg = ExperimentConfig()
    cfg.adv.enabled = True
    return cfg
