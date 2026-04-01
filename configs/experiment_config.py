"""
Experiment configuration for tessellation dynamics study.
All hyperparameters in one place for reproducibility.

Updated March 2026: added support for separate adversarial training
hyperparameters (wider networks, longer training, LR schedules) per
reviewer feedback on convergence.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


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
    hidden_dims: List[int] = field(default_factory=lambda: [50])
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
    scheduler: str = "none"    # "none", "cosine", or "step"
    scheduler_step_size: int = 50   # for StepLR
    scheduler_gamma: float = 0.5    # for StepLR
    eval_interval: int = 10    # evaluate test accuracy every N epochs


@dataclass
class AdversarialConfig:
    """PGD adversarial training parameters."""
    enabled: bool = False
    epsilon: float = 0.1       # perturbation radius (l2)
    step_size: float = 0.025   # alpha = epsilon / 4 (recomputed dynamically)
    num_steps: int = 7         # K=7 PGD steps per paper
    norm: str = "l2"           # l2 perturbation per paper


@dataclass
class TessellationConfig:
    """SplineCam tessellation analysis parameters."""
    num_checkpoints: int = 50  # M=50 evenly spaced checkpoints
    checkpoint_epochs: List[int] = field(default_factory=list)
    domain_vertices: int = 4
    resolution: int = 200      # grid resolution for boundary distance computation
    local_complexity_radius: float = 0.1  # r=0.1 per paper
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


def _compute_checkpoint_epochs(total_epochs, num_checkpoints=50):
    """Compute M evenly spaced checkpoint epochs."""
    if num_checkpoints >= total_epochs:
        return list(range(1, total_epochs + 1))
    step = total_epochs / num_checkpoints
    return sorted(set(max(1, round(step * i)) for i in range(1, num_checkpoints + 1)))


def make_config(dataset="spirals", hidden_dims=None, epochs=None,
                epsilon=0.1, lr=0.01, scheduler="none",
                adversarial=False, seed=42, device="cpu",
                scheduler_step_size=50, scheduler_gamma=0.5):
    """
    Flexible config builder. Computes step_size = epsilon/4 automatically.
    """
    cfg = ExperimentConfig()
    cfg.data.dataset = dataset
    cfg.seed = seed
    cfg.device = device

    if hidden_dims is not None:
        cfg.model.hidden_dims = hidden_dims

    # Set epochs per paper defaults if not overridden
    if epochs is not None:
        cfg.train.epochs = epochs
    elif dataset == "concentric_rings":
        cfg.train.epochs = 300
    else:
        cfg.train.epochs = 500

    cfg.train.lr = lr
    cfg.train.scheduler = scheduler
    cfg.train.scheduler_step_size = scheduler_step_size
    cfg.train.scheduler_gamma = scheduler_gamma

    cfg.adv.enabled = adversarial
    cfg.adv.epsilon = epsilon
    cfg.adv.step_size = epsilon / 4  # always compute dynamically

    cfg.tess.checkpoint_epochs = _compute_checkpoint_epochs(
        cfg.train.epochs, cfg.tess.num_checkpoints
    )

    return cfg


# Legacy helpers (backward compatible)
def get_standard_config(dataset="spirals"):
    return make_config(dataset=dataset, adversarial=False)


def get_adversarial_config(dataset="spirals"):
    return make_config(dataset=dataset, adversarial=True)
