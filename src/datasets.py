"""
Synthetic 2D datasets for tessellation experiments.
All datasets produce points in roughly [-1, 1]^2 for clean visualization.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _normalize_to_unit_square(X):
    """Normalize points to [-1, 1]^2 by centering and scaling by max abs value."""
    X = X - X.mean(axis=0)
    scale = np.abs(X).max()
    if scale > 0:
        X = X / scale
    return X


def make_spirals(n_samples=500, noise=0.1, seed=42):
    """
    Two interleaved Archimedean spirals with additive Gaussian noise (sigma=noise).
    Normalized to [-1, 1]^2 per paper specification.
    """
    rng = np.random.RandomState(seed)
    n = n_samples // 2

    theta = np.sqrt(np.linspace(0, 1, n)) * 3 * np.pi

    # Spiral 1
    r1 = theta + np.pi
    x1 = np.column_stack([
        r1 * np.cos(theta) / (3 * np.pi),
        r1 * np.sin(theta) / (3 * np.pi),
    ])
    x1 += rng.randn(n, 2) * noise

    # Spiral 2
    r2 = theta + np.pi
    x2 = np.column_stack([
        -r2 * np.cos(theta) / (3 * np.pi),
        -r2 * np.sin(theta) / (3 * np.pi),
    ])
    x2 += rng.randn(n, 2) * noise

    X = np.vstack([x1, x2]).astype(np.float32)
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.int64)

    # Normalize to [-1, 1]^2
    X = _normalize_to_unit_square(X)

    return X, y


def make_concentric_rings(n_samples=500, noise=0.05, seed=42):
    """
    Two concentric rings with a gap, admitting a radially symmetric
    optimal boundary. Normalized to [-1, 1]^2.
    """
    rng = np.random.RandomState(seed)
    n = n_samples // 2

    # Inner ring
    theta1 = rng.uniform(0, 2 * np.pi, n)
    r1 = 0.4 + rng.randn(n) * noise
    x1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    # Outer ring
    theta2 = rng.uniform(0, 2 * np.pi, n)
    r2 = 1.0 + rng.randn(n) * noise
    x2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])

    X = np.vstack([x1, x2]).astype(np.float32)
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(np.int64)

    # Normalize to [-1, 1]^2
    X = _normalize_to_unit_square(X)

    return X, y


def make_moons(n_samples=500, noise=0.1, seed=42):
    """Two interleaving half-moons."""
    from sklearn.datasets import make_moons as _make_moons
    X, y = _make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    # Center and scale to [-1, 1]
    X = X.astype(np.float32)
    X -= X.mean(axis=0)
    X /= np.abs(X).max() * 1.1
    y = y.astype(np.int64)
    return X, y


DATASET_REGISTRY = {
    "spirals": make_spirals,
    "concentric_rings": make_concentric_rings,
    "moons": make_moons,
}


def get_dataset(name, n_samples=500, noise=0.1, seed=42, n_test=200):
    """
    Generate train and test datasets.

    Returns:
        train_dataset: TensorDataset for training
        X_train, y_train: numpy arrays
        X_test, y_test: numpy arrays (n_test points, separate seed)
    """
    make_fn = DATASET_REGISTRY[name]
    X_train, y_train = make_fn(n_samples=n_samples, noise=noise, seed=seed)
    # Generate test set with a different seed
    X_test, y_test = make_fn(n_samples=n_test, noise=noise, seed=seed + 1000)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)
    train_dataset = TensorDataset(X_tensor, y_tensor)

    return train_dataset, X_train, y_train, X_test, y_test


def get_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
