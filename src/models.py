"""
ReLU MLP architectures for tessellation experiments.
Models are built as nn.Sequential for SplineCam compatibility.
"""

import torch
import torch.nn as nn


def make_relu_mlp(input_dim=2, hidden_dims=[32, 32, 32], output_dim=2):
    """
    Build a ReLU MLP as nn.Sequential.
    SplineCam requires Sequential models with supported layers.
    """
    layers = []
    prev_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.ReLU())
        prev_dim = h_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    model = nn.Sequential(*layers)
    return model


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation_pattern(model, x):
    """
    Extract the binary activation pattern (which ReLUs are on/off)
    for a given input x. Returns a binary vector.
    """
    patterns = []
    h = x
    for layer in model:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            patterns.append((h > 0).float())
    return torch.cat(patterns, dim=-1)


def count_unique_activation_patterns(model, X, batch_size=256):
    """
    Count the number of distinct activation patterns (i.e., linear regions
    visited) for a set of inputs X.
    """
    model.eval()
    all_patterns = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            pat = get_activation_pattern(model, batch)
            all_patterns.append(pat)
    all_patterns = torch.cat(all_patterns, dim=0)

    # Convert to hashable tuples to count unique patterns
    unique = set()
    for row in all_patterns:
        unique.add(tuple(row.int().tolist()))

    return len(unique)
