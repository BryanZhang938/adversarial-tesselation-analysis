"""
PGD adversarial attack for adversarial training.
Implements the Madry et al. (2018) min-max formulation.
"""

import torch
import torch.nn as nn


def pgd_attack(model, x, y, epsilon, step_size, num_steps,
               norm="linf", random_start=True):
    """
    Projected Gradient Descent (PGD) attack.

    Args:
        model: classifier
        x: clean inputs, shape (B, D)
        y: true labels, shape (B,)
        epsilon: perturbation budget
        step_size: step size per iteration
        num_steps: number of PGD steps
        norm: "linf" or "l2"
        random_start: whether to initialize from random point in ball

    Returns:
        x_adv: adversarial examples, shape (B, D)
    """
    model.eval()
    x_adv = x.clone().detach()

    if random_start:
        if norm == "linf":
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        elif norm == "l2":
            delta = torch.randn_like(x_adv)
            delta = delta / delta.norm(dim=-1, keepdim=True) * epsilon
            x_adv = x_adv + delta
        x_adv = x_adv.detach()

    criterion = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()

        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + step_size * grad.sign()
                # Project back to epsilon ball
                delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = x + delta
            elif norm == "l2":
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                x_adv = x_adv + step_size * grad / grad_norm
                # Project back to epsilon ball
                delta = x_adv - x
                delta_norm = delta.norm(dim=-1, keepdim=True)
                factor = torch.min(
                    torch.ones_like(delta_norm),
                    epsilon / delta_norm.clamp(min=1e-12)
                )
                x_adv = x + delta * factor

        x_adv = x_adv.detach()

    model.train()
    return x_adv


def compute_boundary_distance(model, x, y, epsilon=1.0, step_size=0.01,
                               num_steps=200, norm="l2"):
    """
    Estimate the distance from each point x to the nearest decision boundary
    using a targeted DeepFool-style iterative approach.

    Returns the l2 distance from each point to the boundary.
    """
    model.eval()
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    criterion = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        preds = logits.argmax(dim=-1)

        # Stop for points that already crossed the boundary
        crossed = (preds != y)
        if crossed.all():
            break

        loss = criterion(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()

        with torch.no_grad():
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            step = step_size * grad / grad_norm
            # Only step for points that haven't crossed yet
            x_adv = x_adv + step * (~crossed).float().unsqueeze(-1)

        x_adv = x_adv.detach()

    distances = (x_adv - x_orig).norm(dim=-1)
    model.train()
    return distances.detach()
