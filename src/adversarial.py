"""
PGD adversarial attacks for adversarial training.
Implements the Madry et al. (2018) min-max formulation for L1, L2,
and Linf perturbation sets.
"""

import torch
import torch.nn as nn


def normalize_norm(norm):
    """Normalize common norm aliases to {"l1", "l2", "linf"}."""
    key = str(norm).lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "l1",
        "l1": "l1",
        "2": "l2",
        "l2": "l2",
        "inf": "linf",
        "infinity": "linf",
        "linf": "linf",
        "l∞": "linf",
    }
    if key not in aliases:
        raise ValueError(f"Unknown adversarial norm: {norm}")
    return aliases[key]


def project_l1_ball(delta, epsilon):
    """
    Project a batch of perturbations onto the L1 ball of radius epsilon.

    Implements the batched Duchi et al. simplex projection on abs(delta),
    then restores signs. `delta` is expected to have shape (B, D).
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if epsilon == 0:
        return torch.zeros_like(delta)

    orig_shape = delta.shape
    flat = delta.reshape(delta.shape[0], -1)
    abs_flat = flat.abs()
    l1_norm = abs_flat.sum(dim=1, keepdim=True)
    inside = l1_norm <= epsilon

    # Sort each row descending and find the threshold theta for rows outside
    # the L1 ball. Rows already inside are left unchanged below.
    u, _ = torch.sort(abs_flat, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1)
    idx = torch.arange(1, flat.shape[1] + 1, device=delta.device).view(1, -1)
    cond = u * idx > (cssv - epsilon)
    rho = cond.sum(dim=1, keepdim=True).clamp(min=1)
    theta = (cssv.gather(1, rho - 1) - epsilon) / rho
    projected = flat.sign() * torch.clamp(abs_flat - theta, min=0.0)
    projected = torch.where(inside, flat, projected)
    return projected.reshape(orig_shape)


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
        norm: "l1", "l2", or "linf"
        random_start: whether to initialize from random point in ball

    Returns:
        x_adv: adversarial examples, shape (B, D)
    """
    norm = normalize_norm(norm)
    model.eval()
    x_adv = x.clone().detach()

    if random_start:
        if norm == "linf":
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        elif norm == "l2":
            delta = torch.randn_like(x_adv)
            radius = torch.rand(x_adv.shape[0], 1, device=x_adv.device)
            radius = radius.pow(1.0 / x_adv.shape[1])
            delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            delta = delta * epsilon * radius
            x_adv = x_adv + delta
        elif norm == "l1":
            delta = torch.empty_like(x_adv).uniform_(-1.0, 1.0)
            delta = project_l1_ball(delta, epsilon)
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
            elif norm == "l1":
                # Steepest ascent direction for an L1 constraint uses the
                # dual L_inf norm: put the step on the largest-gradient
                # coordinate for each sample, then project back to the L1 ball.
                idx = grad.abs().argmax(dim=-1, keepdim=True)
                step = torch.zeros_like(grad).scatter_(
                    1, idx, grad.gather(1, idx).sign()
                )
                x_adv = x_adv + step_size * step
                delta = project_l1_ball(x_adv - x, epsilon)
                x_adv = x + delta

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
