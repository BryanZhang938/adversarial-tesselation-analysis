"""
Tests for PGD adversarial helpers in src/adversarial.py.

Run with:
    python tests/test_adversarial.py

No pytest dependency: each test_* function uses plain asserts and is
called from __main__.
"""

import os
import sys

import torch

# Allow `python tests/test_adversarial.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adversarial import normalize_norm, pgd_attack, project_l1_ball


def test_normalize_norm_aliases():
    assert normalize_norm("l1") == "l1"
    assert normalize_norm("L_2") == "l2"
    assert normalize_norm("l-inf") == "linf"
    assert normalize_norm("infinity") == "linf"


def test_project_l1_ball_keeps_inside_rows_unchanged():
    delta = torch.tensor([[0.2, -0.3], [0.0, 0.5]], dtype=torch.float32)
    projected = project_l1_ball(delta, epsilon=1.0)
    assert torch.allclose(projected, delta)


def test_project_l1_ball_projects_outside_rows():
    delta = torch.tensor([[2.0, -1.0], [3.0, 0.0]], dtype=torch.float32)
    projected = project_l1_ball(delta, epsilon=1.0)
    l1_norms = projected.abs().sum(dim=1)
    assert torch.all(l1_norms <= 1.0 + 1e-6), l1_norms
    # Projection preserves signs for nonzero projected coordinates.
    assert projected[0, 0] >= 0
    assert projected[0, 1] <= 0
    assert projected[1, 0] >= 0


def test_pgd_attack_respects_l1_constraint():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,))
    epsilon = 0.25
    x_adv = pgd_attack(
        model, x, y, epsilon=epsilon, step_size=epsilon / 4,
        num_steps=4, norm="l1", random_start=True,
    )
    delta_l1 = (x_adv - x).abs().sum(dim=1)
    assert torch.all(delta_l1 <= epsilon + 1e-5), delta_l1.max().item()


def test_pgd_attack_respects_l2_constraint():
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,))
    epsilon = 0.3
    x_adv = pgd_attack(
        model, x, y, epsilon=epsilon, step_size=epsilon / 4,
        num_steps=4, norm="l2", random_start=True,
    )
    delta_l2 = (x_adv - x).norm(dim=1)
    assert torch.all(delta_l2 <= epsilon + 1e-5), delta_l2.max().item()


def test_pgd_attack_respects_linf_constraint():
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,))
    epsilon = 0.2
    x_adv = pgd_attack(
        model, x, y, epsilon=epsilon, step_size=epsilon / 4,
        num_steps=4, norm="linf", random_start=True,
    )
    delta_linf = (x_adv - x).abs().max(dim=1).values
    assert torch.all(delta_linf <= epsilon + 1e-5), delta_linf.max().item()


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
