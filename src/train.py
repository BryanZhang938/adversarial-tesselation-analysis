"""
Training loops for standard (ERM) and adversarial (PGD-AT) training.
Saves model checkpoints at specified epochs for tessellation analysis.

Updated April 2026: added evaluate_accuracy / evaluate_robust_accuracy
helpers and optional convergence tracking (X_test, y_test) for the
(gap, epsilon) feasibility sweep.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.adversarial import pgd_attack


def evaluate_accuracy(model, X, y, device="cpu"):
    """Compute clean accuracy on a held-out set."""
    model.eval()
    X_t = torch.from_numpy(X).to(device) if isinstance(X, np.ndarray) else X.to(device)
    y_t = torch.from_numpy(y).to(device) if isinstance(y, np.ndarray) else y.to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1)
    return (preds == y_t).float().mean().item()


def evaluate_robust_accuracy(model, X, y, epsilon, step_size=None,
                             num_steps=7, norm="l2", device="cpu"):
    """Compute robust accuracy (accuracy on PGD adversarial examples)."""
    model.eval()
    X_t = torch.from_numpy(X).to(device) if isinstance(X, np.ndarray) else X.to(device)
    y_t = torch.from_numpy(y).to(device) if isinstance(y, np.ndarray) else y.to(device)
    if step_size is None:
        step_size = epsilon / 4
    X_adv = pgd_attack(model, X_t, y_t, epsilon=epsilon,
                       step_size=step_size, num_steps=num_steps, norm=norm)
    with torch.no_grad():
        preds = model(X_adv).argmax(dim=1)
    return (preds == y_t).float().mean().item()


def train_model(model, dataloader, config, checkpoint_dir="checkpoints",
                run_name="standard", device="cpu",
                X_test=None, y_test=None):
    """
    Train a model with standard or adversarial training.

    Args:
        model: nn.Sequential ReLU MLP
        dataloader: training data
        config: ExperimentConfig
        checkpoint_dir: where to save checkpoints
        run_name: prefix for checkpoint filenames
        device: "cpu" or "cuda"
        X_test: optional test inputs for convergence tracking
        y_test: optional test labels for convergence tracking

    Returns:
        history: dict with training metrics per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if config.train.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.lr,
            momentum=getattr(config.train, 'momentum', 0.9),
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer}")

    scheduler = None
    if config.train.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.epochs
        )
    elif config.train.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
    # scheduler == "none" -> no scheduler (paper does not use one)

    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "adv_loss": [],
        "adv_acc": [],
        "test_clean_acc": [],
        "test_robust_acc": [],
        "lr": [],
    }

    checkpoint_epochs = set(config.tess.checkpoint_epochs)

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_adv_loss = 0.0
        total_adv_correct = 0
        total_samples = 0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_size = x_batch.size(0)

            # Generate adversarial examples if adversarial training is enabled
            if config.adv.enabled:
                x_adv = pgd_attack(
                    model, x_batch, y_batch,
                    epsilon=config.adv.epsilon,
                    step_size=config.adv.step_size,
                    num_steps=config.adv.num_steps,
                    norm=config.adv.norm,
                )
                # Train on adversarial examples
                model.train()
                optimizer.zero_grad()
                logits_adv = model(x_adv)
                loss = criterion(logits_adv, y_batch)
                loss.backward()
                optimizer.step()

                total_adv_loss += loss.item() * batch_size
                total_adv_correct += (logits_adv.argmax(1) == y_batch).sum().item()
            else:
                # Standard training
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate clean accuracy
            model.eval()
            with torch.no_grad():
                logits_clean = model(x_batch)
                clean_loss = criterion(logits_clean, y_batch)
                total_loss += clean_loss.item() * batch_size
                total_correct += (logits_clean.argmax(1) == y_batch).sum().item()
            model.train()

            total_samples += batch_size

        if scheduler:
            scheduler.step()

        # Record metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / total_samples)
        history["train_acc"].append(total_correct / total_samples)
        history["adv_loss"].append(
            total_adv_loss / total_samples if config.adv.enabled else 0.0
        )
        history["adv_acc"].append(
            total_adv_correct / total_samples if config.adv.enabled else 0.0
        )
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Test evaluation (if test data provided)
        if X_test is not None and y_test is not None:
            test_clean = evaluate_accuracy(model, X_test, y_test, device=device)
            history["test_clean_acc"].append(test_clean)
            if config.adv.enabled:
                test_robust = evaluate_robust_accuracy(
                    model, X_test, y_test,
                    epsilon=config.adv.epsilon,
                    num_steps=config.adv.num_steps,
                    norm=config.adv.norm,
                    device=device,
                )
                history["test_robust_acc"].append(test_robust)
            else:
                history["test_robust_acc"].append(0.0)
        else:
            history["test_clean_acc"].append(0.0)
            history["test_robust_acc"].append(0.0)

        # Save checkpoint at specified epochs
        if epoch in checkpoint_epochs:
            ckpt_path = os.path.join(
                checkpoint_dir, f"{run_name}_epoch{epoch:04d}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt_path)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            msg = (
                f"[{run_name}] Epoch {epoch}/{config.train.epochs} | "
                f"Loss: {history['train_loss'][-1]:.4f} | "
                f"Acc: {history['train_acc'][-1]:.4f}"
            )
            if config.adv.enabled:
                msg += (
                    f" | Adv Loss: {history['adv_loss'][-1]:.4f} | "
                    f"Adv Acc: {history['adv_acc'][-1]:.4f}"
                )
            if X_test is not None:
                msg += f" | Test: {history['test_clean_acc'][-1]:.4f}"
                if config.adv.enabled:
                    msg += f" | Rob: {history['test_robust_acc'][-1]:.4f}"
            print(msg)

    return history
