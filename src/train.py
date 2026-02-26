"""
Training loops for standard (ERM) and adversarial (PGD-AT) training.
Saves model checkpoints at specified epochs for tessellation analysis.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.adversarial import pgd_attack


def train_model(model, dataloader, config, checkpoint_dir="checkpoints",
                run_name="standard", device="cpu"):
    """
    Train a model with standard or adversarial training.

    Args:
        model: nn.Sequential ReLU MLP
        dataloader: training data
        config: ExperimentConfig
        checkpoint_dir: where to save checkpoints
        run_name: prefix for checkpoint filenames
        device: "cpu" or "cuda"

    Returns:
        history: dict with training metrics per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if config.train.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.lr,
            momentum=0.9,
            weight_decay=config.train.weight_decay
        )

    scheduler = None
    if config.train.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.epochs
        )
    elif config.train.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )

    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "adv_loss": [],
        "adv_acc": [],
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
            print(msg)

    return history
