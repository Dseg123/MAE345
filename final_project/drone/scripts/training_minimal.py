#!/usr/bin/env python
"""Minimal training script to debug Hydra hang."""

import os
import sys
from pathlib import Path

print("Step 1: Script started")
print(f"Current working directory: {os.getcwd()}")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Step 2: Added {project_root} to sys.path")

# Import Hydra
import hydra
from omegaconf import DictConfig, OmegaConf
print("Step 3: Hydra imported")

# Import other modules one by one to see which causes the hang
import torch
print("Step 4: Torch imported")

import torch.nn as nn
print("Step 5: torch.nn imported")

from typing import Tuple, Dict, Any
print("Step 6: typing imported")

import json
print("Step 7: json imported")

from datetime import datetime
print("Step 8: datetime imported")

from hydra.utils import instantiate
print("Step 9: hydra.utils imported")

# Now import dataloader - this might be the culprit
from drone.datasets.dataloader import create_dataloaders
print("Step 10: dataloader imported")

def initialize_model(cfg: DictConfig) -> nn.Module:
    """Initialize model based on configuration using Hydra instantiate.

    Args:
        cfg: Configuration object containing model specifications under 'models' key

    Returns:
        Initialized PyTorch model
    """
    # Use Hydra's instantiate to create model from config
    # Exclude 'output_space' which is metadata for loss function, not a constructor arg
    model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    output_space = model_cfg.pop('output_space', None)  # Remove before instantiation
    model = instantiate(model_cfg)
    return model


def loss_fn(outputs: torch.Tensor, labels: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Loss function for training.

    Args:
        outputs: Model outputs
        labels: Ground-truth labels (continuous actions)
        cfg: Configuration object with models config

    Returns:
        Computed loss value
    """
    if cfg.models.output_space == 'discrete':
        # outputs: (B, action_dim, num_bins)
        # labels: (B, action_dim) continuous actions

        # Convert continuous labels to bin indices
        clipped = labels.clamp(min=cfg.models.action_low, max=cfg.models.action_high)
        norm = (clipped - cfg.models.action_low) / (cfg.models.action_high - cfg.models.action_low)
        bin_labels = (norm * (cfg.models.num_bins - 1)).round().long()

        # Flatten and compute cross-entropy
        B, action_dim, num_bins = outputs.shape
        outputs_flat = outputs.reshape(B * action_dim, num_bins)
        labels_flat = bin_labels.reshape(B * action_dim)

        loss = nn.functional.cross_entropy(outputs_flat, labels_flat)
        return loss

    elif cfg.models.output_space == 'continuous':
        # outputs: (B, action_dim)
        # labels: (B, action_dim)
        loss = nn.functional.mse_loss(outputs, labels)
        return loss

    else:
        raise ValueError(f"Unknown output space: {cfg.models.output_space}")


# REMOVED do_training() function - it had problematic for...enumerate() syntax that Hydra scans
# All training logic is now inline in main()

@hydra.main(version_base=None, config_path="/home/de7281/MAE345/final_project/drone/configs", config_name="train_config_test")
def main(cfg: DictConfig):
    print("ENTERED MAIN FUNCTION!")
    sys.stdout.flush()

    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print("About to print config...")
    sys.stdout.flush()
    print(OmegaConf.to_yaml(cfg))
    sys.stdout.flush()
    print("=" * 80)
    sys.stdout.flush()

    # Determine experiment directory
    if "experiment_dir" in cfg and cfg.experiment_dir is not None:
        # Use provided experiment directory (from SLURM script)
        experiment_dir = Path(cfg.experiment_dir)
    else:
        # Create new experiment directory with timestamp
        # scratch_dir = Path("/scratch/gpfs/TSILVER/de7281/MAE345")
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # experiment_dir = scratch_dir / f"run_{timestamp}"
        raise ValueError("Must specify an experiment directory")

    # Create directory structure
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)

    print(f"Experiment directory: {experiment_dir}")
    sys.stdout.flush()

    # # Train model
    print("HIIIIII")
    print("hello")
    print("im going to trainnow")
    sys.stdout.flush()
    # model, history = do_training(cfg)
    print("\nLoading datasets...")
    sys.stdout.flush()

    config = cfg


    print("=" * 80)
    print("Starting training")
    print("=" * 80)
    sys.stdout.flush()


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        data_dir=config.dataset.data_dir,
        train_trials=config.training.train_trials,
        val_trials=config.training.val_trials,
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size,
        normalize_states=config.dataset.normalize_states,
        normalize_actions=config.dataset.normalize_actions,
        normalize_images=config.dataset.normalize_images,
        num_workers=config.training.num_workers,
        shuffle_train=config.training.shuffle_train,
        augment=config.dataset.augment
    )

    # Initialize model
    print("\nInitializing model...")
    model = initialize_model(config)
    model = model.to(device)
    print(f"Model type: {config.models.output_space}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }

    # Training loop
    print("\n" + "=" * 80)
    print(f"Training for {config.training.num_epochs} epochs")
    print("=" * 80)

    for epoch in range(config.training.num_epochs):
        # ============= Training Phase =============
        print("Epoch number", epoch)
        model.train()
        train_loss = 0.0
        train_batches = 0

        print(f"About to create iterator from train_loader (type: {type(train_loader)})")
        print(f"Train loader length: {len(train_loader)}")
        print(f"Train loader batch_size: {train_loader.batch_size}")
        print(f"Train loader num_workers: {train_loader.num_workers}")
        print("About to start iterating...")

        # Use manual iterator instead of for loop to avoid Hydra issue
        iterator = iter(train_loader)
        batch_idx = 0
        while True:
            try:
                batch = next(iterator)
                print(f"Batch {batch_idx}/{len(train_loader)}")

                # Move data to device
                images = batch['observation'].to(device)
                actions = batch['action'].to(device)
                print("Moved to device")

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                print("Forward pass complete")

                # Compute loss
                loss = loss_fn(outputs, actions, config)
                print("Starting Backprop")

                # Backward pass
                loss.backward()
                print("Backprop complete")

                optimizer.step()
                print("Stepped optimizer")

                # Track metrics
                train_loss += loss.item()
                train_batches += 1
                print(f"Loss: {loss.item():.4f}")

                batch_idx += 1

            except StopIteration:
                print("Finished training batches")
                break

        # Average training loss
        avg_train_loss = train_loss / train_batches
        history['train_losses'].append(avg_train_loss)

        # ============= Validation Phase =============
        model.eval()
        val_loss = 0.0
        val_batches = 0

        print("\nStarting validation...")
        with torch.no_grad():
            # Use manual iterator for validation too
            val_iterator = iter(val_loader)
            val_batch_idx = 0
            while True:
                try:
                    batch = next(val_iterator)
                    print(f"Val batch {val_batch_idx}/{len(val_loader)}")

                    # Move data to device
                    images = batch['observation'].to(device)
                    actions = batch['action'].to(device)

                    # Forward pass
                    outputs = model(images)

                    # Compute loss
                    loss = loss_fn(outputs, actions, config)

                    # Track metrics
                    val_loss += loss.item()
                    val_batches += 1
                    print(f"Val loss: {loss.item():.4f}")

                    val_batch_idx += 1

                except StopIteration:
                    print("Finished validation batches")
                    break

        # Average validation loss
        avg_val_loss = val_loss / val_batches
        history['val_losses'].append(avg_val_loss)

        # Track best model
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Best Val Loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch']+1})")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']+1}")
    print("=" * 80)


if __name__ == "__main__":
    print("Step 13: About to call main()")
    sys.stdout.flush()
    main()
    print("Step 14: Returned from main()")
