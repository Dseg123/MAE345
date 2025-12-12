"""
Training script for drone imitation learning models.

This script provides functions to train and evaluate models on SLURM clusters.
It supports both discrete and continuous action models.

Usage with Hydra:
    python training.py
    python training.py model_name=discrete_action_model
    python training.py training.lr=0.0001 training.num_epochs=100
    python training.py save_dir=/path/to/save
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from typing import Tuple, Dict, Any
import json
from datetime import datetime
import hydra
from hydra.utils import instantiate

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from drone.datasets.dataloader import create_dataloaders




def initialize_model(cfg: DictConfig) -> nn.Module:
    """Initialize model based on configuration using Hydra instantiate.

    Args:
        cfg: Configuration object containing model specifications under 'models' key

    Returns:
        Initialized PyTorch model
    """
    # Use Hydra's instantiate to create model from config
    # Exclude 'output_space' which is metadata for loss function, not a constructor arg
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.to_container(cfg.models, resolve=True)
    output_space = model_cfg.pop('output_space', None)  # Remove before instantiation
    model = instantiate(model_cfg)
    return model


def continuous_to_bins(actions: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Convert continuous actions to bin indices.

    Args:
        actions: (B, action_dim) tensor of continuous actions
        cfg: Configuration object with models config

    Returns:
        (B, action_dim) LongTensor of bin indices
    """
    clipped = actions.clamp(min=cfg.models.action_low, max=cfg.models.action_high)
    norm = (clipped - cfg.models.action_low) / (cfg.models.action_high - cfg.models.action_low)
    bin_labels = (norm * (cfg.models.num_bins - 1)).round().long()
    return bin_labels


def compute_bin_accuracy(outputs: torch.Tensor, labels: torch.Tensor, cfg: DictConfig) -> Tuple[float, torch.Tensor]:
    """Compute bin-wise accuracy for discrete action models.

    Args:
        outputs: Model outputs (B, action_dim, num_bins) logits
        labels: Ground-truth continuous actions (B, action_dim)
        cfg: Configuration object with models config

    Returns:
        Tuple of (overall_accuracy, per_action_accuracy)
        - overall_accuracy: float, fraction of correct bin predictions across all actions
        - per_action_accuracy: (action_dim,) tensor with accuracy per action dimension
    """
    if cfg.models.output_space != 'discrete':
        return 0.0, torch.zeros(1)

    # Convert continuous labels to bin indices
    bin_labels = continuous_to_bins(labels, cfg)  # (B, action_dim)

    # Get predicted bins (argmax of logits)
    predicted_bins = torch.argmax(outputs, dim=-1)  # (B, action_dim)

    # Compute overall accuracy
    correct = (predicted_bins == bin_labels).float()
    overall_accuracy = correct.mean().item()

    # Compute per-action accuracy
    per_action_accuracy = correct.mean(dim=0)  # (action_dim,)

    return overall_accuracy, per_action_accuracy


def loss_fn(outputs: torch.Tensor, labels: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Loss function for training with increased weight on non-zero y-coordinate examples.

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
        bin_labels = continuous_to_bins(labels, cfg)  # (B, action_dim)

        # Get weighting factor for non-zero y-coordinate samples
        nonzero_y_weight = cfg.training.get('nonzero_y_weight', 1.0)

        B, action_dim, num_bins = outputs.shape
        center_bin = (num_bins - 1) // 2  # Middle bin represents 0 action

        # Create per-sample weights based on y-coordinate (index 1 = vy)
        sample_weights = torch.ones(B, device=outputs.device)

        # Identify samples with non-zero vy (obstacle avoidance maneuvers)
        vy_bins = bin_labels[:, 1]  # Extract vy bins for all samples
        nonzero_vy_mask = (vy_bins != center_bin)  # True where vy != 0

        # Apply higher weight to non-zero vy samples
        sample_weights[nonzero_vy_mask] = nonzero_y_weight

        # Compute loss per action dimension, then weight by sample
        total_loss = 0.0
        for action_idx in range(action_dim):
            action_outputs = outputs[:, action_idx, :]  # (B, num_bins)
            action_labels = bin_labels[:, action_idx]  # (B,)

            # Compute cross-entropy for this action dimension
            action_loss = nn.functional.cross_entropy(action_outputs, action_labels, reduction='none')  # (B,)

            # Weight by sample weights and average
            weighted_action_loss = (action_loss * sample_weights).mean()
            total_loss += weighted_action_loss

        # Average across action dimensions
        loss = total_loss / action_dim
        return loss

    elif cfg.models.output_space == 'continuous':
        # outputs: (B, action_dim)
        # labels: (B, action_dim)
        loss = nn.functional.mse_loss(outputs, labels)
        return loss

    else:
        raise ValueError(f"Unknown output space: {cfg.models.output_space}")


def train(config: DictConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the model.

    Args:
        config: Configuration object containing all training parameters

    Returns:
        Tuple of (trained_model, training_history)
    """
    print("=" * 80)
    print("Starting training")
    print("=" * 80)

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
    sys.stdout.flush()

    # Initialize model
    print("\nInitializing model...")
    model = initialize_model(config)
    model = model.to(device)
    print(f"Model type: {config.models.output_space}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Initialize optimizer with weight decay (L2 regularization)
    weight_decay = config.training.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=weight_decay)
    print(f"Optimizer: Adam(lr={config.training.lr}, weight_decay={weight_decay})")

    # Learning rate scheduler - reduce LR when validation loss plateaus
    use_scheduler = config.training.get('use_lr_scheduler', True)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.get('lr_factor', 0.5),  # Reduce LR by half
            patience=config.training.get('lr_patience', 5),  # Wait 5 epochs before reducing
            min_lr=config.training.get('min_lr', 1e-6)
        )
        print(f"LR Scheduler: ReduceLROnPlateau(factor={config.training.get('lr_factor', 0.5)}, patience={config.training.get('lr_patience', 5)})")
    else:
        scheduler = None
        print("No LR scheduler")

    # Enable cudnn benchmarking for faster convolutions (if on GPU)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn.benchmark for faster GPU training")

    # Early stopping setup
    early_stopping_patience = config.training.get('early_stopping_patience', 10)
    early_stopping_min_delta = config.training.get('early_stopping_min_delta', 0.001)
    epochs_without_improvement = 0
    print(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'best_val_loss': float('inf'),
        'best_val_accuracy': 0.0,
        'best_epoch': 0,
        'stopped_early': False,
        'total_epochs_run': 0
    }
    sys.stdout.flush()

    # Training loop
    print("\n" + "=" * 80)
    print(f"Training for up to {config.training.num_epochs} epochs")
    print("=" * 80)
    sys.stdout.flush()

    for epoch in range(config.training.num_epochs):
        # ============= Training Phase =============
        sys.stdout.flush()
        print("Epoch number", epoch)
        model.train()
        train_loss = 0.0
        train_batches = 0

        print(f"About to create iterator from train_loader (type: {type(train_loader)})")
        print(f"Train loader length: {len(train_loader)}")
        print(f"Train loader batch_size: {train_loader.batch_size}")
        print(f"Train loader num_workers: {train_loader.num_workers}")
        print("About to start iterating...")
        sys.stdout.flush()

        iterator = iter(train_loader)


        for i in range(len(train_loader)):
            batch = next(iterator)
            train_batches += 1

            # Move data to device
            images = batch['observation'].to(device)
            actions = batch['action'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Compute loss and backward
            loss = loss_fn(outputs, actions, config)
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()

            # Print progress
            print(f"Batch {train_batches}/{len(train_loader)}, Loss: {loss.item():.4f}")
            sys.stdout.flush()

        # Average training loss
        avg_train_loss = train_loss / train_batches
        history['train_losses'].append(avg_train_loss)

        # ============= Validation Phase =============
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        val_per_action_acc = torch.zeros(config.models.action_dim)

        print("\nStarting validation...")
        iterator = iter(val_loader)
        with torch.no_grad():
            for i in range(len(val_loader)):
                print(f"Val batch {val_batches}/{len(val_loader)}")
                batch = next(iterator)
                # Move data to device
                images = batch['observation'].to(device)
                actions = batch['action'].to(device)

                # Forward pass
                outputs = model(images)

                # Compute loss
                loss = loss_fn(outputs, actions, config)

                # Compute accuracy (for discrete models)
                batch_acc, batch_per_action_acc = compute_bin_accuracy(outputs, actions, config)

                # DEBUG: Print detailed accuracy info for first batch
                if val_batches == 0:
                    predicted_bins = torch.argmax(outputs, dim=-1)
                    true_bins = continuous_to_bins(actions, config)
                    print(f"  DEBUG - First batch predictions:")
                    print(f"    Predicted bins sample: {predicted_bins[0].cpu().numpy()}")
                    print(f"    True bins sample: {true_bins[0].cpu().numpy()}")
                    print(f"    Output logits sample: {outputs[0, :, :].cpu().detach().numpy()}")
                    print(f"    Batch accuracy: {batch_acc:.4f}")

                # Track metrics
                val_loss += loss.item()
                val_accuracy += batch_acc
                val_per_action_acc += batch_per_action_acc.cpu()
                val_batches += 1

                print(f"Val loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")

        # Average validation metrics
        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_accuracy / val_batches
        avg_per_action_acc = val_per_action_acc / val_batches

        history['val_losses'].append(avg_val_loss)
        history['val_accuracies'].append(avg_val_accuracy)

        # Track best model and check for early stopping (based on val loss)
        improved = False
        if avg_val_loss < history['best_val_loss'] - early_stopping_min_delta:
            # Significant improvement in validation loss
            improved = True
            epochs_without_improvement = 0
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch

            # Also track best accuracy for discrete models
            if config.models.output_space == 'discrete':
                history['best_val_accuracy'] = avg_val_accuracy

            # Save best model checkpoint
            checkpoint_dir = Path(config.experiment_dir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            # Remove previous best checkpoint to save space
            for old_ckpt in checkpoint_dir.glob("best_model_epoch_*.pth"):
                old_ckpt.unlink()

            # Save new best checkpoint
            best_ckpt_path = checkpoint_dir / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy if config.models.output_space == 'discrete' else None,
                'config': OmegaConf.to_container(config, resolve=True)
            }, best_ckpt_path)
            print(f"  💾 Saved best model checkpoint: {best_ckpt_path.name}")

        else:
            # No improvement
            epochs_without_improvement += 1

            # For discrete models, also update best accuracy if it improved
            if config.models.output_space == 'discrete' and avg_val_accuracy > history['best_val_accuracy']:
                history['best_val_accuracy'] = avg_val_accuracy

        # Update total epochs run
        history['total_epochs_run'] = epoch + 1

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history.setdefault('learning_rates', []).append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        if config.models.output_space == 'discrete':
            print(f"  Val Accuracy: {avg_val_accuracy:.4f} ({avg_val_accuracy*100:.2f}%)")
            action_names = ['vx', 'vy', 'vz', 'yaw_rate']
            for i, name in enumerate(action_names):
                print(f"    {name}: {avg_per_action_acc[i]:.4f}")
            print(f"  Best Val Acc: {history['best_val_accuracy']:.4f} (Epoch {history['best_epoch']+1})")
        print(f"  Best Val Loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch']+1})")
        if scheduler is not None:
            print(f"  Learning Rate: {current_lr:.2e}")

        # Early stopping status
        if improved:
            print(f"  ✓ Validation loss improved!")
        else:
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        print("-" * 80)

        # Check early stopping condition
        if epochs_without_improvement >= early_stopping_patience:
            print("\n" + "=" * 80)
            print(f"EARLY STOPPING triggered after {epoch+1} epochs")
            print(f"No improvement in validation loss for {early_stopping_patience} consecutive epochs")
            print("=" * 80)
            history['stopped_early'] = True
            break

    print("\n" + "=" * 80)
    if history['stopped_early']:
        print(f"Training stopped early at epoch {history['total_epochs_run']}")
    else:
        print("Training complete!")
    print(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']+1}")
    if config.models.output_space == 'discrete':
        print(f"Best validation accuracy: {history['best_val_accuracy']:.4f} ({history['best_val_accuracy']*100:.2f}%)")
    print("=" * 80)

    return model, history


def evaluate(model: nn.Module, config: DictConfig) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: Trained PyTorch model
        config: Configuration object

    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating model on validation set")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create validation dataloader
    _, val_loader = create_dataloaders(
        data_dir=config.dataset.data_dir,
        train_trials=config.training.train_trials,
        val_trials=config.training.val_trials,
        batch_size=config.training.batch_size,
        image_size=config.dataset.image_size,
        normalize_states=config.dataset.normalize_states,
        normalize_actions=config.dataset.normalize_actions,
        normalize_images=config.dataset.normalize_images,
        num_workers=config.training.num_workers,
        shuffle_train=False,
        augment=False  # No augmentation for evaluation
    )

    # Evaluation metrics
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    total_per_action_acc = torch.zeros(config.models.action_dim)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move data to device
            images = batch['observation'].to(device)
            actions = batch['action'].to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = loss_fn(outputs, actions, config)

            # Compute accuracy (for discrete models)
            batch_acc, batch_per_action_acc = compute_bin_accuracy(outputs, actions, config)

            # Track metrics
            total_loss += loss.item()
            total_accuracy += batch_acc
            total_per_action_acc += batch_per_action_acc.cpu()
            total_batches += 1

    # Compute average metrics
    avg_loss = total_loss / total_batches
    avg_accuracy = total_accuracy / total_batches
    avg_per_action_acc = total_per_action_acc / total_batches

    metrics = {
        'val_loss': avg_loss,
        'val_accuracy': avg_accuracy,
        'num_batches': total_batches,
        'num_samples': total_batches * config.training.batch_size
    }

    # Add per-action accuracies for discrete models
    if config.models.output_space == 'discrete':
        action_names = ['vx', 'vy', 'vz', 'yaw_rate']
        for i, name in enumerate(action_names):
            metrics[f'val_accuracy_{name}'] = avg_per_action_acc[i].item()

    print(f"\nValidation Loss: {avg_loss:.4f}")
    if config.models.output_space == 'discrete':
        print(f"Validation Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print("\nPer-action accuracy:")
        action_names = ['vx', 'vy', 'vz', 'yaw_rate']
        for i, name in enumerate(action_names):
            print(f"  {name}: {avg_per_action_acc[i]:.4f} ({avg_per_action_acc[i]*100:.2f}%)")
    print(f"Total samples evaluated: {metrics['num_samples']}")
    print("=" * 80)

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def train_and_evaluate(cfg: DictConfig) -> None:
    """Complete training and evaluation pipeline using Hydra.

    Args:
        cfg: Hydra configuration object (automatically loaded from configs/train_config.yaml)
             Can be overridden via command line: python training.py training.lr=0.001
    """

    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

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

    # Train model
    print("HIIIIII")
    print("hello")
    print("im going to trainnow")
    sys.stdout.flush()
    model, history = train(cfg)

    # Load best checkpoint for final evaluation and saving
    checkpoint_dir = experiment_dir / "checkpoints"
    best_checkpoint = list(checkpoint_dir.glob("best_model_epoch_*.pth"))

    if best_checkpoint:
        best_ckpt_path = best_checkpoint[0]
        print(f"\nLoading best checkpoint from epoch {history['best_epoch']+1}: {best_ckpt_path.name}")
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best checkpoint val_loss: {checkpoint['val_loss']:.4f}")
    else:
        print(f"\nNo checkpoint found, using final model state")

    # Evaluate best model
    metrics = evaluate(model, cfg)

    # Save final best model (clean state dict only)
    model_path = experiment_dir / "models" / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nBest model saved to: {model_path}")

    # Also copy the best checkpoint to models directory for reference
    if best_checkpoint:
        best_model_full_path = experiment_dir / "models" / "best_model_full.pth"
        torch.save(checkpoint, best_model_full_path)
        print(f"Full checkpoint saved to: {best_model_full_path}")

    # Save configuration
    config_save_path = experiment_dir / "configs" / "config.yaml"
    OmegaConf.save(cfg, config_save_path)
    print(f"Configuration saved to: {config_save_path}")

    # Save training history
    history_path = experiment_dir / "results" / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Save evaluation metrics
    metrics_path = experiment_dir / "results" / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")

    # Save summary
    timestamp = experiment_dir.name.split('_', 1)[-1] if '_' in experiment_dir.name else datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'model_type': cfg.models.output_space,
        'timestamp': timestamp,
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch'] + 1,
        'eval_loss': metrics['val_loss'],
        'num_epochs': cfg.training.num_epochs,
        'num_samples': metrics['num_samples']
    }

    # Add accuracy metrics for discrete models
    if cfg.models.output_space == 'discrete':
        summary['final_val_accuracy'] = history['val_accuracies'][-1]
        summary['best_val_accuracy'] = history['best_val_accuracy']
        summary['eval_accuracy'] = metrics['val_accuracy']
        # Add per-action accuracies
        action_names = ['vx', 'vy', 'vz', 'yaw_rate']
        for name in action_names:
            if f'val_accuracy_{name}' in metrics:
                summary[f'eval_accuracy_{name}'] = metrics[f'val_accuracy_{name}']

    summary_path = experiment_dir / "results" / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print(f"All results saved to: {experiment_dir}")
    print("=" * 80)


if __name__ == "__main__":
    print("HELLO")
    print(f"Current working directory: {os.getcwd()}")
    print("About to call train_and_evaluate (decorated with @hydra.main)")
    sys.stdout.flush()  # Force flush to ensure we see this
    train_and_evaluate()
    print("Returned from train_and_evaluate")
