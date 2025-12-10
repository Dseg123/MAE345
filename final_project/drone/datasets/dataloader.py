"""
PyTorch DataLoader for Crazyflie Imitation Learning Dataset.
Loads synchronized state-action-observation triplets for training IL policies.
"""

import json
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2
from typing import List, Dict, Tuple, Optional


class CrazyflieILDataset(Dataset):
    """
    PyTorch Dataset for Crazyflie imitation learning data.
    
    Each sample contains:
        - observation: camera image (processed)
        - state: drone state (position, velocity, orientation)
        - action: commanded velocities [vx, vy, vz, yaw_rate]
    """
    
    def __init__(
        self,
        data_dir: str = 'imitation_data',
        trial_numbers: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        normalize_images: bool = True,
        normalize_states: bool = True,
        normalize_actions: bool = False,
        augment: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing trial folders
            trial_numbers: List of trial numbers to include (None = all trials)
            image_size: Target size for images (height, width)
            normalize_images: Whether to normalize images to [-1, 1]
            normalize_states: Whether to normalize state values
            normalize_actions: Whether to normalize action values
            augment: Whether to apply data augmentation (for training)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.normalize_states = normalize_states
        self.normalize_actions = normalize_actions
        self.augment = augment
        
        # Load all trials
        self.trials = []
        self.samples = []
        self._load_trials(trial_numbers)
        
        # Compute normalization statistics if needed
        self.state_stats = None
        self.action_stats = None
        if normalize_states or normalize_actions:
            self._compute_normalization_stats()
        
        # Setup image transforms
        self._setup_transforms()
        
        print(f"Dataset initialized:")
        print(f"  Trials: {len(self.trials)}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Image size: {image_size}")
        print(f"  Normalization: images={normalize_images}, states={normalize_states}, actions={normalize_actions}")
    
    def _load_trials(self, trial_numbers: Optional[List[int]]):
        """Load trial data from disk."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Find all trial directories
        trial_dirs = sorted([d for d in self.data_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('trial_')])
        
        if len(trial_dirs) == 0:
            raise ValueError(f"No trials found in {self.data_dir}")
        
        # Filter by trial numbers if specified
        print(type(trial_numbers), trial_dirs)
        if trial_numbers is not None:
            trial_dirs = [d for d in trial_dirs 
                         if int(d.name.split('_')[1]) in trial_numbers]
        
        # Load each trial
        for trial_dir in trial_dirs:
            trial_data = self._load_single_trial(trial_dir)
            if trial_data is not None:
                self.trials.append(trial_data)

                # Add samples from this trial, but only if image exists
                for i in range(len(trial_data['data_log'])):
                    entry = trial_data['data_log'][i]
                    image_path = trial_data['trial_dir'] / entry['image_path']

                    # Only add sample if image file exists
                    if image_path.exists():
                        self.samples.append({
                            'trial_idx': len(self.trials) - 1,
                            'sample_idx': i
                        })
                    else:
                        print(f"Warning: Skipping frame {i} in {trial_dir.name} - image not found: {entry['image_path']}")
    
    def _load_single_trial(self, trial_dir: Path) -> Optional[Dict]:
        """Load a single trial's data."""
        data_log_path = trial_dir / "data_log.json"
        
        if not data_log_path.exists():
            print(f"Warning: Skipping {trial_dir.name} - no data_log.json found")
            return None
        
        with open(data_log_path, 'r') as f:
            data_log = json.load(f)
        
        # Load metadata if available
        metadata = None
        metadata_path = trial_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            'trial_dir': trial_dir,
            'data_log': data_log,
            'metadata': metadata
        }
    
    def _compute_normalization_stats(self):
        """Compute mean and std for state and action normalization."""
        all_states = []
        all_actions = []
        
        for trial in self.trials:
            for entry in trial['data_log']:
                # Collect state
                state = [
                    entry['state']['x'],
                    entry['state']['y'],
                    entry['state']['z'],
                    entry['state']['vx'],
                    entry['state']['vy'],
                    entry['state']['vz'],
                    entry['state']['roll'],
                    entry['state']['pitch'],
                    entry['state']['yaw']
                ]
                all_states.append(state)
                
                # Collect action
                all_actions.append(entry['action'])
        
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        
        self.state_stats = {
            'mean': torch.FloatTensor(all_states.mean(axis=0)),
            'std': torch.FloatTensor(all_states.std(axis=0) + 1e-8)  # Add small epsilon
        }
        
        self.action_stats = {
            'mean': torch.FloatTensor(all_actions.mean(axis=0)),
            'std': torch.FloatTensor(all_actions.std(axis=0) + 1e-8)
        }
        
        print(f"\nNormalization statistics computed:")
        print(f"  State mean: {self.state_stats['mean'].numpy()}")
        print(f"  State std: {self.state_stats['std'].numpy()}")
        print(f"  Action mean: {self.action_stats['mean'].numpy()}")
        print(f"  Action std: {self.action_stats['std'].numpy()}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
        ]
        
        if self.augment:
            # TODO: Add augmentation for training
            transform_list.extend([
                transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.3),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=12),
                transforms.RandomInvert(p=0.1),
            ])
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize_images:
            # Normalize to [-1, 1] (ImageNet stats could also be used)
            transform_list.extend([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                v2.GaussianNoise(sigma=0.07)
            ])
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'observation': Image tensor [3, H, W]
                - 'state': State vector [9]
                - 'action': Action vector [4]
        """
        sample_info = self.samples[idx]
        trial = self.trials[sample_info['trial_idx']]
        entry = trial['data_log'][sample_info['sample_idx']]
        
        # Load and process image
        image_path = trial['trial_dir'] / entry['image_path']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        observation = self.transform(image)
        
        # Extract state vector
        state = torch.FloatTensor([
            entry['state']['x'],
            entry['state']['y'],
            entry['state']['z'],
            entry['state']['vx'],
            entry['state']['vy'],
            entry['state']['vz'],
            entry['state']['roll'],
            entry['state']['pitch'],
            entry['state']['yaw']
        ])
        
        # Normalize state if needed
        if self.normalize_states and self.state_stats is not None:
            state = (state - self.state_stats['mean']) / self.state_stats['std']
        
        # Extract action vector
        action = torch.FloatTensor(entry['action'])
        
        # Normalize action if needed
        if self.normalize_actions and self.action_stats is not None:
            action = (action - self.action_stats['mean']) / self.action_stats['std']
        
        return {
            'observation': observation,
            'state': state,
            'action': action
        }
    
    def get_data_stats(self) -> Dict:
        """Get dataset statistics for analysis."""
        return {
            'num_trials': len(self.trials),
            'num_samples': len(self.samples),
            'state_stats': self.state_stats,
            'action_stats': self.action_stats,
            'image_size': self.image_size
        }


def create_dataloaders(
    data_dir: str = 'imitation_data',
    train_trials: Optional[List[int]] = None,
    val_trials: Optional[List[int]] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    normalize_states: bool = True,
    normalize_actions: bool = False,
    normalize_images: bool = True,
    num_workers: int = 4,
    shuffle_train: bool = True,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Root directory containing trial folders
        train_trials: List of trial numbers for training (None = auto-split)
        val_trials: List of trial numbers for validation (None = auto-split)
        batch_size: Batch size
        image_size: Target image size
        normalize_states: Whether to normalize states
        normalize_actions: Whether to normalize actions
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader
    """
    
    # Auto-split trials if not specified
    if train_trials is None and val_trials is None:
        data_path = Path(data_dir)
        all_trials = sorted([int(d.name.split('_')[1]) 
                           for d in data_path.iterdir() 
                           if d.is_dir() and d.name.startswith('trial_')])
        
        # Use 80-20 split
        split_idx = int(len(all_trials) * 0.8)
        train_trials = all_trials[:split_idx]
        val_trials = all_trials[split_idx:]
        
        print(f"Auto-split: {len(train_trials)} train trials, {len(val_trials)} val trials")
    
    # Create datasets
    train_dataset = CrazyflieILDataset(
        data_dir=data_dir,
        trial_numbers=train_trials,
        image_size=image_size,
        normalize_images=normalize_images,
        normalize_states=normalize_states,
        normalize_actions=normalize_actions,
        augment=augment  # Use augmentation for training
    )
    
    val_dataset = CrazyflieILDataset(
        data_dir=data_dir,
        trial_numbers=val_trials,
        image_size=image_size,
        normalize_images=normalize_images,
        normalize_states=normalize_states,
        normalize_actions=normalize_actions,
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Quick test
    print("Testing CrazyflieILDataset...")
    
    try:
        dataset = CrazyflieILDataset(
            data_dir='imitation_data',
            image_size=(224, 224),
            normalize_states=True
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(dataset)}")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"\nSample shapes:")
        print(f"  Observation: {sample['observation'].shape}")
        print(f"  State: {sample['state'].shape}")
        print(f"  Action: {sample['action'].shape}")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch = next(iter(loader))
        
        print(f"\nBatch shapes:")
        print(f"  Observations: {batch['observation'].shape}")
        print(f"  States: {batch['state'].shape}")
        print(f"  Actions: {batch['action'].shape}")
        
        print("\n✓ DataLoader test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
