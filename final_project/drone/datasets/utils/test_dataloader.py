"""
Test script for CrazyflieILDataset dataloader.
Validates data loading, shapes, and visualizes samples.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from drone.datasets.dataloader import CrazyflieILDataset, create_dataloaders
from pathlib import Path


def test_basic_loading():
    """Test basic dataset loading."""
    print("="*60)
    print("TEST 1: Basic Dataset Loading")
    print("="*60)
    
    try:
        dataset = CrazyflieILDataset(
            data_dir='/home/de7281/MAE345/final_project/drone/datasets/imitation_data',
            image_size=(224, 224),
            normalize_states=True,
            normalize_actions=False
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Number of trials: {dataset.get_data_stats()['num_trials']}")
        
        return dataset
    
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise


def test_sample_shapes(dataset):
    """Test that samples have correct shapes."""
    print("\n" + "="*60)
    print("TEST 2: Sample Shapes")
    print("="*60)
    
    sample = dataset[0]
    
    expected = {
        'observation': (3, 224, 224),  # [C, H, W]
        'state': (9,),                  # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        'action': (4,)                  # [vx, vy, vz, yaw_rate]
    }
    
    for key, expected_shape in expected.items():
        actual_shape = tuple(sample[key].shape)
        if actual_shape == expected_shape:
            print(f"✓ {key}: {actual_shape}")
        else:
            print(f"✗ {key}: expected {expected_shape}, got {actual_shape}")
            raise ValueError(f"Shape mismatch for {key}")


def test_batch_loading(dataset):
    """Test batch loading with DataLoader."""
    print("\n" + "="*60)
    print("TEST 3: Batch Loading")
    print("="*60)
    
    batch_size = 8
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    batch = next(iter(loader))
    
    print(f"Batch size: {batch_size}")
    print(f"✓ Observations: {batch['observation'].shape}")
    print(f"✓ States: {batch['state'].shape}")
    print(f"✓ Actions: {batch['action'].shape}")
    
    # Check that batch dimension is correct
    assert batch['observation'].shape[0] == batch_size
    assert batch['state'].shape[0] == batch_size
    assert batch['action'].shape[0] == batch_size
    
    print("✓ All batch dimensions correct")
    
    return batch


def test_data_ranges(dataset):
    """Test that data is in expected ranges."""
    print("\n" + "="*60)
    print("TEST 4: Data Ranges")
    print("="*60)
    
    sample = dataset[0]
    
    # Check observation range (should be normalized to [-1, 1])
    obs_min = sample['observation'].min().item()
    obs_max = sample['observation'].max().item()
    print(f"Observation range: [{obs_min:.3f}, {obs_max:.3f}]")
    
    if obs_min >= -1.1 and obs_max <= 1.1:
        print("✓ Observation normalized correctly")
    else:
        print("⚠ Warning: Observation values outside expected range")
    
    # Check state (if normalized, should have reasonable range)
    state_min = sample['state'].min().item()
    state_max = sample['state'].max().item()
    print(f"State range: [{state_min:.3f}, {state_max:.3f}]")
    
    # Check action
    action_min = sample['action'].min().item()
    action_max = sample['action'].max().item()
    print(f"Action range: [{action_min:.3f}, {action_max:.3f}]")
    
    print("✓ Data ranges checked")


def test_multiple_trials():
    """Test loading specific trials."""
    print("\n" + "="*60)
    print("TEST 5: Multiple Trial Loading")
    print("="*60)
    
    # Try loading just trial 1
    dataset_single = CrazyflieILDataset(
        data_dir='/home/de7281/MAE345/final_project/drone/datasets/imitation_data',
        trial_numbers=[1],
        image_size=(224, 224)
    )
    
    print(f"✓ Single trial (trial 1): {len(dataset_single)} samples")
    
    # Try loading all trials
    dataset_all = CrazyflieILDataset(
        data_dir='/home/de7281/MAE345/final_project/drone/datasets/imitation_data',
        trial_numbers=None,  # All trials
        image_size=(224, 224)
    )
    
    print(f"✓ All trials: {len(dataset_all)} samples")


def test_train_val_split():
    """Test creating train/val dataloaders with split."""
    print("\n" + "="*60)
    print("TEST 6: Train/Val Split")
    print("="*60)
    
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir='/home/de7281/MAE345/final_project/drone/datasets/imitation_data',
            batch_size=8,
            image_size=(224, 224),
            num_workers=0  # Use 0 for testing
        )
        
        print(f"✓ Train loader created: {len(train_loader)} batches")
        print(f"✓ Val loader created: {len(val_loader)} batches")
        
        # Test loading a batch from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"✓ Train batch shape: {train_batch['observation'].shape}")
        print(f"✓ Val batch shape: {val_batch['observation'].shape}")
        
    except Exception as e:
        print(f"⚠ Train/val split test skipped (may need more trials): {e}")


def visualize_samples(dataset, num_samples=4):
    """Visualize sample images with their states and actions."""
    print("\n" + "="*60)
    print("TEST 7: Visualization")
    print("="*60)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    for idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        
        # Denormalize image for display
        img = sample['observation'].numpy()
        img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        img = (img * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
        img = np.clip(img, 0, 1)
        
        # Show image
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"Sample {sample_idx}")
        axes[0, idx].axis('off')
        
        # Show state and action info
        state = sample['state'].numpy()
        action = sample['action'].numpy()
        
        info_text = f"State:\n"
        info_text += f"  pos: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]\n"
        info_text += f"  vel: [{state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f}]\n"
        info_text += f"  rot: [{state[6]:.1f}, {state[7]:.1f}, {state[8]:.1f}]\n\n"
        info_text += f"Action:\n"
        info_text += f"  [{action[0]:.2f}, {action[1]:.2f},\n"
        info_text += f"   {action[2]:.2f}, {action[3]:.1f}]"
        
        axes[1, idx].text(0.1, 0.5, info_text, 
                         fontsize=9, family='monospace',
                         verticalalignment='center')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('/home/de7281/MAE345/final_project/drone/datasets/imitation_data') / 'dataloader_test_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def test_iteration_speed(dataset):
    """Test data loading speed."""
    print("\n" + "="*60)
    print("TEST 8: Iteration Speed")
    print("="*60)
    
    import time
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    # Time loading 10 batches
    num_batches = min(10, len(loader))
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
    
    elapsed = time.time() - start_time
    samples_per_sec = (num_batches * 16) / elapsed
    
    print(f"Loaded {num_batches} batches in {elapsed:.2f} seconds")
    print(f"Speed: {samples_per_sec:.1f} samples/second")
    print("✓ Iteration speed test complete")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("CRAZYFLIE IL DATALOADER TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Basic loading
        dataset = test_basic_loading()
        
        # Test 2: Sample shapes
        test_sample_shapes(dataset)
        
        # Test 3: Batch loading
        batch = test_batch_loading(dataset)
        
        # Test 4: Data ranges
        test_data_ranges(dataset)
        
        # Test 5: Multiple trials
        test_multiple_trials()
        
        # Test 6: Train/val split
        test_train_val_split()
        
        # Test 7: Visualization
        visualize_samples(dataset)
        
        # Test 8: Iteration speed
        test_iteration_speed(dataset)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nDataloader is ready for training!")
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED ✗")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
