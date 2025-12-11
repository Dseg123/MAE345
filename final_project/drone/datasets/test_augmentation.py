"""
Test script to verify coupled augmentation works correctly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import the custom augmentation
from drone.datasets.dataloader import CoupledAugmentation
import torchvision.transforms.functional as TF


def test_horizontal_flip():
    """Test that horizontal flip negates vy and yaw_rate."""
    print("Testing Horizontal Flip (left-right)...")
    print("=" * 60)

    # Create a simple test image (with asymmetry so we can see the flip)
    img = Image.new('RGB', (100, 100), color='white')
    # Draw a red square on the left side
    pixels = img.load()
    for i in range(20, 40):
        for j in range(10, 30):
            pixels[j, i] = (255, 0, 0)

    # Original action: moving right (vy=0.2) with positive yaw
    original_action = torch.tensor([0.1, 0.2, 0.0, 0.15])  # [vx, vy, vz, yaw_rate]

    # Apply horizontal flip (100% probability)
    aug = CoupledAugmentation(horizontal_flip_prob=1.0, vertical_flip_prob=0.0)
    flipped_img, flipped_action = aug(img, original_action)

    print(f"Original action: {original_action.numpy()}")
    print(f"Flipped action:  {flipped_action.numpy()}")
    print(f"\nChanges:")
    print(f"  vx (forward):   {original_action[0]:.3f} -> {flipped_action[0]:.3f} (unchanged ✓)")
    print(f"  vy (lateral):   {original_action[1]:.3f} -> {flipped_action[1]:.3f} (negated ✓)")
    print(f"  vz (vertical):  {original_action[2]:.3f} -> {flipped_action[2]:.3f} (unchanged ✓)")
    print(f"  yaw_rate:       {original_action[3]:.3f} -> {flipped_action[3]:.3f} (negated ✓)")

    # Verify
    assert flipped_action[0] == original_action[0], "vx should be unchanged"
    assert flipped_action[1] == -original_action[1], "vy should be negated"
    assert flipped_action[2] == original_action[2], "vz should be unchanged"
    assert flipped_action[3] == -original_action[3], "yaw_rate should be negated"

    print("\n✓ Horizontal flip test PASSED!\n")
    return img, flipped_img


def test_vertical_flip():
    """Test that vertical flip negates vz."""
    print("Testing Vertical Flip (top-bottom)...")
    print("=" * 60)

    # Create a simple test image (with asymmetry so we can see the flip)
    img = Image.new('RGB', (100, 100), color='white')
    # Draw a blue square on the top
    pixels = img.load()
    for i in range(10, 30):
        for j in range(20, 40):
            pixels[j, i] = (0, 0, 255)

    # Original action: moving up (vz=0.15) with right lateral movement
    original_action = torch.tensor([0.1, 0.2, 0.15, 0.0])  # [vx, vy, vz, yaw_rate]

    # Apply vertical flip (100% probability)
    aug = CoupledAugmentation(horizontal_flip_prob=0.0, vertical_flip_prob=1.0)
    flipped_img, flipped_action = aug(img, original_action)

    print(f"Original action: {original_action.numpy()}")
    print(f"Flipped action:  {flipped_action.numpy()}")
    print(f"\nChanges:")
    print(f"  vx (forward):   {original_action[0]:.3f} -> {flipped_action[0]:.3f} (unchanged ✓)")
    print(f"  vy (lateral):   {original_action[1]:.3f} -> {flipped_action[1]:.3f} (unchanged ✓)")
    print(f"  vz (vertical):  {original_action[2]:.3f} -> {flipped_action[2]:.3f} (negated ✓)")
    print(f"  yaw_rate:       {original_action[3]:.3f} -> {flipped_action[3]:.3f} (unchanged ✓)")

    # Verify
    assert flipped_action[0] == original_action[0], "vx should be unchanged"
    assert flipped_action[1] == original_action[1], "vy should be unchanged"
    assert flipped_action[2] == -original_action[2], "vz should be negated"
    assert flipped_action[3] == original_action[3], "yaw_rate should be unchanged"

    print("\n✓ Vertical flip test PASSED!\n")
    return img, flipped_img


def test_both_flips():
    """Test that both flips work together."""
    print("Testing Both Flips Together...")
    print("=" * 60)

    img = Image.new('RGB', (100, 100), color='white')
    original_action = torch.tensor([0.1, 0.2, 0.15, 0.1])  # [vx, vy, vz, yaw_rate]

    # Apply both flips (100% probability)
    aug = CoupledAugmentation(horizontal_flip_prob=1.0, vertical_flip_prob=1.0)
    flipped_img, flipped_action = aug(img, original_action)

    print(f"Original action: {original_action.numpy()}")
    print(f"Flipped action:  {flipped_action.numpy()}")
    print(f"\nExpected: vy, vz, and yaw_rate all negated")

    # Verify
    assert flipped_action[0] == original_action[0], "vx should be unchanged"
    assert flipped_action[1] == -original_action[1], "vy should be negated"
    assert flipped_action[2] == -original_action[2], "vz should be negated"
    assert flipped_action[3] == -original_action[3], "yaw_rate should be negated"

    print("✓ Both flips test PASSED!\n")


def test_probabilistic():
    """Test that probabilistic flipping works."""
    print("Testing Probabilistic Flipping...")
    print("=" * 60)

    img = Image.new('RGB', (100, 100), color='white')
    original_action = torch.tensor([0.1, 0.2, 0.0, 0.0])

    aug = CoupledAugmentation(horizontal_flip_prob=0.5, vertical_flip_prob=0.5)

    # Run 100 times and count flips
    h_flips = 0
    v_flips = 0

    for _ in range(100):
        _, flipped_action = aug(img, original_action)
        if flipped_action[1] < 0:  # vy was negated
            h_flips += 1
        if flipped_action[2] < 0:  # vz was negated
            v_flips += 1

    print(f"Horizontal flips: {h_flips}/100 (expected ~50)")
    print(f"Vertical flips:   {v_flips}/100 (expected ~50)")

    # Allow some variance (30-70 range is reasonable for 100 samples at 50% probability)
    assert 30 <= h_flips <= 70, "Horizontal flip probability seems wrong"
    assert 30 <= v_flips <= 70, "Vertical flip probability seems wrong"

    print("✓ Probabilistic test PASSED!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING COUPLED AUGMENTATION")
    print("=" * 60 + "\n")

    # Run all tests
    test_horizontal_flip()
    test_vertical_flip()
    test_both_flips()
    test_probabilistic()

    print("=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nThe coupled augmentation is working correctly:")
    print("  - Horizontal flip: negates vy (lateral) and yaw_rate")
    print("  - Vertical flip: negates vz (vertical)")
    print("  - Both can be applied independently with specified probabilities")
