#!/usr/bin/env python
"""Minimal test to debug Hydra initialization."""

import os
import sys
from pathlib import Path

print("Step 1: Script started")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Step 2: Added {project_root} to sys.path")

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    print("Step 3: Hydra imports successful")
except Exception as e:
    print(f"Step 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from drone.datasets.dataloader import create_dataloaders
    print("Step 4: Dataloader import successful")
except Exception as e:
    print(f"Step 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def test_main(cfg: DictConfig):
    print("Step 5: Inside Hydra decorated function")
    print(f"Working directory after Hydra: {os.getcwd()}")
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))
    print("Step 6: Test complete!")

if __name__ == "__main__":
    print("Step 7: About to call Hydra main")
    test_main()
