"""
Download pretrained MobileNet weights for offline use.
Run this on a machine with internet access, then transfer to cluster.
"""

import torch
from torchvision.models import mobilenet_v2, mobilenet_v3_small
from pathlib import Path

# Create directory for pretrained weights
weights_dir = Path(__file__).parent.parent / "pretrained_weights"
weights_dir.mkdir(exist_ok=True)

print("Downloading MobileNetV2 pretrained weights...")
model_v2 = mobilenet_v2(weights="IMAGENET1K_V1")
v2_path = weights_dir / "mobilenet_v2_imagenet.pth"
torch.save(model_v2.state_dict(), v2_path)
print(f"Saved to: {v2_path}")

print("\nDownloading MobileNetV3-Small pretrained weights...")
model_v3 = mobilenet_v3_small(weights="IMAGENET1K_V1")
v3_path = weights_dir / "mobilenet_v3_small_imagenet.pth"
torch.save(model_v3.state_dict(), v3_path)
print(f"Saved to: {v3_path}")

print("\n✓ Done! Now transfer these files to the cluster:")
print(f"  scp {weights_dir}/*.pth username@della.princeton.edu:/home/de7281/MAE345/final_project/drone/pretrained_weights/")
