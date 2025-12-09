import torch.nn as nn
from torchvision.models import mobilenet_v2


class ContinuousActionModel(nn.Module):
    """MobileNetV2-based regressor for continuous control.

    Args:
        pretrained (bool): If True, load ImageNet pretrained weights for the
            MobileNetV2 backbone. Defaults to False.
        action_dim (int): Number of continuous outputs (e.g., 4 for
            [roll, pitch, yaw, thrust]). Defaults to 4.

    Attributes:
        backbone (nn.Module): The MobileNetV2 model with its classifier
            replaced by a dropout + linear layer producing `action_dim`
            outputs.
    """

    def __init__(self, pretrained: bool = False, action_dim: int = 4):
        super().__init__()
        # Use the torchvision MobileNetV2 constructor with the modern weights
        # enum when requesting pretrained weights.
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        in_feature_dim = self.backbone.classifier[1].in_features

        # Replace the classifier with a small head that outputs continuous
        # actions. Dropout is kept to match typical MobileNet configs.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feature_dim, action_dim),  # continuous control outputs
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw continuous outputs of shape (B, action_dim).
        """
        return self.backbone(x)

    def output_to_executable_actions(self, output):
        """Convert model outputs to a CPU NumPy action vector.

        This helper moves the tensor to CPU (if necessary), converts to NumPy,
        and flattens the result to a 1D array. It is intended for quick
        integration with downstream controllers that expect a simple numeric
        array. The caller is responsible for clamping / scaling actions if
        required by the vehicle's actuator ranges.

        Args:
            output (torch.Tensor): Model output tensor (B, action_dim) or
                (action_dim,) for a single example.

        Returns:
            numpy.ndarray: 1D array of length B*action_dim or action_dim
                (for a single example) containing continuous actions.
        """
        # Directly return the continuous actions as a flattened NumPy array.
        return output.cpu().numpy().flatten()