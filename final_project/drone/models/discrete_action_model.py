import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small
from typing import Optional


# Training configuration
num_bins = 11
action_dim = 4
action_low = -1.0   # ASSUMPTION: action continuous range lower bound
action_high = 1.0   # ASSUMPTION: action continuous range upper bound



class DiscreteActionModel(nn.Module):
    """MobileNetV2 backbone producing discretized action logits.

    This model predicts discrete motion by outputting logits for a fixed number
    of bins per action channel. The final linear layer outputs `action_dim * num_bins`
    values which are reshaped to (batch, action_dim, num_bins).

    Args:
        pretrained: whether to load ImageNet weights for the backbone.
        action_dim: number of action channels (e.g. 4).
        num_bins: number of discrete bins per action.

    Forward:
        If `return_probs` is False (default) returns logits with shape
        (batch, action_dim, num_bins). If `return_probs` is True returns
        softmax probabilities along the last (bin) dimension.
    """

    def __init__(self,
                 pretrained: bool = False,
                 action_dim: int = 4,
                 num_bins: int = 11,
                 action_low: Optional[float] = -1.0,
                 action_high: Optional[float] = 1.0,
                 v3: bool = False
                 ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_bins = int(num_bins)
        self.action_low = action_low
        self.action_high = action_high

        # Load backbone. When pretrained=False this will init random weights.
        if v3:
            # MobileNetV3 Small
            self.backbone = mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
            # MobileNetV3 classifier: [0] Linear(576->1024), [1] Hardswish, [2] Dropout, [3] Linear(1024->1000)
            # Get input features from the FIRST Linear layer (feature extractor output)
            in_feature_dim = self.backbone.classifier[0].in_features  # 576
        else:
            # MobileNetV2
            self.backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            # MobileNetV2 classifier: [0] Dropout, [1] Linear
            in_feature_dim = self.backbone.classifier[1].in_features  # 1280

        out_features = self.action_dim * self.num_bins
        # Replace classifier with a dropout + linear that outputs logits for all bins
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feature_dim, out_features),
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: images tensor of shape (batch, 3, H, W).
            return_probs: if True, return softmax probabilities over bins.

        Returns:
            logits of shape (batch, action_dim, num_bins) or probabilities if
            `return_probs` is True.
        """
        logits = self.backbone(x)
        # logits: (batch, action_dim * num_bins) -> reshape
        batch = logits.shape[0]
        logits = logits.view(batch, self.action_dim, self.num_bins)

        if return_probs:
            return torch.softmax(logits, dim=-1)

        return logits
    
    def continuous_to_bins(self, 
                           actions: torch.Tensor,
                           ) -> torch.Tensor:
        """Convert continuous actions to integer bin indices.

        Args:
            actions: (B, A) tensor of continuous actions.
            low: minimum action value (clamp lower bound).
            high: maximum action value (clamp upper bound).
            num_bins: number of discrete bins.

        Returns:
            (B, A) LongTensor of bin indices in [0, num_bins-1].
        """
        # clamp and normalize to [0,1]
        clipped = actions.clamp(min=self.action_low, max=self.action_high)
        denom = (self.action_high - self.action_low) if (self.action_high - self.action_low) != 0 else 1.0
        norm = (clipped - self.action_low) / denom
        # map to [0, num_bins-1] and round to nearest bin
        bins = (norm * (num_bins - 1)).round().long()
        return bins
    
    def bins_to_continuous(self,
                           bins: torch.Tensor,
                           ) -> torch.Tensor:
        """Convert integer bin indices to continuous actions.

        Args:
            bins: (B, A) LongTensor of bin indices in [0, num_bins-1].
            low: minimum action value.
            high: maximum action value.
            num_bins: number of discrete bins.

        Returns:
            (B, A) tensor of continuous actions.
        """
        # normalize bin indices to [0, 1]
        norm = (bins + 0.5) / (self.num_bins)
        # scale to [action_low, action_high]
        continuous = norm * (self.action_high - self.action_low) + self.action_low
        return continuous
    
    def output_to_executable_actions(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output logits to executable continuous actions.

        Args:
            output: (B, action_dim, num_bins) tensor of logits.

        Returns:
            (B, action_dim) tensor of continuous actions.
        """
        # find the bin with maximum logit for each action dimension
        bins = torch.argmax(output, dim=-1)  # (B, action_dim)
        # convert bin indices to continuous actions
        continuous = self.bins_to_continuous(bins)
        return continuous