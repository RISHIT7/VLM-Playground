import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 384, num_classes: int = 10, multi_label: bool = False):
        """
        A strictly linear evaluation head (no hidden layers, no activations).
        """
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)
        self.multi_label = multi_label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input 'x' will be the frozen GAP over Patches from the ViT backbone.
        """
        return self.head(x)
