import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm

class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 384, hidden_dim: int = 2048, bottleneck_dim: int = 256, out_dim: int = 4096, use_bn: bool = False):
        """
        API Description:
        The exact projection head from Caron et al., 2021.
        Features a 3-layer MLP. Supports dynamic insertion of BatchNorm1d for ablation studies.
        The final output is L2-normalized, followed by a Weight-Normalized linear layer without biases.
        """
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original1.data.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)        
        x = self.last_layer(x)
        return x


class LinearAblationHead(nn.Module):
    def __init__(self, in_dim: int = 384, out_dim: int = 4096):
        """
        API Description:
        A strictly linear projection. Guarantees dimensional collapse.
        """
        super().__init__()
        self.projection_head = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(x)


class UnnormalizedMLPHead(nn.Module):
    def __init__(self, in_dim: int = 384, hidden_dim: int = 2048, bottleneck_dim: int = 256, out_dim: int = 4096, use_bn: bool = False):
        """
        API Description:
        A 3-layer MLP identical to the Canonical DINO head, BUT lacking the final L2 
        normalization and Weight-Normalized linear layer. Proves temperature instability.
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, out_dim)) # Directly to out_dim
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AsymmetricPredictor(nn.Module):
    def __init__(self, out_dim: int = 4096, hidden_dim: int = 512):
        """
        API Description:
        An extra 2-layer MLP applied ONLY to the Student network.
        Uses ReLU as per standard BYOL/SimSiam implementations.
        """
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)