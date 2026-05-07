import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 384, num_heads: int = 6, mlp_dim: int = 1536, non_linearity: str = "gelu"):
        """
        API Description:
        Initializes a single Transformer block (Pre-Norm architecture).
        
        Args:
            embed_dim: Dimension of the input and output embeddings.
            num_heads: Number of attention heads.
            mlp_dim: Dimension of the MLP hidden layer.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        if non_linearity == "gelu":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim)
            )
        elif non_linearity == "relu":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, embed_dim)
            )

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor=None) -> torch.Tensor:
        """
        API Description:
        Executes the forward pass of a single Transformer block.
        
        Args:
            x: FloatTensor of shape (B, N, D) representing a batch of token embeddings.
            padding_mask: Boolean tensor of shape (B, N) representing the padding mask.

        Returns:
            FloatTensor of shape (B, N, D) representing the output embeddings after passing through the Transformer block.
        """
        if padding_mask is not None:
            x = x + self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=padding_mask)[0]
        else:
            x = x + self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
        