import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 384, depth: int = 6, num_heads: int = 6, mlp_dim: int = 1536):
        """
        Implements the 6-layer text transformer. 
        Must include a token embedding layer and learnable 1D positional embeddings.
        """
        pass

    def forward(self, tokens: torch.LongTensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L)
            padding_mask: (B, L) - Critical for ensuring padded tokens do not corrupt attention.
            
        Returns:
            The feature vector from the [EOS] (End of Sequence) token, or the max-pooled 
            sequence, of shape (B, embed_dim).
        """
        pass
