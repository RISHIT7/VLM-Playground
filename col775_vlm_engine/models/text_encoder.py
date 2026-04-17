import torch
import math
import torch.nn as nn
from .transformer_block import TransformerBlock

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 384, depth: int = 6, num_heads: int = 6, mlp_dim: int = 1536, max_len: int=77):
        """
        Implements the 6-layer text transformer. 
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        pe = self._build_sincos_embeddings(max_len, embed_dim) # using sin cos embedding as per Vaswani et. al
        self.register_buffer("pos_embeddings", pe)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, non_linearity="relu") for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def _build_sincos_embeddings(self, max_len: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # (1, max_len, embed_dim)

    def forward(self, tokens: torch.LongTensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L)
            padding_mask: (B, L) - Critical for ensuring padded tokens do not corrupt attention.
            
        Returns:
            The feature vector from the [EOS] (End of Sequence) token, or the max-pooled 
            sequence, of shape (B, embed_dim).
        """
        eos_token_id = tokens.argmax(dim=-1)

        x = self.token_embedding(tokens) + self.pos_embeddings[:, :tokens.size(1), :]
        for block in self.transformer_blocks:
            x = block(x, padding_mask)
        x = self.norm(x) # (B, L, embed_dim)
        
        x = x[torch.arange(x.shape[0]), eos_token_id] # (B, embed_dim)
        return x
