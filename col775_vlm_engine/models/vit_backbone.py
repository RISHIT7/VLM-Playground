import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 384, num_heads: int = 6, mlp_dim: int = 1536):
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
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        API Description:
        Executes the forward pass of a single Transformer block.
        
        Args:
            x: FloatTensor of shape (B, N, D) representing a batch of token embeddings.
            
        Returns:
            FloatTensor of shape (B, N, D) representing the output embeddings after passing through the Transformer block.
        """
        x = x + self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
        

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 384, depth: int = 12, num_heads: int = 6, mlp_dim: int = 1536):
        """
        API Description:
        Initializes the shared ViT backbone without using nn.Transformer. 
        Must instantiate patch embeddings (Conv2d is optimal), a learnable [CLS] token, 
        and 1D learnable positional embeddings. Instantiates `depth` layers of custom 
        Transformer blocks (Pre-Norm architecture).
        """
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.num_channels = 3
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size) # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim) where B is batch size
        # self.patch_embedding = nn.Linear(patch_size ** 2 * self.num_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) # (1, num_patches + 1, embed_dim)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # if isinstance(self.patch_embedding, nn.Linear):
        #     nn.init.trunc_normal_(self.patch_embedding.weight, std=0.02)
        #     if self.patch_embedding.bias is not None:
        #         nn.init.constant_(self.patch_embedding.bias, 0)

        if isinstance(self.patch_embedding, nn.Conv2d):
            nn.init.trunc_normal_(self.patch_embedding.weight, std=0.02)
            if self.patch_embedding.bias is not None:
                nn.init.constant_(self.patch_embedding.bias, 0)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        API Description:
        Dynamically resizes the positional embeddings for local crops of different resolutions.
        Temporarily separates the [CLS] token embedding, reshapes the remaining 1D spatial 
        sequence into a 2D grid, applies bicubic interpolation, and flattens it back before 
        adding it to the patch embeddings.

        Args:
            x: FloatTensor of shape (B, N, D) representing a batch of token embeddings.
            w: Integer representing the width of the image.
            h: Integer representing the height of the image.
            
        Returns:
            FloatTensor of shape (1, num_patches + 1, embed_dim) representing the interpolated positional embeddings.
        """
        num_patches = x.shape[1] - 1
        N = self.pos_embedding.shape[1] - 1
        if num_patches == N and w == h:
            return self.pos_embedding
        
        cls_emb = self.pos_embedding[:, 0:1, :]
        pos_tokens = self.pos_embedding[:, 1:, :]
        
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        w1 = int(math.sqrt(N))
        h1 = int(math.sqrt(N))

        # pos_tokens shape: (1, N, D)
        pos_tokens = pos_tokens.reshape(1, h1, w1, dim).permute(0, 3, 1, 2).contiguous()

        pos_embedding = nn.functional.interpolate(
            pos_tokens,
            size=(h0, w0),
            mode="bicubic",
            align_corners=False,
        )

        pos_embedding = pos_embedding.flatten(2).transpose(1, 2)
        pos_embedding = torch.cat([cls_emb, pos_embedding], dim=1)
        return pos_embedding

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        """
        API Description:
        Executes the forward pass of the Vision Transformer.
        
        Args:
            x: FloatTensor of shape (B, 3, img_size, img_size) representing a batch of images.
            return_patches: If True, returns the tensor of shape (B, N, D) excluding the [CLS] token.
            
        Returns:
            FloatTensor of shape (B, embed_dim) representing the unprojected [CLS] token,
            or if return_patches is True, shape (B, N, D) for the GAP over spatial patches.
        """
        B, C, H, W = x.shape
        
        x = self.patch_embedding(x) # (B, embed_dim, num_patches)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        
        cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1) # (B, num_patches + 1, embed_dim)
        
        x = x + self.interpolate_pos_encoding(x, W, H)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x) # (B, num_patches + 1, embed_dim)
        
        if return_patches:
            return x[:, 1:, :] # (B, num_patches, embed_dim)
        else:
            return x[:, 0, :] # (B, embed_dim)
        