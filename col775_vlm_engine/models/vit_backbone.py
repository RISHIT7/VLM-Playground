class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 384, depth: int = 12, num_heads: int = 6):
        """
        API Description:
        Initializes the shared ViT backbone without using nn.Transformer. 
        Must instantiate patch embeddings (Conv2d is optimal), a learnable [CLS] token, 
        and 1D learnable positional embeddings. Instantiates `depth` layers of custom 
        Transformer blocks (Pre-Norm architecture).
        """
        pass

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        API Description:
        Dynamically resizes the positional embeddings for local crops of different resolutions.
        Temporarily separates the [CLS] token embedding, reshapes the remaining 1D spatial 
        sequence into a 2D grid, applies bicubic interpolation, and flattens it back before 
        adding it to the patch embeddings.
        """
        pass

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
        pass