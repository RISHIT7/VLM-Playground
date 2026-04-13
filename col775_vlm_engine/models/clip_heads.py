class CLIPEngine(nn.Module):
    def __init__(self, vit_backbone: nn.Module, text_encoder: nn.Module, embed_dim: int = 384, proj_dim: int = 512):
        """
        API Description:
        Initializes the visual and textual towers. Instantiates linear projection heads 
        to map both modalities to `proj_dim`. Initializes `logit_scale` parameter.
        """
        pass

    def forward(self, images: torch.Tensor, tokens: torch.LongTensor, padding_mask: torch.BoolTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        API Description:
        Passes modalities through their respective backbones and projection heads.
        CRITICAL: Must L2-normalize the resulting embeddings before returning.
        
        Args:
            images: Shape (B, 3, 224, 224)
            tokens: Shape (B, L)
            padding_mask: Shape (B, L)
            
        Returns:
            Tuple of (image_features, text_features), both of shape (B, proj_dim).
        """
        pass

    def compute_loss(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, raw_captions: list[str]) -> torch.Tensor:
        """
        API Description:
        Computes the symmetric contrastive loss. Dynamically constructs a binary target 
        matrix T using `raw_captions` to prevent penalization of identical CLEVR captions 
        within the same batch (Soft InfoNCE).
        
        Returns:
            Scalar loss tensor.
        """
        pass
        
    def get_optim_groups(self, weight_decay: float = 1e-3) -> list[dict]:
        """
        API Description:
        Separates parameters into two groups to enforce proper regularization. 
        Returns a list of dictionaries for the AdamW optimizer ensuring weight_decay 
        is applied to weights, but strictly set to 0.0 for all LayerNorm and Bias terms.
        """
        pass