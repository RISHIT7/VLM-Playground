import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CLIPEngine(nn.Module):
    def __init__(self, vit_backbone: nn.Module, text_encoder: nn.Module, embed_dim: int = 384, proj_dim: int = 512):
        """
        API Description:
        Initializes the visual and textual towers. Instantiates linear projection heads 
        to map both modalities to `proj_dim`. Initializes `logit_scale` parameter.
        """
        super().__init__()

        self.vit_backbone = vit_backbone
        self.text_encoder = text_encoder
        self.image_projection = nn.Linear(embed_dim, proj_dim)
        self.text_projection = nn.Linear(embed_dim, proj_dim)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

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
        img_feat = self.vit_backbone(images)
        txt_feat = self.text_encoder(tokens, padding_mask)

        img_feat = self.image_projection(img_feat)
        txt_feat = self.text_projection(txt_feat)
        
        img_feat = F.normalize(img_feat, p=2, dim=1)
        txt_feat = F.normalize(txt_feat, p=2, dim=1)
        
        return img_feat, txt_feat

    def compute_loss(self, img_feat: torch.Tensor, txt_feat: torch.Tensor, raw_captions: list[str]) -> torch.Tensor:
        """
        API Description:
        Computes the symmetric contrastive loss. Dynamically constructs a binary target 
        matrix T using `raw_captions` to prevent penalization of identical CLEVR captions 
        within the same batch (Soft InfoNCE).
        
        Returns:
            Scalar loss tensor.
        """
        device = img_feat.device

        logit_scale = self.logit_scale.clamp(max=math.log(100)).exp()
        logits_per_image = logit_scale * (img_feat @ txt_feat.T)
        logits_per_text = logits_per_image.T

        target = torch.tensor(
            [[c1 == c2 for c2 in raw_captions] for c1 in raw_captions],
            dtype=img_feat.dtype,
            device=device
        )

        target_img = target / target.sum(dim=1, keepdim=True)
        target_txt = target.T / target.T.sum(dim=1, keepdim=True)

        # If Pytorch 1.10+
        loss_img = F.cross_entropy(logits_per_image, target_img)
        loss_txt = F.cross_entropy(logits_per_text, target_txt)

        # If Pytorch < 1.10
        # loss_img = -torch.sum(F.log_softmax(logits_per_image, dim=1) * target_img, dim=1).mean()
        # loss_txt = -torch.sum(F.log_softmax(logits_per_text, dim=1) * target_txt, dim=1).mean()

        return (loss_img + loss_txt)/2.0
        
    def get_optim_groups(self, weight_decay: float = 1e-3) -> list[dict]:
        """
        API Description:
        Separates parameters into two groups to enforce proper regularization. 
        Returns a list of dictionaries for the AdamW optimizer ensuring weight_decay 
        is applied to weights, but strictly set to 0.0 for all LayerNorm and Bias terms.
        """
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias"): # ndim <= 1 removes the gains as well
                no_decay.append(param)
            else:
                decay.append(param)
        
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]