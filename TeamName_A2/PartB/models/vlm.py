import torch
import torch.nn as nn
from transformers import PreTrainedModel


class VLMProjector(nn.Module):
    def __init__(
        self, vit_dim: int = 384, expansion_factor: int = 2, llm_dim: int = 2560
    ):
        super().__init__()
        hidden_dim = llm_dim * expansion_factor
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VLMModel(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        llm: PreTrainedModel,
        img_placeholder_id: int,
        expansion_factor: int = 2,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.img_placeholder_id = img_placeholder_id

        llm_dim = llm.config.hidden_size
        vit_dim = 384

        self.projector = VLMProjector(
            vit_dim=vit_dim, llm_dim=llm_dim, expansion_factor=expansion_factor
        )

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def extract_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            patch_tokens = self.vision_encoder(images, return_patches=True)

        projected_visuals = self.projector(patch_tokens)
        return projected_visuals

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        visual_embeds = self.extract_visual_features(images)
        img_mask = input_ids == self.img_placeholder_id
        lookup_ids = input_ids.clone()
        lookup_ids[img_mask] = 0

        text_embeds = self.llm.get_input_embeddings()(lookup_ids)

        inputs_embeds = text_embeds.to(visual_embeds.dtype).clone()

        inputs_embeds[img_mask] = visual_embeds.reshape(-1, visual_embeds.shape[-1])

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        visual_embeds = self.extract_visual_features(images)
        img_mask = input_ids == self.img_placeholder_id
        lookup_ids = input_ids.clone()
        lookup_ids[img_mask] = 0
        text_embeds = self.llm.get_input_embeddings()(lookup_ids)
        inputs_embeds = text_embeds.to(visual_embeds.dtype).clone()
        inputs_embeds[img_mask] = visual_embeds.reshape(-1, visual_embeds.shape[-1])

        return self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
