import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List

class CLEVRDataset(Dataset):
    def __init__(self, config, mode: str, split: str = "train",
                 transform=None, tokenizer=None):
        """
        Args:
            config: EnvConfig instance
            mode: "clip", "dino", or "linear_probe"
            split: "train" or "val"
            transform: one of the transforms from transforms.py
            tokenizer: CLEVRTokenizer (required for mode="clip")
        """
        self.mode = mode
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer

        if mode in ["clip", "dino", "vae", "ldm"]:
            base_dir = getattr(config, 'base_dir_part_c', config.base_dir_part_a) if mode in ["vae", "ldm"] else config.base_dir_part_a
            self.image_dir = os.path.join(base_dir, split, "images")
            json_path = os.path.join(base_dir, split, f"clevr_{split}_captions.json")
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)
            if isinstance(self.annotations, dict) and "examples" in self.annotations:
                self.annotations = self.annotations["examples"]
            
            # For LDM, we might load precomputed latents and text embeddings
            if mode == "ldm":
                self.latent_dir = getattr(config, 'latent_dir', os.path.join(base_dir, split, "latents"))
                self.embed_dir = getattr(config, 'embed_dir', os.path.join(base_dir, split, "embeds"))
        else:
            base_dir = config.base_dir_part_aa
            self.image_dir = os.path.join(base_dir, "Clevr_official", "images", split)
            count_json = os.path.join(base_dir, "Probe-Datasets", f"clevr_count_{split}.json")
            color_json = os.path.join(base_dir, "Probe-Datasets", f"clevr_colors_{split}.json")
            caption_json = os.path.join(base_dir, f"clevr_{split}_captions.json")
            with open(count_json, 'r') as fc, open(color_json, 'r') as fp, open(caption_json, 'r') as fcap:
                count_examples = json.load(fc)["examples"]
                color_examples = json.load(fp)["examples"]
                caption_examples = json.load(fcap)
            self.counts = []
            self.color_sets = []
            self.annotations = []
            for count_ex, color_ex, cap_ex in zip(count_examples, color_examples, caption_examples):
                self.counts.append(count_ex["label"])
                self.color_sets.append(torch.tensor(color_ex["multi_hot"], dtype=torch.float32))
                self.annotations.append({"image_filename": count_ex["image_filename"], "caption": cap_ex["caption"]})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_filename = ann.get("image_filename", ann.get("image", ann.get("filename", "")))
        img_path = os.path.join(self.image_dir, img_filename)

        if self.mode == "ldm":
            # Load precomputed tensors
            latent_path = os.path.join(self.latent_dir, img_filename.replace('.png', '.pt').replace('.jpg', '.pt'))
            embed_path = os.path.join(self.embed_dir, img_filename.replace('.png', '.pt').replace('.jpg', '.pt'))
            
            latent = torch.load(latent_path) if os.path.exists(latent_path) else torch.zeros((4, 16, 16))
            text_embed = torch.load(embed_path) if os.path.exists(embed_path) else torch.zeros((77, 512))
            
            return {
                "latent": latent,
                "text_embed": text_embed
            }

        image = Image.open(img_path).convert("RGB")

        if self.mode == "clip":
            img_tensor = self.transform(image)
            caption = ann.get("caption", "")
            tokens, mask = self.tokenizer.encode(caption)
            return {
                "image": img_tensor,
                "tokens": tokens,
                "padding_mask": mask,
                "raw_caption": caption
            }
        elif self.mode == "dino":
            crops = self.transform(image)
            return crops
        elif self.mode == "linear_probe":
            img_tensor = self.transform(image)
            return {
                "image": img_tensor,
                "count_label": self.counts[idx],
                "color_label": self.color_sets[idx]
            }
        elif self.mode == "vae":
            img_tensor = self.transform(image)
            return {"image": img_tensor}

class CLEVRCollateFn:
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        if self.mode == "clip":
            images = torch.stack([b["image"] for b in batch])
            tokens = torch.stack([b["tokens"] for b in batch])
            masks = torch.stack([b["padding_mask"] for b in batch])
            raw_captions = [b["raw_caption"] for b in batch]
            return {
                "images": images,
                "tokens": tokens,
                "padding_mask": masks,
                "raw_captions": raw_captions
            }
        elif self.mode == "dino":
            global_crops = [b["global_crops"] for b in batch]
            local_crops = [b["local_crops"] for b in batch]
            global_stacked = torch.stack([torch.stack(crops) for crops in global_crops])
            local_stacked = torch.stack([torch.stack(crops) for crops in local_crops])
            return {
                "global_crops": global_stacked,
                "local_crops": local_stacked
            }
        elif self.mode == "linear_probe":
            images = torch.stack([b["image"] for b in batch])
            count_labels = torch.tensor([b["count_label"] for b in batch], dtype=torch.long)
            color_labels = torch.stack([b["color_label"] for b in batch])
            return {
                "images": images,
                "count_labels": count_labels,
                "color_labels": color_labels
            }
        elif self.mode == "vae":
            images = torch.stack([b["image"] for b in batch])
            return {"image": images}
        elif self.mode == "ldm":
            latents = torch.stack([b["latent"] for b in batch])
            text_embeds = torch.stack([b["text_embed"] for b in batch])
            return {
                "latent": latents,
                "text_embed": text_embeds
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")