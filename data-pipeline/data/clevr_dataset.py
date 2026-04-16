import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple

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

        # Determine base directory and load annotations
        if mode in ["clip", "dino"]:
            base_dir = config.base_dir_part_a
        else:
            base_dir = config.base_dir_part_aa
        self.image_dir = os.path.join(base_dir, "images")
        json_path = os.path.join(base_dir, f"{split}.json")
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)

        # For linear_probe, pre-extract labels
        if mode == "linear_probe":
            self.counts = []
            self.color_sets = []
            for ann in self.annotations:
                # Adjust keys according to actual JSON structure
                count = ann.get("count", ann.get("object_count", 0))
                color_set = ann.get("color_set", ann.get("color_vector", []))
                # If color_set is a list of strings, convert to multi-hot (example)
                if isinstance(color_set, list) and len(color_set) > 0 and isinstance(color_set[0], str):
                    # This is a placeholder – in real CLEVR you'd have a fixed color order
                    # We'll assume the JSON already provides a binary vector
                    raise NotImplementedError("Please preprocess color_set to a binary vector.")
                self.counts.append(count)
                self.color_sets.append(torch.tensor(color_set, dtype=torch.float32))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_filename = ann.get("image", ann.get("filename", ""))
        img_path = os.path.join(self.image_dir, img_filename)
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
            crops = self.transform(image)   # returns dict with global_crops, local_crops
            return crops
        elif self.mode == "linear_probe":
            img_tensor = self.transform(image)
            return {
                "image": img_tensor,
                "count_label": self.counts[idx],
                "color_label": self.color_sets[idx]
            }

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
            # batch: list of dicts each with "global_crops" (list of 2 tensors)
            # and "local_crops" (list of 8 tensors)
            global_crops = [b["global_crops"] for b in batch]   # each is list of 2
            local_crops = [b["local_crops"] for b in batch]     # each is list of 8
            # Stack into (B, 2, 3, 224, 224) and (B, 8, 3, 96, 96)
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
        else:
            raise ValueError(f"Unknown mode: {self.mode}")