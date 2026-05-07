import torch
from torch.utils.data import Dataset
import json
import os
from typing import Any, Dict, List
import random
from PIL import Image

class CLEVRCaptionDataset(Dataset):
    def __init__(self, config: 'VLMConfig', split: str = "train", transform=None, tokenizer=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.base_dir = self.config.data_root
        self.json_path = os.path.join(self.base_dir, "Part_Aa", f"clevr_{split}_captions.json")
        self.image_dir = os.path.join(self.base_dir, "Part_Aa", "Clevr_official", "images", split)
        
        with open(self.json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image_filename"])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            fallback_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = fallback_transform(image)
            
        return {
            "image": image,
            "caption": item["caption"],
            "image_filename": item["image_filename"]
        }

class CLEVRQADataset(Dataset):
    def __init__(self, config: 'VLMConfig', split: str = "train", transform=None, tokenizer=None):
        """
        API Description:
        Dataset for Stage-2: Visual Instruction Tuning.
        Loads QA pairs from Part_Aa/Clevr_official/questions/CLEVR_{split}_questions.json.

        CRITICAL: Must extract the 'program' field to form the Chain-of-Thought (CoT)
        factual explanation before appending the final answer.
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.base_dir = self.config.data_root
        self.questions_path = os.path.join(
            self.base_dir,
            "Part_Aa",
            "Clevr_official",
            "questions",
            f"CLEVR_{split}_questions.json",
        )
        self.image_dir = os.path.join(self.base_dir, "Part_Aa", "Clevr_official", "images", split)

        with open(self.questions_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.data = payload["questions"] if isinstance(payload, dict) and "questions" in payload else payload

    @staticmethod
    def _format_program_step(step: Dict[str, Any]) -> str:
        function = str(step.get("function", "unknown"))
        value_inputs = step.get("value_inputs", []) or []

        if function == "scene":
            return "inspect the full scene"
        if function == "filter_color" and value_inputs:
            return f"keep objects that are {value_inputs[0]}"
        if function == "filter_size" and value_inputs:
            return f"keep objects that are {value_inputs[0]}"
        if function == "filter_material" and value_inputs:
            return f"keep objects made of {value_inputs[0]}"
        if function == "filter_shape" and value_inputs:
            return f"keep objects shaped like {value_inputs[0]}"
        if function == "relate" and value_inputs:
            return f"look for objects {value_inputs[0]} of the current object"
        if function == "unique":
            return "select the unique object from the current set"
        if function == "count":
            return "count how many objects remain"
        if function == "exist":
            return "check whether any object remains"
        if function == "query_color":
            return "query the color of the selected object"
        if function == "query_size":
            return "query the size of the selected object"
        if function == "query_material":
            return "query the material of the selected object"
        if function == "query_shape":
            return "query the shape of the selected object"
        if function == "same_color":
            return "find other objects with the same color"
        if function == "same_size":
            return "find other objects with the same size"
        if function == "same_material":
            return "find other objects with the same material"
        if function == "same_shape":
            return "find other objects with the same shape"
        if function.startswith("equal_") or function.startswith("less_than") or function.startswith("greater_than"):
            return f"compare values using {function.replace('_', ' ')}"

        if value_inputs:
            values = ", ".join(map(str, value_inputs))
            return f"apply {function.replace('_', ' ')} with {values}"
        return f"apply {function.replace('_', ' ')}"

    @classmethod
    def _program_to_explanation(cls, program: List[Dict[str, Any]]) -> str:
        if not program:
            return "The question is answered directly from the visual evidence."

        steps = [f"Step {idx}: {cls._format_program_step(step)}." for idx, step in enumerate(program, start=1)]
        return " ".join(steps)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image_filename"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms

            fallback_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = fallback_transform(image)

        question = item.get("question", "")
        answer = item.get("answer", "")
        explanation = self._program_to_explanation(item.get("program", []))

        return {
            "image": image,
            "prompt_text": question,
            "target_text": answer,
            "explanation": explanation,
            "image_filename": item.get("image_filename", ""),
        }

class VLMCollateFn:
    def __init__(self, tokenizer, mode: str = "stage1", num_image_tokens: int = 196):
        """
        Custom collate function handling dynamic text padding, LLaVA-style prompt 
        randomization, and strict autoregressive label masking.
        
        Args:
            tokenizer: The HuggingFace Qwen tokenizer.
            mode: "stage1" (Captioning) or "stage2" (CoT QA).
            num_image_tokens: 196 for ViT-Base (224/16 * 224/16).
        """
        self.tokenizer = tokenizer
        self.mode = mode
        self.num_image_tokens = num_image_tokens
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        self.img_placeholder_id = -200

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = torch.stack([item["image"] for item in batch])
        
        input_ids_list = []
        labels_list = []
        
        img_pad = torch.full((self.num_image_tokens,), self.img_placeholder_id, dtype=torch.long)
        newline_token = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        
        for item in batch:
            if self.mode == "stage1" or self.mode == "eval_stage1":
                base_prompt = "Describe this image in detail."
                caption = item["caption"]
                
                prompt_tokens = self.tokenizer(base_prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                target_tokens = self.tokenizer(f" {caption}{self.tokenizer.eos_token}", return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                
                if random.random() < 0.5 and self.mode != "eval_stage1":
                    context_ids = torch.cat([img_pad, newline_token, prompt_tokens])
                else:
                    context_ids = torch.cat([prompt_tokens, newline_token, img_pad])
                    
                if self.mode == "eval_stage1":
                    input_ids = context_ids
                    labels = target_tokens
                else:
                    input_ids = torch.cat([context_ids, target_tokens])
                    labels = input_ids.clone()
                    labels[:len(context_ids)] = -100
                
            elif self.mode == "stage2" or self.mode == "eval_stage2":
                question = item["prompt_text"]
                explanation = item.get("explanation", "")
                answer = item["target_text"]
                
                prompt_str = f"Question: {question}\nAnswer:"
                target_str = f" {explanation} Therefore, the answer is {answer}.{self.tokenizer.eos_token}"
                
                prompt_tokens = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                target_tokens = self.tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                
                context_ids = torch.cat([img_pad, newline_token, prompt_tokens])
                
                if self.mode == "eval_stage2":
                    input_ids = context_ids
                    labels = target_tokens # Just storing the target to compute metrics later
                else:
                    input_ids = torch.cat([context_ids, target_tokens])
                    labels = input_ids.clone()
                    labels[:len(context_ids)] = -100
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            
        input_ids_batched = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels_batched = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask_list = [torch.ones_like(ids) for ids in input_ids_list]
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        
        return {
            "images": images,
            "input_ids": input_ids_batched,
            "attention_mask": attention_mask,
            "labels": labels_batched
        }