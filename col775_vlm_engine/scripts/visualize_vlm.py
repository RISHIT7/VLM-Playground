import os
import sys
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import textwrap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from models.vlm import VLMModel
from models.vit_backbone import VisionTransformer
from data.clevr_vlm_dataset import CLEVRCaptionDataset, CLEVRQADataset, VLMCollateFn
from configs.vlm_config import VLMConfig


@torch.no_grad()
def visualize_vlm(cfg, stage: int, ckpt_path: str, vit_checkpoint_path: str, num_samples: int = 5, output_file: str = "vlm_visualization.png"):
    device = torch.device(cfg.device if hasattr(cfg, 'device') else "cuda")
    print(f"Loading VLM Stage {stage} for visualization on {device}...")
    
    if "FP8" in cfg.llm_model_id.upper() or "INT8" in cfg.llm_model_id.upper():
        dtype = "auto"
    elif device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
        
    max_seq_length = 2048
    
    if stage == 1:
        llm, tokenizer = FastLanguageModel.from_pretrained(model_name=cfg.llm_model_id, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=False)
        if dtype != "auto": llm = llm.to(device)
        llm.requires_grad_(False)
        
        vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
        if vit_checkpoint_path and os.path.exists(vit_checkpoint_path):
            ckpt = torch.load(vit_checkpoint_path, map_location="cpu")
            state_dict = {k.replace("vit_backbone.", ""): v for k, v in ckpt.get('model_state', ckpt).items() if k.startswith("vit_backbone.")}
            vision_encoder.load_state_dict(state_dict, strict=False)
        vision_encoder = vision_encoder.to(device)
        vision_encoder.requires_grad_(False)
        
        model = VLMModel(vision_encoder, llm, img_placeholder_id=-200, expansion_factor=cfg.expansion_factor).to(device)
        
        proj_ckpt = torch.load(ckpt_path, map_location=device)
        if 'projector_state_dict' in proj_ckpt:
            model.projector.load_state_dict(proj_ckpt['projector_state_dict'])
        else:
            model.projector.load_state_dict(proj_ckpt)
            
        val_ds = CLEVRCaptionDataset(cfg, split="val", tokenizer=tokenizer)
        collate_fn = VLMCollateFn(tokenizer, mode="eval_stage1")
        
    elif stage == 2:
        llm, tokenizer = FastLanguageModel.from_pretrained(model_name=ckpt_path, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=False)
        if dtype != "auto": llm = llm.to(device)
        FastLanguageModel.for_inference(llm) 
        
        vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
        if vit_checkpoint_path and os.path.exists(vit_checkpoint_path):
            ckpt = torch.load(vit_checkpoint_path, map_location="cpu")
            state_dict = {k.replace("vit_backbone.", ""): v for k, v in ckpt.get('model_state', ckpt).items() if k.startswith("vit_backbone.")}
            vision_encoder.load_state_dict(state_dict, strict=False)
        vision_encoder = vision_encoder.to(device)
        vision_encoder.requires_grad_(False)
        
        model = VLMModel(vision_encoder, llm, img_placeholder_id=(tokenizer.unk_token_id or 0), expansion_factor=cfg.expansion_factor).to(device)
        
        proj_path = ckpt_path.replace("_lora_", "_proj_") + ".pt"
        if os.path.exists(proj_path):
            model.projector.load_state_dict(torch.load(proj_path, map_location=device))
            
        val_ds = CLEVRQADataset(cfg, split="val", tokenizer=tokenizer)
        collate_fn = VLMCollateFn(tokenizer, mode="eval_stage2")

    model.eval()
    
    # Pick random samples
    indices = random.sample(range(len(val_ds)), min(num_samples, len(val_ds)))
    samples = [val_ds[i] for i in indices]
    
    # Process batch
    batch = collate_fn(samples)
    images = batch["images"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    print("Generating predictions...")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=autocast_device):
        outputs = model.generate(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128 if stage == 2 else 64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        
    input_len = input_ids.shape[1]
    generated_ids = outputs[:, input_len:]
    pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Plotting
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i, (sample, pred) in enumerate(zip(samples, pred_texts)):
        ax = axes[i]
        
        # Display image
        image_path = os.path.join(val_ds.image_dir, sample["image_filename"])
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Format text
        if stage == 1:
            gt_text = sample.get("caption", sample.get("target_text", ""))
            text = f"GROUND TRUTH:\n{textwrap.fill(gt_text, width=60)}\n\nPREDICTION:\n{textwrap.fill(pred.strip(), width=60)}"
        else:
            question = sample["prompt_text"]
            gt_ans = sample["target_text"]
            pred_text = pred.strip()
            
            # Simple exact match check
            is_correct = False
            try:
                pred_ans = pred_text.split("therefore, the answer is ")[1].replace(".", "").strip().lower()
                gt_ans_clean = str(gt_ans).strip().lower()
                is_correct = (pred_ans == gt_ans_clean)
            except:
                is_correct = (pred_text.lower() == str(gt_ans).lower())
                
            color = "green" if is_correct else "red"
            text = f"Q: {question}\n\nGROUND TRUTH: {gt_ans}\n\nPREDICTION: {textwrap.fill(pred_text, width=60)}"
            
            ax.set_title("CORRECT" if is_correct else "INCORRECT", color=color, fontweight='bold')
            
        ax.text(1.05, 0.5, text, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vit-ckpt", type=str, default="../checkpoints/dino/checkpoint_best.pt")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="vlm_visualization.png")
    parser.add_argument("--data-root", type=str, default="/scratch/work/rishit/COL775/assignment-2-data")
    args = parser.parse_args()
    
    cfg = VLMConfig(data_root=args.data_root)
    visualize_vlm(cfg, args.stage, args.ckpt, args.vit_ckpt, args.num_samples, args.output)
