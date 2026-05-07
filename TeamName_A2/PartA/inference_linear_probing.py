import os
import json
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Import local models
from models.vit_backbone import VisionTransformer
from models.linear_probe import LinearProbe

# Provide the CLEVR colors in the order they were used during training
# IMPORTANT: Adjust this order if your training multi_hot labels used a different index mapping!
CLEVR_COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

def main():
    parser = argparse.ArgumentParser(description="Part A: Linear Probing Inference")
    parser.add_argument("--model_type", type=str, required=True, choices=["clip", "dino_student", "dino_teacher"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--pooling_type", type=str, required=True, choices=["cls", "gap"])
    parser.add_argument("--probe_task", type=str, required=True, choices=["count", "color"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the backbone
    # (Using the default parameters as specified in Assignment requirements)
    backbone = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    
    # Load backbone weights
    checkpoint_name = {
        "clip": "clip.pth",
        "dino_student": "dino_student.pth",
        "dino_teacher": "dino_teacher.pth"
    }[args.model_type]
    
    backbone_path = os.path.join(args.model_dir, checkpoint_name)
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_path}")
        
    backbone_state = torch.load(backbone_path, map_location="cpu")
    # Handle possible nested state dicts (e.g. if saved under 'model' or 'state_dict' key)
    if 'state_dict' in backbone_state:
        backbone_state = backbone_state['state_dict']
    elif 'model' in backbone_state:
        backbone_state = backbone_state['model']
        
    # Strictly load backbone (ignoring any projection heads in the same dict)
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(device)
    backbone.eval()
    
    # 2. Instantiate and load the linear probe
    num_classes = 11 if args.probe_task == "count" else 8 # Clevr max count is 10, colors is 8
    multi_label = (args.probe_task == "color")
    
    probe = LinearProbe(in_dim=384, num_classes=num_classes, multi_label=multi_label)
    probe_name = f"linear_probe_{args.model_type}_{args.probe_task}_{args.pooling_type}.pth"
    probe_path = os.path.join(args.model_dir, probe_name)
    
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Linear Probe checkpoint not found: {probe_path}")
        
    probe_state = torch.load(probe_path, map_location="cpu")
    if 'state_dict' in probe_state:
        probe_state = probe_state['state_dict']
    elif 'model' in probe_state:
        probe_state = probe_state['model']
        
    probe.load_state_dict(probe_state)
    probe.to(device)
    probe.eval()
    
    # 3. Load Data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
        
    examples = data.get("examples", data)
    transform = get_inference_transform()
    
    results = {}
    
    with torch.no_grad():
        for example in examples:
            img_filename = example.get("image_filename")
            img_path = example.get("image_path")
            
            if not img_path or not os.path.exists(img_path):
                # Fallback if image_path is somehow broken, though instructions say it will be absolute
                continue
                
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Forward pass
            return_patches = (args.pooling_type == "gap")
            features = backbone(img_tensor, return_patches=return_patches)
            
            if return_patches:
                # Global Average Pooling over patches
                features = features.mean(dim=1)
                
            logits = probe(features).squeeze(0) # (num_classes,)
            
            if args.probe_task == "count":
                pred_count = logits.argmax().item()
                results[img_filename] = pred_count
                
            elif args.probe_task == "color":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
                pred_colors = [CLEVR_COLORS[idx] for idx in preds if idx < len(CLEVR_COLORS)]
                results[img_filename] = pred_colors

    # 4. Save Output
    # Ensure output directory exists if provided as path
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
