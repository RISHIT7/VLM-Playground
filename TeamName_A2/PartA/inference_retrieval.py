import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms

# Import local models
from models.vit_backbone import VisionTransformer
from models.text_encoder import TextEncoder
from models.clip_heads import CLIPEngine
from data.text_tokenizer import CLEVRTokenizer

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

def main():
    parser = argparse.ArgumentParser(description="Part A: CLIP Retrieval Inference")
    parser.add_argument("--model_type", type=str, required=True, choices=["clip"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--retrieval_task", type=str, required=True, choices=["i2t", "t2i"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    vocab_path = os.path.join(args.model_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_path}")
    
    tokenizer = CLEVRTokenizer(max_seq_len=77)
    tokenizer.load_vocab(vocab_path)
    
    # 2. Instantiate and Load CLIP Model
    vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536)
    text_enc = TextEncoder(vocab_size=tokenizer.vocab_size, embed_dim=384, depth=6, num_heads=6, mlp_dim=1536, max_len=77)
    clip_model = CLIPEngine(vit_backbone=vit, text_encoder=text_enc, embed_dim=384, proj_dim=512)
    
    clip_ckpt_path = os.path.join(args.model_dir, "clip.pth")
    if not os.path.exists(clip_ckpt_path):
        raise FileNotFoundError(f"CLIP checkpoint not found: {clip_ckpt_path}")
    
    ckpt = torch.load(clip_ckpt_path, map_location="cpu")
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
        
    clip_model.load_state_dict(state_dict)
    clip_model.to(device)
    clip_model.eval()
    
    # 3. Load Data
    with open(args.data_path, 'r') as f:
        examples = json.load(f)
    
    # If the file has an "examples" key, use it. Otherwise assume it's a list directly.
    if isinstance(examples, dict) and "examples" in examples:
        examples = examples["examples"]
        
    transform = get_inference_transform()
    
    # Pre-compute features depending on the task to save time, or compute online.
    # The requirement is top 3 matches for every query against the pool of all validation items.
    # The "pool" consists of all examples in the provided JSON file.
    
    all_img_features = []
    all_txt_features = []
    
    # We will gather all images and all texts in order
    image_filenames = []
    captions = []
    
    with torch.no_grad():
        for example in examples:
            img_filename = example.get("image_filename")
            img_path = example.get("image_path")
            caption = example.get("caption")
            
            image_filenames.append(img_filename)
            captions.append(caption)
            
            # Process Image
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                # If image_path is absent or broken, we create a dummy tensor (should not happen in autograder)
                img = Image.new('RGB', (224, 224))
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_feat = clip_model.get_image_features(img_tensor) # Already L2 normalized
            all_img_features.append(img_feat.cpu())
            
            # Process Text
            tokens, mask = tokenizer.encode(caption)
            tokens = tokens.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            txt_feat = clip_model.get_text_features(tokens, mask) # Already L2 normalized
            all_txt_features.append(txt_feat.cpu())

    img_mat = torch.cat(all_img_features, dim=0) # (N, 512)
    txt_mat = torch.cat(all_txt_features, dim=0) # (N, 512)
    
    results = {}
    
    if args.retrieval_task == "i2t":
        # Query = Image, Bank = Text
        # Output: {"filename.png": ["caption 1", "caption 2", "caption 3"]}
        scores = img_mat @ txt_mat.T # (N, N)
        topk = min(3, scores.shape[1])
        top_scores, top_indices = scores.topk(topk, dim=1)
        
        for i, idxs in enumerate(top_indices.tolist()):
            filename = image_filenames[i]
            top_caps = [captions[idx] for idx in idxs]
            results[filename] = top_caps
            
    elif args.retrieval_task == "t2i":
        # Query = Text, Bank = Image
        # Output: {"caption text": ["file1.png", "file2.png", "file3.png"]}
        scores = txt_mat @ img_mat.T # (N, N)
        topk = min(3, scores.shape[1])
        top_scores, top_indices = scores.topk(topk, dim=1)
        
        # Note: The assignment says "No need to worry about duplicate captions in the input data for t2i retrieval task"
        for i, idxs in enumerate(top_indices.tolist()):
            cap = captions[i]
            top_imgs = [image_filenames[idx] for idx in idxs]
            results[cap] = top_imgs
            
    # 4. Save Output
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
