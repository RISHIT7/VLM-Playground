import sys
import os
import json
import tempfile
import shutil
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Ensure we can import from parent (data-pipeline) directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.data_config import get_config
from data.text_tokenizer import CLEVRTokenizer
from data.transforms import CLIPTransforms, DINOMultiCropTransforms, LinearProbeTransforms
from data.clevr_dataset import CLEVRDataset, CLEVRCollateFn

def create_dummy_data(num_samples=4):
    """Create temporary dummy JSON and images for testing."""
    temp_dir = tempfile.mkdtemp()
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create dummy images (at least num_samples)
    for i in range(num_samples):
        img = Image.new('RGB', (224, 224), color=(i*50, 100, 200))
        img.save(os.path.join(images_dir, f"dummy_{i}.png"))
    
    # Create dummy JSON with num_samples entries
    dummy_data = []
    for i in range(num_samples):
        dummy_data.append({
            "image": f"dummy_{i}.png",
            "caption": f"An image with {i+1} objects: 1 small red metal cube, {i} large blue rubber spheres",
            "count": i+1,
            "color_set": [1,0,0,1,0,0]  # example multi-hot of length 6
        })
    
    json_path = os.path.join(temp_dir, "train.json")
    with open(json_path, "w") as f:
        json.dump(dummy_data, f)
    
    return temp_dir

def test_clip():
    print("=" * 40)
    print("Testing CLIP mode with dummy data...")
    dummy_dir = create_dummy_data(num_samples=4)
    config = get_config("local")
    config.base_dir_part_a = dummy_dir
    config.base_dir_part_aa = dummy_dir
    config.batch_size = 2  # ensure batch size <= num_samples

    tokenizer = CLEVRTokenizer(max_seq_len=77)
    tokenizer.build_vocab(os.path.join(dummy_dir, "train.json"))
    dataset = CLEVRDataset(config, mode="clip", split="train",
                           transform=CLIPTransforms(), tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=config.batch_size, 
                        collate_fn=CLEVRCollateFn("clip"), drop_last=False)  # drop_last=False to avoid empty batch
    batch = next(iter(loader))
    print("CLIP batch shapes:")
    print(" images:", batch["images"].shape)          # (B,3,224,224)
    print(" tokens:", batch["tokens"].shape)          # (B,77)
    print(" padding_mask:", batch["padding_mask"].shape)
    print(" raw_captions length:", len(batch["raw_captions"]))
    print("✓ CLIP test passed")
    shutil.rmtree(dummy_dir)

def test_dino():
    print("=" * 40)
    print("Testing DINO mode with dummy data...")
    dummy_dir = create_dummy_data(num_samples=4)
    config = get_config("local")
    config.base_dir_part_a = dummy_dir
    config.batch_size = 2

    dataset = CLEVRDataset(config, mode="dino", split="train",
                           transform=DINOMultiCropTransforms(local_crops_number=8), tokenizer=None)
    loader = DataLoader(dataset, batch_size=config.batch_size, 
                        collate_fn=CLEVRCollateFn("dino"), drop_last=False)
    batch = next(iter(loader))
    print("DINO batch shapes:")
    print(" global_crops:", batch["global_crops"].shape)   # (B,2,3,224,224)
    print(" local_crops:", batch["local_crops"].shape)     # (B,8,3,96,96)
    print("✓ DINO test passed")
    shutil.rmtree(dummy_dir)

def test_linear_probe():
    print("=" * 40)
    print("Testing LinearProbe mode with dummy data...")
    dummy_dir = create_dummy_data(num_samples=4)
    config = get_config("local")
    config.base_dir_part_aa = dummy_dir
    config.batch_size = 2

    dataset = CLEVRDataset(config, mode="linear_probe", split="train",
                           transform=LinearProbeTransforms(), tokenizer=None)
    loader = DataLoader(dataset, batch_size=config.batch_size, 
                        collate_fn=CLEVRCollateFn("linear_probe"), drop_last=False)
    batch = next(iter(loader))
    print("Linear probe batch shapes:")
    print(" images:", batch["images"].shape)               # (B,3,224,224)
    print(" count_labels:", batch["count_labels"].shape)   # (B,)
    print(" color_labels:", batch["color_labels"].shape)   # (B, num_colors)
    print("✓ LinearProbe test passed")
    shutil.rmtree(dummy_dir)

if __name__ == "__main__":
    test_clip()
    test_dino()
    test_linear_probe()
    print("\n" + "=" * 40)
    print("All tests passed! Your data pipeline is ready.")