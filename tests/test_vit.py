import torch
from ..models.vit_backbone import VisionTransformer

def test_vit():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize ViT with default params as per a2.pdf
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536).to(device)
    
    # Test 1: Global Crop (224x224) -> default [CLS] token output
    x_global = torch.randn(2, 3, 224, 224).to(device)
    out_global = model(x_global)
    assert out_global.shape == (2, 384), f"Test 1 Failed: Expected (2, 384), got {out_global.shape}"
    print("Test 1 Passed: Global Crop [CLS] target dimension achieved.")
    
    # Test 2: Global Crop (224x224) -> Patches for GAP
    out_patches = model(x_global, return_patches=True)
    assert out_patches.shape == (2, 196, 384), f"Test 2 Failed: Expected (2, 196, 384), got {out_patches.shape}"
    print("Test 2 Passed: Global Crop sequence token dimension achieved (GAP Ready).")
    
    # Test 3: Local Crop (96x96) -> [CLS] token, testing interpolate_pos_encoding
    x_local = torch.randn(2, 3, 96, 96).to(device)
    out_local = model(x_local)
    assert out_local.shape == (2, 384), f"Test 3 Failed: Expected (2, 384), got {out_local.shape}"
    print("Test 3 Passed: Local Crop (96x96) dynamic positional interpolation successful.")
    
    print("All architecture structural verification tests passed!")

if __name__ == "__main__":
    test_vit()
