import sys
import os
import torch

# Add the root directory to sys.path so we can import configs and data
sys.path.append("/Users/rishit/Documents/Programming/Courses/COL775/Assignment-2_COL775")

from col775_vlm_engine.data.text_tokenizer import CLEVRTokenizer
from col775_vlm_engine.data.clevr_dataset import CLEVRDataset, CLEVRCollateFn
from col775_vlm_engine.configs.data_config import get_config, EnvConfig

def test_suite_1():
    print("--- Test Suite 1: Tokenizer EOS Contract ---")
    annotations_file = "dummy_annotations.json"
    import json
    with open(annotations_file, "w") as f:
        json.dump([
            {"caption": "A short caption."},
            {"caption": "A much longer caption for testing. It has many more words just in case."}
        ], f)
    
    tokenizer = CLEVRTokenizer(max_seq_len=77)
    tokenizer.build_vocab(annotations_file)
    
    short_tokens, short_mask = tokenizer.encode("A short caption.")
    long_tokens, long_mask = tokenizer.encode("A much longer caption for testing.")
    
    eos_id = tokenizer.word2idx["[EOS]"]
    print(f"Max ID in vocab: {max(tokenizer.word2idx.values())}, EOS ID: {eos_id}")
    
    try:
        assert max(tokenizer.word2idx.values()) == eos_id, "EOS is not the maximum integer ID"
        assert tokenizer.word2idx["[PAD]"] == 0, "PAD is not 0"
        
        short_argmax = short_tokens.argmax(dim=-1).item()
        long_argmax = long_tokens.argmax(dim=-1).item()
        
        print(f"Short argmax index: {short_argmax}, Token there: {short_tokens[short_argmax]}")
        print(f"Long argmax index: {long_argmax}, Token there: {long_tokens[long_argmax]}")
        
        if short_tokens[short_argmax] != eos_id:
            print("[FAIL] Test Suite 1")
        else:
            print("[PASS] Test Suite 1")
    except Exception as e:
        print(f"[FAIL] Test Suite 1: {e}")

def test_suite_2():
    print("\n--- Test Suite 2: CLIP Batch Collision Protocol ---")
    batch = [
        {"image": torch.zeros(3, 224, 224), "tokens": torch.zeros(77, dtype=torch.long), "padding_mask": torch.ones(77, dtype=torch.bool), "raw_caption": "Caption 1"},
        {"image": torch.zeros(3, 224, 224), "tokens": torch.zeros(77, dtype=torch.long), "padding_mask": torch.ones(77, dtype=torch.bool), "raw_caption": "Caption 2"}
    ]
    collate = CLEVRCollateFn(mode="clip")
    try:
        out = collate(batch)
        raw_captions = out["raw_captions"]
        if isinstance(raw_captions, list) and all(isinstance(c, str) for c in raw_captions) and len(raw_captions) == 2:
            print("[PASS] Test Suite 2")
        else:
            print("[FAIL] Test Suite 2: raw_captions is not a list[str] of length B")
    except Exception as e:
         print(f"[FAIL] Test Suite 2: Exception {e}")

def test_suite_3():
    print("\n--- Test Suite 3: DINO 5D Tensor Stacking Contract ---")
    batch = [
        {"global_crops": [torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)],
         "local_crops": [torch.zeros(3, 96, 96) for _ in range(8)]},
        {"global_crops": [torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)],
         "local_crops": [torch.zeros(3, 96, 96) for _ in range(8)]}
    ]
    collate = CLEVRCollateFn(mode="dino")
    try:
        out = collate(batch)
        global_crops = out["global_crops"]
        local_crops = out["local_crops"]
        
        if isinstance(global_crops, torch.Tensor) and global_crops.shape == (2, 2, 3, 224, 224):
            if isinstance(local_crops, torch.Tensor) and local_crops.shape == (2, 8, 3, 96, 96):
                print("[PASS] Test Suite 3")
            else:
                 print(f"[FAIL] Test Suite 3: local_crops shape mismatch or not Tensor: {local_crops.shape if isinstance(local_crops, torch.Tensor) else type(local_crops)}")
        else:
            print(f"[FAIL] Test Suite 3: global_crops shape mismatch or not Tensor: {global_crops.shape if isinstance(global_crops, torch.Tensor) else type(global_crops)}")
            
    except Exception as e:
         print(f"[FAIL] Test Suite 3: Exception {e}")

def test_suite_4():
    print("\n--- Test Suite 4: Environment Configurations (Local vs. HPC) ---")
    local_cfg = get_config("local")
    hpc_cfg = get_config("hpc")
    try:
        passed = True
        if not (local_cfg.num_workers == 0 and local_cfg.pin_memory == False):
            print("[FAIL] Test Suite 4: local env conditions not met")
            passed = False
        
        if not (hpc_cfg.num_workers >= 8 and hpc_cfg.pin_memory == True):
            print("[FAIL] Test Suite 4: hpc env conditions not met")
            passed = False
            
        if not hasattr(hpc_cfg, "drop_last") or not hpc_cfg.drop_last:
            print("[FAIL] Test Suite 4: hpc missing drop_last=True")
            passed = False
            
        if passed:
            print("[PASS] Test Suite 4")
    except Exception as e:
        print(f"[FAIL] Test Suite 4: Exception {e}")

if __name__ == "__main__":
    test_suite_1()
    test_suite_2()
    test_suite_3()
    test_suite_4()
