import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

# Ensure repo root on path (workspace root)
REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "col775_vlm_engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

FastLanguageModel: Any
try:
    from unsloth import FastLanguageModel as _FastLanguageModel

    FastLanguageModel = _FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    FastLanguageModel = None
    _HAS_UNSLOTH = False

from configs.vlm_config import VLMConfig
from models.vlm import VLMModel
from models.vit_backbone import VisionTransformer
from data.clevr_vlm_dataset import CLEVRQADataset, VLMCollateFn
from engine.evaluator import evaluate_vlm_stage2


def _infer_dtype(device: torch.device) -> torch.dtype | str:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM Stage-2 evaluation")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "Dataset"),
        help="Path to dataset root (contains Part_Aa)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base LLM id (used if Unsloth is unavailable)",
    )
    parser.add_argument(
        "--lora-ckpt",
        type=str,
        default=str(
            REPO_ROOT
            / "checkpoints"
            / "vlm"
            / "vlm_stage2_lora_ep1"
            / "adapter_model.safetensors"
        ),
        help="Path to Stage-2 LoRA checkpoint (folder or .safetensors file)",
    )
    parser.add_argument(
        "--proj-ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "vlm" / "vlm_stage2_proj_ep1.pt"),
        help="Path to Stage-2 projector checkpoint",
    )
    parser.add_argument(
        "--vit-ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "clip" / "checkpoint_best.pt"),
        help="Path to CLIP ViT checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, evaluate only this many samples (smoke test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _infer_dtype(device)

    lora_path = Path(args.lora_ckpt)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_ckpt}")

    if lora_path.is_dir():
        safetensors = sorted(lora_path.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError(
                f"No .safetensors file found in LoRA folder: {lora_path}"
            )
        lora_file = safetensors[0]
        lora_dir = lora_path
    else:
        lora_file = lora_path
        lora_dir = lora_path.parent
    if not os.path.exists(args.proj_ckpt):
        raise FileNotFoundError(f"Projector checkpoint not found: {args.proj_ckpt}")
    if not os.path.exists(args.vit_ckpt):
        raise FileNotFoundError(f"ViT checkpoint not found: {args.vit_ckpt}")

    cfg = VLMConfig(data_root=args.data_root)
    cfg.num_workers = args.num_workers
    cfg.stage2_per_device_bs = args.batch_size

    max_seq_length = 2048
    if _HAS_UNSLOTH:
        llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_dir),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
        )
        if dtype != "auto":
            llm = llm.to(device)
        FastLanguageModel.for_inference(llm)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        llm = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype if isinstance(dtype, torch.dtype) else None,
        )
        llm = PeftModel.from_pretrained(llm, str(lora_dir))
        llm = llm.to(device)
        llm.eval()

    vision_encoder = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536
    )

    ckpt = torch.load(args.vit_ckpt, map_location="cpu")
    full_state = ckpt.get("model_state", ckpt)
    state_dict = {
        k.replace("vit_backbone.", ""): v
        for k, v in full_state.items()
        if k.startswith("vit_backbone.")
    }
    vision_encoder.load_state_dict(state_dict, strict=False)
    vision_encoder = vision_encoder.to(device)
    vision_encoder.requires_grad_(False)

    model = VLMModel(
        vision_encoder=vision_encoder,
        llm=llm,  # type: ignore[arg-type]
        img_placeholder_id=-200,
        expansion_factor=cfg.expansion_factor,
    ).to(device)

    model.projector.load_state_dict(torch.load(args.proj_ckpt, map_location=device))
    model.eval()

    val_ds = CLEVRQADataset(cfg, split="val", tokenizer=tokenizer)
    if args.max_samples and args.max_samples > 0:
        max_samples = min(args.max_samples, len(val_ds))
        val_ds = Subset(val_ds, list(range(max_samples)))

    collate_fn = VLMCollateFn(tokenizer, mode="eval_stage2")
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.stage2_per_device_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    metrics = evaluate_vlm_stage2(model, val_dl, device, tokenizer)
    print(metrics)


if __name__ == "__main__":
    main()
