import argparse
import os
import sys
from pathlib import Path

import modal

APP_NAME = "col775-a2-vlm-eval"
DEFAULT_VIT_CKPT = "/checkpoints/clip/checkpoint_best.pt"
DEFAULT_VLM_CHECKPOINT_DIR = "/checkpoints/vlm"
DEFAULT_DATA_ROOT = "/data/A2_dataset"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=2.2.6",
        "pillow>=12.2.0",
        "scikit-learn>=1.7.2",
        "torch==2.5.0",
        "torchvision>=0.20.0",
        "tqdm>=4.67.3",
        "huggingface_hub>=0.34.0",
        "transformers==4.50.0",
        "peft>=0.17.0",
        "accelerate>=1.10.0",
        "sacrebleu>=2.5.1",
        "wandb>=0.26.0",
    )
    .pip_install("unsloth", "bitsandbytes", "xformers", "trl")
    .add_local_dir(
        local_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        remote_path="/workspace/col775_vlm_engine",
    )
)

dataset_volume = modal.Volume.from_name("col775-a2-dataset", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("checkpoints", create_if_missing=True)


def _parse_args(cli_args: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="modal_eval_vlm.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--vlm-device", type=str, default="cuda")
    p.add_argument("--vlm-vit-ckpt", type=str, default=DEFAULT_VIT_CKPT)
    p.add_argument("--vlm-checkpoint-dir", type=str, default=DEFAULT_VLM_CHECKPOINT_DIR)
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--lora-ckpt",
        type=str,
        default="/checkpoints/vlm/vlm_stage2_lora_ep1",
        help="LoRA checkpoint dir or .safetensors file",
    )
    p.add_argument(
        "--proj-ckpt",
        type=str,
        default="/checkpoints/vlm/vlm_stage2_proj_ep1.pt",
        help="Stage-2 projector checkpoint",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base LLM id (used if Unsloth is unavailable)",
    )
    return p.parse_args(cli_args)


def _resolve_lora_paths(lora_ckpt: str) -> tuple[Path, Path]:
    lora_path = Path(lora_ckpt)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_ckpt}")

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

    return lora_dir, lora_file


def _infer_dtype(device: "torch.device") -> "torch.dtype":
    import torch

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


@app.function(
    cpu=6.0,
    image=image,
    gpu="A100-40GB",
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 6,
)
def eval_vlm_stage2_remote(args: list[str]):
    import torch
    from torch.utils.data import DataLoader, Subset

    os.chdir("/workspace/col775_vlm_engine")
    if "/workspace/col775_vlm_engine" not in sys.path:
        sys.path.insert(0, "/workspace/col775_vlm_engine")

    parsed = _parse_args(args)

    from configs.vlm_config import VLMConfig
    from models.vlm import VLMModel
    from models.vit_backbone import VisionTransformer
    from data.clevr_vlm_dataset import CLEVRQADataset, VLMCollateFn
    from engine.evaluator import evaluate_vlm_stage2

    device = torch.device(parsed.vlm_device)
    dtype = _infer_dtype(device)

    lora_dir, _ = _resolve_lora_paths(parsed.lora_ckpt)

    try:
        from unsloth import FastLanguageModel

        has_unsloth = True
    except Exception:
        FastLanguageModel = None  # type: ignore[assignment]
        has_unsloth = False

    max_seq_length = 2048
    if has_unsloth:
        llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_dir),
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
        )
        llm = llm.to(device)
        FastLanguageModel.for_inference(llm)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(parsed.base_model)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        llm = AutoModelForCausalLM.from_pretrained(
            parsed.base_model,
            torch_dtype=dtype,
        )
        llm = PeftModel.from_pretrained(llm, str(lora_dir))
        llm = llm.to(device)
        llm.eval()

    cfg = VLMConfig(data_root=parsed.data_root)
    cfg.num_workers = parsed.num_workers
    cfg.stage2_per_device_bs = parsed.batch_size

    vision_encoder = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_dim=1536
    )

    ckpt = torch.load(
        parsed.vlm_vit_ckpt,
    )
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

    model.projector.load_state_dict(torch.load(parsed.proj_ckpt, map_location=device))
    model.eval()

    val_ds = CLEVRQADataset(cfg, split="val", tokenizer=tokenizer)
    if parsed.max_samples and parsed.max_samples > 0:
        max_samples = min(parsed.max_samples, len(val_ds))
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


@app.local_entrypoint()
def main(*args: str):
    """
    Run locally to trigger Modal remote VLM stage-2 evaluation.

    Examples:
      modal run scripts/modal_eval_vlm.py -- --max-samples 128
      modal run scripts/modal_eval_vlm.py -- --lora-ckpt /checkpoints/vlm/vlm_stage2_lora_ep1
    """
    eval_vlm_stage2_remote.remote(list(args))
