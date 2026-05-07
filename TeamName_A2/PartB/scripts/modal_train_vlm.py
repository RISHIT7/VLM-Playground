import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import modal

APP_NAME = "col775-a2-vlm"
DEFAULT_VIT_CKPT = "/checkpoints/clip/checkpoint_best.pt"
DEFAULT_VLM_CHECKPOINT_DIR = "/checkpoints/vlm"
DEFAULT_DATA_ROOT = "/data/A2_dataset"
DEFAULT_CAPTIONS_JSON_DIR = "/data/A2_dataset/Part_Aa"


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
        "wandb>=0.26.0",
        "huggingface_hub>=0.34.0",
        "transformers>=4.55.0",
        "peft>=0.17.0",
        "accelerate>=1.10.0",
        "sacrebleu>=2.5.1",
    )
    .add_local_dir(
        local_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        remote_path="/workspace/col775_vlm_engine",
    )
)


dataset_volume = modal.Volume.from_name("col775-a2-dataset", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("checkpoints", create_if_missing=True)


def download_dataset() -> None:
    huggingface_hub = __import__("huggingface_hub")
    snapshot_download = huggingface_hub.snapshot_download

    if not os.path.exists("/data/A2_dataset/Part_A/train/clevr_train_captions.json"):
        if not glob.glob("/data/archive.tar.part*"):
            print("Downloading dataset aggr8/COL775-A2-Clevr-Extended-100k to /data...")
            snapshot_download(
                repo_id="aggr8/COL775-A2-Clevr-Extended-100k",
                repo_type="dataset",
                local_dir="/data",
            )
        else:
            print("Tarballs already present in /data, skipping download...")

        print("Extracting dataset from tarballs...")
        subprocess.run(
            "cat /data/archive.tar.part* | tar -xf - -C /data", shell=True, check=True
        )
        print("Extraction complete. Cleaning up tarballs...")
        subprocess.run(
            "rm -f /data/archive.tar.part* .gitattributes README.md",
            shell=True,
            check=True,
        )
    else:
        print("Dataset already present and extracted in /data.")

    os.makedirs("/data/A2_dataset/Part_Aa/Probe-Datasets", exist_ok=True)

    for filepath in glob.glob("/workspace/Probe-Datasets/*.json"):
        filename = os.path.basename(filepath)
        dst_probe = f"/data/A2_dataset/Part_Aa/Probe-Datasets/{filename}"
        if not os.path.exists(dst_probe):
            shutil.copy2(filepath, dst_probe)

        if "captions" in filename:
            dst_root = f"/data/A2_dataset/Part_Aa/{filename}"
            if not os.path.exists(dst_root):
                shutil.copy2(filepath, dst_root)


def _set_wandb_mode(wandb_offline: bool = False) -> None:
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"


def _common_parser(stage_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"modal_train_vlm.py ({stage_name})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    p.add_argument("--vlm-device", type=str, default="cuda", help="cuda | cpu | mps")
    p.add_argument("--vlm-vit-ckpt", type=str, default=DEFAULT_VIT_CKPT)
    p.add_argument("--vlm-checkpoint-dir", type=str, default=DEFAULT_VLM_CHECKPOINT_DIR)
    p.add_argument("--vlm-llm-id", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument(
        "--vlm-batch-size",
        type=int,
        default=None,
        help="Per-device batch size override.",
    )
    p.add_argument("--vlm-epochs", type=int, default=None)
    p.add_argument("--vlm-lr", type=float, default=None)
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--captions-json-dir", type=str, default=DEFAULT_CAPTIONS_JSON_DIR)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-offline", action="store_true")

    return p


def _parse_stage1_args(cli_args: list[str]) -> argparse.Namespace:
    return _common_parser("stage1").parse_args(cli_args)


def _parse_stage2_args(cli_args: list[str]) -> argparse.Namespace:
    p = _common_parser("stage2")
    p.add_argument(
        "--stage1-ckpt",
        type=str,
        default=None,
        help="Path to a Stage-1 projector checkpoint. If omitted, latest in --vlm-checkpoint-dir is used.",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args(cli_args)


def _maybe_scale_stage1_lr(cfg) -> None:
    base_eff_bs = cfg.stage1_target_bs
    new_eff_bs = cfg.stage1_per_device_bs * cfg.num_gpus * cfg.stage1_grad_accum
    if new_eff_bs != base_eff_bs:
        old_lr = cfg.stage1_lr
        cfg.stage1_lr = old_lr * (new_eff_bs / base_eff_bs)
        print(
            f"[Stage 1] Effective batch size changed. Scaling LR: {old_lr:.2e} -> {cfg.stage1_lr:.2e} "
            f"(Eff BS: {new_eff_bs})"
        )


def _maybe_scale_stage2_lr(cfg) -> None:
    base_eff_bs = cfg.stage2_target_bs
    new_eff_bs = cfg.stage2_per_device_bs * cfg.num_gpus * cfg.stage2_grad_accum
    if new_eff_bs != base_eff_bs:
        old_lr = cfg.stage2_lr
        cfg.stage2_lr = old_lr * (new_eff_bs / base_eff_bs)
        print(
            f"[Stage 2] Effective batch size changed. Scaling LR: {old_lr:.2e} -> {cfg.stage2_lr:.2e} "
            f"(Eff BS: {new_eff_bs})"
        )


def _resolve_latest_stage1_ckpt(vlm_checkpoint_dir: str) -> str:
    pattern = os.path.join(vlm_checkpoint_dir, "vlm_stage1_proj_ep*.pt")
    candidates = glob.glob(pattern)

    if not candidates:
        fallback = os.path.join(vlm_checkpoint_dir, "vlm_stage1_proj_ep1.pt")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(
            "No stage-1 projector checkpoint found. "
            f"Expected something like: {pattern}"
        )

    def epoch_id(path: str) -> int:
        m = re.search(r"ep(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    candidates.sort(key=epoch_id)
    return candidates[-1]


@app.function(
    cpu=8.0,
    image=image,
    gpu="L40S",
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("col775")],
    timeout=60 * 60 * 24,
)
def train_vlm_stage1_remote(args: list[str]):
    """
    Modal entrypoint for VLM Stage 1 (projector-only alignment with frozen CLIP vision encoder + frozen LLM).
    """
    download_dataset()

    os.chdir("/workspace/col775_vlm_engine")
    sys.path.insert(0, "/workspace/col775_vlm_engine")

    parsed = _parse_stage1_args(args)
    _set_wandb_mode(parsed.wandb_offline)

    from configs.vlm_config import VLMConfig
    from engine.trainer_vlm import train_vlm_stage1_launcher

    cfg = VLMConfig(
        num_gpus=parsed.num_gpus,
        device=parsed.vlm_device,
        llm_model_id=parsed.vlm_llm_id,
    )

    cfg.data_root = parsed.data_root
    setattr(cfg, "captions_json", parsed.captions_json_dir)
    cfg.checkpoint_dir = parsed.vlm_checkpoint_dir

    if parsed.vlm_batch_size is not None:
        cfg.stage1_per_device_bs = parsed.vlm_batch_size
    if parsed.vlm_epochs is not None:
        cfg.stage1_epochs = parsed.vlm_epochs
    if parsed.vlm_lr is not None:
        cfg.stage1_lr = parsed.vlm_lr
    else:
        _maybe_scale_stage1_lr(cfg)
    if parsed.num_workers is not None:
        cfg.num_workers = parsed.num_workers
    if parsed.wandb_project is not None:
        cfg.wandb_project = parsed.wandb_project

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"Running VLM stage-1 with args: {args}")
    print(f"Using fixed CLIP checkpoint: {parsed.vlm_vit_ckpt}")
    print(f"Saving VLM checkpoints to: {cfg.checkpoint_dir}")

    train_vlm_stage1_launcher(cfg, parsed.vlm_vit_ckpt)

    dataset_volume.commit()
    checkpoint_volume.commit()


@app.function(
    cpu=8.0,
    image=image,
    gpu="L40S",
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("col775")],
    timeout=60 * 60 * 24,
)
def train_vlm_stage2_remote(args: list[str]):
    """
    Modal entrypoint for VLM Stage 2 (LoRA finetuning of LLM + projector, frozen CLIP vision encoder).
    """
    download_dataset()

    os.chdir("/workspace/col775_vlm_engine")
    sys.path.insert(0, "/workspace/col775_vlm_engine")

    parsed = _parse_stage2_args(args)
    _set_wandb_mode(parsed.wandb_offline)

    from configs.vlm_config import VLMConfig
    from engine.trainer_vlm import train_vlm_stage2_launcher

    cfg = VLMConfig(
        num_gpus=parsed.num_gpus,
        device=parsed.vlm_device,
        llm_model_id=parsed.vlm_llm_id,
    )

    cfg.data_root = parsed.data_root
    setattr(cfg, "captions_json", parsed.captions_json_dir)
    cfg.checkpoint_dir = parsed.vlm_checkpoint_dir

    if parsed.vlm_batch_size is not None:
        cfg.stage2_per_device_bs = parsed.vlm_batch_size
    if parsed.vlm_epochs is not None:
        cfg.stage2_epochs = parsed.vlm_epochs
    if parsed.vlm_lr is not None:
        cfg.stage2_lr = parsed.vlm_lr
    else:
        _maybe_scale_stage2_lr(cfg)
    if parsed.num_workers is not None:
        cfg.num_workers = parsed.num_workers
    if parsed.wandb_project is not None:
        cfg.wandb_project = parsed.wandb_project

    cfg.lora_rank = parsed.lora_rank
    cfg.lora_alpha = parsed.lora_alpha

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if parsed.stage1_ckpt is not None:
        stage1_ckpt_path = parsed.stage1_ckpt
        if not os.path.exists(stage1_ckpt_path):
            raise FileNotFoundError(
                f"Provided --stage1-ckpt not found: {stage1_ckpt_path}"
            )
    else:
        stage1_ckpt_path = _resolve_latest_stage1_ckpt(cfg.checkpoint_dir)

    print(f"Running VLM stage-2 with args: {args}")
    print(f"Using fixed CLIP checkpoint: {parsed.vlm_vit_ckpt}")
    print(f"Using Stage-1 projector checkpoint: {stage1_ckpt_path}")
    print(f"Saving VLM checkpoints to: {cfg.checkpoint_dir}")

    train_vlm_stage2_launcher(cfg, parsed.vlm_vit_ckpt, stage1_ckpt_path)

    dataset_volume.commit()
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(*args: str):
    """
    Run locally to trigger Modal remote VLM training.

    Examples:
      modal run scripts/modal_train_vlm.py -- --stage stage1 --vlm-epochs 2
      modal run scripts/modal_train_vlm.py -- --stage stage2 --vlm-epochs 1 --stage1-ckpt /checkpoints/vlm/vlm_stage1_proj_ep2.pt
      modal run scripts/modal_train_vlm.py -- stage2 --vlm-epochs 1
    """
    stage = "stage1"
    forwarded_args = list(args)

    # Allow either:
    #   -- stage2 --vlm-epochs 1
    #   -- --stage stage2 --vlm-epochs 1
    if forwarded_args:
        first = forwarded_args[0].strip().lower()
        if first in {"stage1", "stage2"}:
            stage = first
            forwarded_args = forwarded_args[1:]
        elif len(forwarded_args) >= 2 and forwarded_args[0] == "--stage":
            stage = forwarded_args[1].strip().lower()
            forwarded_args = forwarded_args[2:]

    stage = stage.lower().strip()

    if stage == "stage1":
        train_vlm_stage1_remote.remote(forwarded_args)
    elif stage == "stage2":
        train_vlm_stage2_remote.remote(forwarded_args)
    else:
        raise ValueError("stage must be one of: stage1 | stage2")
