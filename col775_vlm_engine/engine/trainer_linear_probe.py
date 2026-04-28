import logging
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.data_config import get_config
from configs.linear_probe_config import LinearProbeConfig
from data.clevr_dataset import CLEVRCollateFn, CLEVRDataset
from data.transforms import LinearProbeTransforms
from engine.evaluator import evaluate_linear_probe, extract_features
from models.linear_probe import LinearProbe
from models.vit_backbone import VisionTransformer
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


# helpers 

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_tag(cfg: LinearProbeConfig) -> str:
    """Unique tag for this (backbone, representation, task) run."""
    return f"{cfg.backbone}_{cfg.representation}_{cfg.task}"


# backbone loading 

def _build_backbone(cfg: LinearProbeConfig, device: torch.device) -> VisionTransformer:
    """
    Instantiate a ViT and load pretrained weights from the
    corresponding CLIP or DINO checkpoint.  All parameters are frozen.
    """

    vit = VisionTransformer(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_num_heads,
        mlp_dim=cfg.vit_mlp_dim,
    ).to(device)

    ckpt_path: Optional[str] = None
    if cfg.backbone == "clip":
        ckpt_path = cfg.clip_checkpoint
    elif cfg.backbone in ("dino", "dino_teacher"):
        ckpt_path = cfg.dino_checkpoint
    else:
        raise ValueError(f"Unknown backbone '{cfg.backbone}', expected 'clip', 'dino', or 'dino_teacher'.")

    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {ckpt_path}. "
            f"Please train the {cfg.backbone.upper()} model first."
        )

    logger.info(f"Loading {cfg.backbone.upper()} backbone weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if cfg.backbone == "clip":
        # CLIPEngine checkpoint stores full model state under "model_state"
        clip_state = ckpt["model_state"]
        # VIT weights live under "vit_backbone.*"
        vit_state = {
            k.replace("vit_backbone.", ""): v
            for k, v in clip_state.items()
            if k.startswith("vit_backbone.")
        }
        vit.load_state_dict(vit_state, strict=True)
        logger.info("  ✓ Loaded ViT backbone from CLIP checkpoint.")

    elif cfg.backbone == "dino":
        student_state = ckpt["student_network_state"]
        vit_state = {
            k.replace("0.", "", 1): v
            for k, v in student_state.items()
            if k.startswith("0.")
        }
        vit.load_state_dict(vit_state, strict=True)
        logger.info("  ✓ Loaded ViT backbone from DINO student checkpoint.")

    elif cfg.backbone == "dino_teacher":
        teacher_state = ckpt["teacher_network_state"]
        vit_state = {
            k.replace("0.", "", 1): v
            for k, v in teacher_state.items()
            if k.startswith("0.")
        }
        vit.load_state_dict(vit_state, strict=True)
        logger.info("  ✓ Loaded ViT backbone from DINO teacher checkpoint.")

    # Freeze everything
    vit.requires_grad_(False)
    vit.eval()
    logger.info("  Backbone weights frozen.")
    return vit


# data loaders 

def _build_dataloaders(cfg: LinearProbeConfig):
    env_cfg = get_config(cfg.env)
    transform = LinearProbeTransforms(image_size=cfg.img_size)
    collate = CLEVRCollateFn(mode="linear_probe")

    train_ds = CLEVRDataset(
        env_cfg, mode="linear_probe", split="train", transform=transform
    )
    val_ds = CLEVRDataset(
        env_cfg, mode="linear_probe", split="val", transform=transform
    )

    batch_size = cfg.batch_size or env_cfg.batch_size

    dl_kwargs = {}
    if env_cfg.num_workers > 0 and env_cfg.prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = env_cfg.prefetch_factor

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=env_cfg.num_workers,
        pin_memory=env_cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
        **dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=env_cfg.num_workers,
        pin_memory=env_cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
        **dl_kwargs,
    )
    return train_dl, val_dl


# checkpoint I/O 

def _save_checkpoint(
    path: str,
    probe: LinearProbe,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    metrics_history: list,
    cfg: LinearProbeConfig,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "metrics_history": metrics_history,
        "probe_state": probe.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg": asdict(cfg),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"  Saved checkpoint → {path}")


def _load_checkpoint(
    path: str,
    probe: LinearProbe,
    optimizer,
    scheduler,
    device: torch.device,
):
    """
    Restores probe, optimizer, scheduler state in-place.

    Returns:
        (start_epoch, global_step, best_metric, metrics_history)
    """
    logger.info(f"Resuming linear-probe training from: {path}")
    ckpt = torch.load(path, map_location=device)

    probe.load_state_dict(ckpt["probe_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    best_metric = ckpt["best_metric"]
    metrics_history = ckpt.get("metrics_history", [])

    logger.info(
        f"  Resumed at epoch={start_epoch}, step={global_step}, "
        f"best_metric={best_metric:.4f}"
    )
    return start_epoch, global_step, best_metric, metrics_history


# main entry point 

def train_linear_probe(cfg: LinearProbeConfig) -> None:
    """
    Complete linear-probe training loop.

    1. Loads frozen backbone (CLIP / DINO).
    2. Pre-extracts features (CLS or GAP) for train & val splits →
       avoids running the ViT every epoch.
    3. Trains a single-layer LinearProbe on the cached features.
    4. Evaluates after each epoch; saves latest + best checkpoints.
    5. Fully resumable from the last saved checkpoint.

    Args:
        cfg: LinearProbeConfig controlling backbone, task, optimisation, etc.
    """
    tag = _run_tag(cfg)
    _set_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    logger.info(f"[LinearProbe:{tag}] Device: {device}")

    # 1. Build frozen backbone 
    backbone = _build_backbone(cfg, device)

    # 2. Build dataloaders 
    train_dl, val_dl = _build_dataloaders(cfg)
    logger.info(
        f"  Train samples: {len(train_dl.dataset)} | "
        f"Val samples: {len(val_dl.dataset)}"
    )

    # 3. Pre-extract features (run backbone once) 
    logger.info(f"  Extracting {cfg.representation.upper()} features from frozen backbone ...")
    train_feats, train_counts, train_colors = extract_features(
        backbone, train_dl, device, representation=cfg.representation
    )
    val_feats, val_counts, val_colors = extract_features(
        backbone, val_dl, device, representation=cfg.representation
    )
    logger.info(
        f"  Feature dim: {train_feats.shape[1]} | "
        f"Train: {train_feats.shape[0]} | Val: {val_feats.shape[0]}"
    )

    # 4. Set up probe 
    in_dim = train_feats.shape[1]
    multi_label = cfg.task == "color"
    num_classes = cfg.num_color_classes if multi_label else cfg.num_count_classes

    probe = LinearProbe(
        in_dim=in_dim,
        num_classes=num_classes,
        multi_label=multi_label,
    ).to(device)

    trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    logger.info(f"  Probe parameters (trainable): {trainable_params:,}")

    # 5. Loss, optimiser, scheduler 
    if cfg.task == "count":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Simple cosine-annealing per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    # 6. Resume handling 
    start_epoch = 0
    global_step = 0
    # For "count" higher accuracy is better; for "color" higher F1 is better
    best_metric = -math.inf
    metrics_history: list = []

    ckpt_dir = Path(cfg.checkpoint_dir) / tag
    latest_path = str(ckpt_dir / "checkpoint_latest.pt")
    best_path = str(ckpt_dir / "checkpoint_best.pt")

    # Auto-detect resume checkpoint
    resume_path = cfg.resume_checkpoint
    if resume_path is None and os.path.isfile(latest_path):
        resume_path = latest_path
        logger.info(f"  Auto-detected latest checkpoint at {latest_path}")

    if resume_path and os.path.isfile(resume_path):
        start_epoch, global_step, best_metric, metrics_history = _load_checkpoint(
            resume_path, probe, optimizer, scheduler, device
        )
    elif resume_path:
        logger.warning(
            f"  resume_checkpoint='{resume_path}' not found; starting fresh."
        )

    # 7. W&B logger 
    wandb_run_name = cfg.wandb_run_name
    if wandb_run_name == "probe-default":
        wandb_run_name = f"probe-{tag}"

    wandb_logger = WandbLogger(
        project_name=cfg.wandb_project,
        config=asdict(cfg),
        run_name=wandb_run_name,
        run_id=cfg.wandb_run_id,
        offline=cfg.wandb_offline,
    )

    def _do_save(path: str, epoch: int) -> None:
        _save_checkpoint(
            path=path,
            probe=probe,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            metrics_history=metrics_history,
            cfg=cfg,
        )

    # 8. Prepare mini-batch iteration on cached features 
    N_train = train_feats.shape[0]
    batch_size = cfg.batch_size or 256

    logger.info(
        f"[LinearProbe:{tag}] Starting training from epoch "
        f"{start_epoch + 1}/{cfg.epochs}"
    )

    for epoch in range(start_epoch, cfg.epochs):
        probe.train()
        epoch_loss = 0.0
        n_steps = 0

        # Shuffle indices each epoch
        perm = torch.randperm(N_train)

        pbar = tqdm(
            range(0, N_train, batch_size),
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [{tag}]",
            leave=False,
        )
        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, N_train)
            idx = perm[start_idx:end_idx]

            feats_b = train_feats[idx].to(device)
            logits = probe(feats_b)

            if cfg.task == "count":
                labels_b = train_counts[idx].to(device)
            else:
                labels_b = train_colors[idx].to(device)

            loss = criterion(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_steps += 1
            global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")

            if global_step % cfg.log_every_n_steps == 0:
                wandb_logger.log_metrics(
                    {
                        f"probe/{tag}/train_loss": loss_val,
                        f"probe/{tag}/lr": current_lr,
                    },
                    step=global_step,
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, n_steps)

        # Evaluate on train + val 
        train_metrics = evaluate_linear_probe(
            probe, train_feats, train_counts, train_colors,
            device, task=cfg.task, batch_size=batch_size,
        )
        val_metrics = evaluate_linear_probe(
            probe, val_feats, val_counts, val_colors,
            device, task=cfg.task, batch_size=batch_size,
        )

        metric_key = "accuracy" if cfg.task == "count" else "f1"
        train_metric_val = train_metrics[metric_key]
        val_metric_val = val_metrics[metric_key]

        logger.info(
            f"  Epoch {epoch + 1:>3}/{cfg.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"train_{metric_key}={train_metric_val:.4f} | "
            f"val_{metric_key}={val_metric_val:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        epoch_log = {
            "epoch": epoch + 1,
            f"probe/{tag}/train_loss_avg": avg_train_loss,
            f"probe/{tag}/train_{metric_key}": train_metric_val,
            f"probe/{tag}/val_{metric_key}": val_metric_val,
        }
        metrics_history.append(epoch_log)
        wandb_logger.log_metrics(epoch_log, step=global_step)

        # Best-checkpoint logic 
        if val_metric_val > best_metric:
            best_metric = val_metric_val
            _do_save(best_path, epoch)
            wandb_logger.log_model_artifact(best_path, artifact_name=f"probe-{tag}-best")
            logger.info(
                f"  ★ New best val_{metric_key}={best_metric:.4f} — saved best checkpoint."
            )

        _do_save(latest_path, epoch)

    logger.info(
        f"[LinearProbe:{tag}] Training complete. "
        f"Best val_{metric_key}={best_metric:.4f}"
    )
    wandb_logger.finish()


# convenience: run all 12 experiments 

def run_all_linear_probes(
    clip_checkpoint: str,
    dino_checkpoint: str,
    base_overrides: Optional[dict] = None,
) -> None:
    from configs.linear_probe_config import get_linear_probe_config

    base = base_overrides or {}

    for backbone in ("clip", "dino", "dino_teacher"):
        for representation in ("cls", "gap"):
            for task in ("count", "color"):
                tag = f"{backbone}_{representation}_{task}"
                logger.info("=" * 60)
                logger.info(f"  Starting linear probe: {tag}")
                logger.info("=" * 60)

                overrides = {
                    **base,
                    "backbone": backbone,
                    "representation": representation,
                    "task": task,
                    "clip_checkpoint": clip_checkpoint,
                    "dino_checkpoint": dino_checkpoint,
                    "wandb_run_name": f"probe-{tag}",
                }
                cfg = get_linear_probe_config(**overrides)
                train_linear_probe(cfg)
