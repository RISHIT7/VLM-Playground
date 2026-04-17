import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.dino_config import DINOTrainConfig
from configs.data_config import get_config
from data.clevr_dataset import CLEVRDataset, CLEVRCollateFn
from data.transforms import DINOMultiCropTransforms
from engine.evaluator import evaluate_dino_val_loss
from models.dino_heads import DINOEngine
from models.projection_heads import DINOProjectionHead
from models.vit_backbone import VisionTransformer
from utils.optimization import (
    build_optimizer,
    build_cosine_warmup_scheduler,
    cosine_schedule,
)
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


# Helpers


def _set_seed(seed: int) -> None:

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: DINOTrainConfig) -> DINOEngine:

    student_vit = VisionTransformer(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_num_heads,
        mlp_dim=cfg.vit_mlp_dim,
    )
    student_head = DINOProjectionHead(
        in_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        bottleneck_dim=cfg.head_bottleneck_dim,
        out_dim=cfg.dino_out_dim,
        use_bn=cfg.use_bn_in_head,
    )
    teacher_head = DINOProjectionHead(
        in_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        bottleneck_dim=cfg.head_bottleneck_dim,
        out_dim=cfg.dino_out_dim,
        use_bn=cfg.use_bn_in_head,
    )
    model = DINOEngine(
        vit_backbone=student_vit,
        student_head=student_head,
        teacher_head=teacher_head,
        out_dim=cfg.dino_out_dim,
        center_momentum=cfg.center_momentum,
    )
    return model


def _build_dataloaders(cfg: DINOTrainConfig):

    env_cfg = get_config(cfg.env)
    transform = DINOMultiCropTransforms(local_crops_number=cfg.local_crops_number)
    collate = CLEVRCollateFn(mode="dino")

    train_ds = CLEVRDataset(env_cfg, mode="dino", split="train", transform=transform)
    val_ds = CLEVRDataset(env_cfg, mode="dino", split="val", transform=transform)

    dl_kwargs = {}
    if env_cfg.num_workers > 0 and env_cfg.prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = env_cfg.prefetch_factor

    train_dl = DataLoader(
        train_ds,
        batch_size=env_cfg.batch_size,
        shuffle=True,
        num_workers=env_cfg.num_workers,
        pin_memory=env_cfg.pin_memory,
        drop_last=env_cfg.drop_last,
        collate_fn=collate,
        **dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=env_cfg.batch_size,
        shuffle=False,
        num_workers=env_cfg.num_workers,
        pin_memory=env_cfg.pin_memory,
        drop_last=False,
        collate_fn=collate,
        **dl_kwargs,
    )
    return train_dl, val_dl


# Checkpoint I/O
def _save_checkpoint(
    path: str,
    model: DINOEngine,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_score: float,
    metrics_history: list,
    cfg: DINOTrainConfig,
) -> None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "metrics_history": metrics_history,
        "student_network_state": model.student_network.state_dict(),
        "teacher_network_state": model.teacher_network.state_dict(),
        "center_state": model.center.data.clone(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg": asdict(cfg),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"  Saved checkpoint → {path}")


def _load_checkpoint(path: str, model: DINOEngine, optimizer, scheduler, device):
    """
    Restores all state from a saved checkpoint in-place.

    Returns:
        (start_epoch, global_step, best_score, metrics_history)
    """

    logger.info(f"Resuming DINO training from: {path}")
    ckpt = torch.load(path, map_location=device)

    model.student_network.load_state_dict(ckpt["student_network_state"])
    model.teacher_network.load_state_dict(ckpt["teacher_network_state"])
    model.center.data.copy_(ckpt["center_state"].to(device))

    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    best_score = ckpt["best_score"]
    metrics_history = ckpt.get("metrics_history", [])

    logger.info(
        f"  Resumed at epoch={start_epoch}, step={global_step}, "
        f"best_val_loss={best_score:.4f}"
    )
    return start_epoch, global_step, best_score, metrics_history


# Main trainer
def train_dino(cfg: DINOTrainConfig) -> None:
    """
    Full DINO training loop with:
      - EMA teacher (cosine-scheduled momentum).
      - Centering mechanism (EMA of teacher outputs).
      - Step-level cosine LR with linear warmup.
      - Exact resume at any interrupted point.
      - Dual checkpointing (latest + best).

    Args:
        cfg: DINOTrainConfig instance.
    """

    _set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    logger.info(f"[DINO] Training on device: {device}")

    model = _build_model(cfg).to(device)
    total_params = sum(
        p.numel() for p in model.student_network.parameters() if p.requires_grad
    )
    logger.info(f"  Trainable (student) parameters: {total_params:,}")

    train_dl, val_dl = _build_dataloaders(cfg)
    steps_per_epoch = len(train_dl)
    logger.info(f"  Train batches: {steps_per_epoch} | Val batches: {len(val_dl)}")

    optimizer = build_optimizer(
        model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        eps=cfg.eps,
    )
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    start_epoch = 0
    global_step = 0
    best_score = math.inf  # lower val_loss is better
    metrics_history = []

    if cfg.resume_checkpoint and os.path.isfile(cfg.resume_checkpoint):
        start_epoch, global_step, best_score, metrics_history = _load_checkpoint(
            cfg.resume_checkpoint, model, optimizer, scheduler, device
        )
    else:
        if cfg.resume_checkpoint:
            logger.warning(
                f"resume_checkpoint='{cfg.resume_checkpoint}' not found; starting fresh."
            )

    wandb_logger = WandbLogger(
        project_name=cfg.wandb_project,
        config=asdict(cfg),
        run_name=cfg.wandb_run_name,
        run_id=cfg.wandb_run_id,
        offline=cfg.wandb_offline,
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    latest_path = str(ckpt_dir / "checkpoint_latest.pt")
    best_path = str(ckpt_dir / "checkpoint_best.pt")

    def _do_save(path: str, epoch: int) -> None:
        _save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            metrics_history=metrics_history,
            cfg=cfg,
        )

    logger.info(f"[DINO] Starting training from epoch {start_epoch + 1}/{cfg.epochs}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        model.teacher_network.requires_grad_(False)  # teacher always frozen

        # Cosine-scheduled teacher EMA momentum
        teacher_momentum = cosine_schedule(
            cfg.momentum_teacher_start,
            cfg.momentum_teacher_end,
            epoch,
            cfg.epochs,
        )

        # Optionally ramp teacher temperature
        if (
            cfg.teacher_temp_warmup_epochs > 0
            and epoch < cfg.teacher_temp_warmup_epochs
        ):
            t = epoch / max(1, cfg.teacher_temp_warmup_epochs)
            teacher_temp = cfg.teacher_temp_start + t * (
                cfg.teacher_temp_end - cfg.teacher_temp_start
            )
        else:
            teacher_temp = cfg.teacher_temp_end

        epoch_loss = 0.0
        n_train_steps = 0

        pbar = tqdm(
            train_dl, desc=f"Epoch {epoch + 1}/{cfg.epochs} [DINO train]", leave=False
        )
        for batch in pbar:
            global_crops = batch["global_crops"].to(device)
            local_crops = batch["local_crops"].to(device)

            optimizer.zero_grad()

            teacher_out = model.forward_teacher(global_crops)  # no_grad inside
            student_out = model.forward_student(global_crops, local_crops)

            loss = model.compute_loss(
                student_out,
                teacher_out,
                student_temp=cfg.student_temp,
                teacher_temp=teacher_temp,
            )
            loss.backward()

            nn.utils.clip_grad_norm_(model.student_network.parameters(), max_norm=3.0)

            optimizer.step()
            scheduler.step()  # step-level

            # EMA updates (must happen AFTER the gradient step)
            model.update_teacher(momentum=teacher_momentum)
            model.update_center(teacher_out)

            loss_val = loss.item()
            current_lr = scheduler.get_last_lr()[0]
            epoch_loss += loss_val
            n_train_steps += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                lr=f"{current_lr:.2e}",
                mom=f"{teacher_momentum:.4f}",
            )

            if global_step % cfg.log_every_n_steps == 0:
                wandb_logger.log_metrics(
                    {
                        "train/loss": loss_val,
                        "train/lr": current_lr,
                        "train/teacher_momentum": teacher_momentum,
                        "train/teacher_temp": teacher_temp,
                    },
                    step=global_step,
                )

        avg_train_loss = epoch_loss / max(1, n_train_steps)

        val_metrics: dict = {}
        if (epoch + 1) % cfg.eval_every_n_epochs == 0:
            val_metrics = evaluate_dino_val_loss(
                model,
                val_dl,
                device,
                student_temp=cfg.student_temp,
                teacher_temp=teacher_temp,
            )
            model.train()

            val_loss = val_metrics["val_loss"]
            logger.info(
                f"  Epoch {epoch + 1:>3} | train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | teacher_mom={teacher_momentum:.4f}"
            )

            epoch_metrics = {
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                "val/val_loss": val_loss,
                "teacher/momentum": teacher_momentum,
                "teacher/temp": teacher_temp,
            }
            metrics_history.append(epoch_metrics)
            wandb_logger.log_metrics(epoch_metrics, step=global_step)

            if val_loss < best_score:
                best_score = val_loss
                _do_save(best_path, epoch)
                wandb_logger.log_model_artifact(best_path, artifact_name="dino-best")
                logger.info(
                    f"  ★ New best val_loss={best_score:.4f} — saved best checkpoint."
                )
        else:
            logger.info(
                f"  Epoch {epoch + 1:>3} | train_loss={avg_train_loss:.4f} | "
                f"teacher_mom={teacher_momentum:.4f}"
            )

        _do_save(latest_path, epoch)

    logger.info(f"[DINO] Training complete. Best val_loss = {best_score:.4f}")
    wandb_logger.finish()
