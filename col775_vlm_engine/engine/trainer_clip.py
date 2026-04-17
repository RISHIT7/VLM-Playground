import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.clip_config import CLIPTrainConfig
from configs.data_config import get_config
from data.clevr_dataset import CLEVRDataset, CLEVRCollateFn
from data.text_tokenizer import CLEVRTokenizer
from data.transforms import CLIPTransforms
from engine.evaluator import evaluate_clip_retrieval
from models.clip_heads import CLIPEngine
from models.vit_backbone import VisionTransformer
from models.text_encoder import TextEncoder
from utils.optimization import build_optimizer, build_cosine_warmup_scheduler
from utils.wandb_logger import WandbLogger
from utils.grad_hooks import compute_grad_metrics, clip_model_groups, clip_scalar_metrics

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


def _build_model(cfg: CLIPTrainConfig, vocab_size: int) -> CLIPEngine:

    vit = VisionTransformer(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_num_heads,
        mlp_dim=cfg.vit_mlp_dim,
    )
    text_enc = TextEncoder(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.text_depth,
        num_heads=cfg.text_num_heads,
        mlp_dim=cfg.text_mlp_dim,
        max_len=cfg.max_seq_len,
    )
    model = CLIPEngine(
        vit_backbone=vit,
        text_encoder=text_enc,
        embed_dim=cfg.embed_dim,
        proj_dim=cfg.proj_dim,
    )
    return model


def _build_dataloaders(cfg: CLIPTrainConfig, tokenizer: CLEVRTokenizer):

    env_cfg = get_config(cfg.env)
    transform = CLIPTransforms(image_size=cfg.img_size)
    collate = CLEVRCollateFn(mode="clip")

    train_ds = CLEVRDataset(
        env_cfg, mode="clip", split="train", transform=transform, tokenizer=tokenizer
    )
    val_ds = CLEVRDataset(
        env_cfg, mode="clip", split="val", transform=transform, tokenizer=tokenizer
    )

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
    model: CLIPEngine,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_score: float,
    metrics_history: list,
    cfg: CLIPTrainConfig,
    tokenizer_state: dict,
) -> None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "metrics_history": metrics_history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg": asdict(cfg),
        "tokenizer_state": tokenizer_state,
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"  Saved checkpoint → {path}")


def _load_checkpoint(path: str, model: CLIPEngine, optimizer, scheduler, device):
    """
    Loads a saved checkpoint and restores all state in-place.

    Returns:
        (start_epoch, global_step, best_score, metrics_history, tokenizer_state)
    """

    logger.info(f"Resuming CLIP training from: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt["epoch"] + 1  # resume from NEXT epoch
    global_step = ckpt["global_step"]
    best_score = ckpt["best_score"]
    metrics_history = ckpt.get("metrics_history", [])
    tokenizer_state = ckpt.get("tokenizer_state", {})

    logger.info(
        f"  Resumed at epoch={start_epoch}, step={global_step}, best_avg_R@1={best_score:.4f}"
    )
    return start_epoch, global_step, best_score, metrics_history, tokenizer_state


# Main trainer
def train_clip(cfg: CLIPTrainConfig) -> None:
    """
    Full CLIP training loop with:
      - Resume from latest or best checkpoint.
      - Step-level cosine LR schedule with linear warmup.
      - End-of-epoch validation (Recall@1/3) for best-checkpoint tracking.
      - W&B logging with graceful fallback.

    Args:
        cfg: CLIPTrainConfig instance; override fields to customise a run.
    """

    _set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    logger.info(f"[CLIP] Training on device: {device}")

    env_cfg = get_config(cfg.env)
    import os

    train_json = os.path.join(
        env_cfg.base_dir_part_a, "train", "clevr_train_captions.json"
    )
    tokenizer = CLEVRTokenizer(max_seq_len=cfg.max_seq_len)
    tokenizer.build_vocab(train_json)
    logger.info(f"  Vocabulary size: {tokenizer.vocab_size}")

    model = _build_model(cfg, vocab_size=tokenizer.vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {total_params:,}")

    train_dl, val_dl = _build_dataloaders(cfg, tokenizer)
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
    best_score = -math.inf
    metrics_history = []

    if cfg.resume_checkpoint and os.path.isfile(cfg.resume_checkpoint):
        start_epoch, global_step, best_score, metrics_history, tok_state = (
            _load_checkpoint(cfg.resume_checkpoint, model, optimizer, scheduler, device)
        )
        # Restore tokenizer vocab if it was saved
        if tok_state:
            tokenizer.word2idx = tok_state["word2idx"]
            tokenizer.idx2word = {int(k): v for k, v in tok_state["idx2word"].items()}
            tokenizer.vocab_size = tok_state["vocab_size"]
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

    def _tokenizer_state() -> dict:
        return {
            "word2idx": tokenizer.word2idx,
            "idx2word": tokenizer.idx2word,
            "vocab_size": tokenizer.vocab_size,
        }

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
            tokenizer_state=_tokenizer_state(),
        )

    logger.info(f"[CLIP] Starting training from epoch {start_epoch + 1}/{cfg.epochs}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_train_steps = 0

        pbar = tqdm(
            train_dl, desc=f"Epoch {epoch + 1}/{cfg.epochs} [CLIP train]", leave=False
        )
        for batch in pbar:
            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device)
            masks = batch["padding_mask"].to(device)
            captions = batch["raw_captions"]

            optimizer.zero_grad()
            img_feat, txt_feat = model(images, tokens, masks)
            loss = model.compute_loss(img_feat, txt_feat, captions)
            loss.backward()

            # ── Capture pre-clip gradient metrics ──
            grad_metrics = {}
            should_log = ((global_step + 1) % cfg.log_every_n_steps == 0)
            if should_log:
                groups = clip_model_groups(model)
                grad_metrics = compute_grad_metrics(groups)
                grad_metrics.update(clip_scalar_metrics(model))

            # Gradient clipping for stability
            clip_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if should_log:
                grad_metrics["grad_norm/post_clip"] = clip_norm.item()

            optimizer.step()
            scheduler.step()  # step-level scheduling

            loss_val = loss.item()
            current_lr = scheduler.get_last_lr()[0]
            epoch_loss += loss_val
            n_train_steps += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")

            if global_step % cfg.log_every_n_steps == 0:
                log_payload = {
                    "train/loss": loss_val,
                    "train/lr": current_lr,
                    "train/effective_update": current_lr * grad_metrics.get("grad_norm/total", 0.0),
                }
                log_payload.update(grad_metrics)
                wandb_logger.log_metrics(log_payload, step=global_step)

        avg_train_loss = epoch_loss / max(1, n_train_steps)

        val_metrics: dict = {}
        if (epoch + 1) % cfg.eval_every_n_epochs == 0:
            val_metrics = evaluate_clip_retrieval(model, val_dl, device)
            model.train()  # switch back after eval

            avg_r1 = val_metrics["avg_R@1"]
            logger.info(
                f"  Epoch {epoch + 1:>3} | train_loss={avg_train_loss:.4f} | "
                f"i2t_R@1={val_metrics['i2t_R@1']:.4f} "
                f"t2i_R@1={val_metrics['t2i_R@1']:.4f} "
                f"avg_R@1={avg_r1:.4f}"
            )

            epoch_metrics = {
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            metrics_history.append(epoch_metrics)
            wandb_logger.log_metrics(epoch_metrics, step=global_step)

            if avg_r1 > best_score:
                best_score = avg_r1
                _do_save(best_path, epoch)
                wandb_logger.log_model_artifact(best_path, artifact_name="clip-best")
                logger.info(
                    f"  ★ New best avg_R@1={best_score:.4f} — saved best checkpoint."
                )
        else:
            logger.info(f"  Epoch {epoch + 1:>3} | train_loss={avg_train_loss:.4f}")

        _do_save(latest_path, epoch)

    logger.info(f"[CLIP] Training complete. Best avg_R@1 = {best_score:.4f}")
    wandb_logger.finish()
