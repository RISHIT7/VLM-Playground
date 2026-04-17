"""
Usage examples
--------------
# Train CLIP only
python main.py --mode clip

# Train DINO only
python main.py --mode dino

# Train both sequentially (CLIP first, then DINO)
python main.py --mode both

# Resume CLIP from latest checkpoint
python main.py --mode clip --clip-resume checkpoints/clip/checkpoint_latest.pt

# Resume DINO from best checkpoint
python main.py --mode dino --dino-resume checkpoints/dino/checkpoint_best.pt

# Override a config field from CLI
python main.py --mode clip --clip-epochs 50 --clip-lr 1e-4 --clip-env kaggle

# Run offline with W&B disabled (useful for HPC without internet)
python main.py --mode both --clip-wandb-offline --dino-wandb-offline
"""

import argparse
import logging
import sys

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
    handlers= [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="COL775 A2 VLM Training — CLIP and/or DINO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode", required=True, choices=["clip", "dino", "both"],
        help="Which model(s) to train.",
    )

    p.add_argument("--seed",   type=int, default=None, help="Global random seed.")
    p.add_argument("--device", type=str, default=None, help="Device: cuda | cpu | mps.")

    clip = p.add_argument_group("CLIP")
    clip.add_argument("--clip-resume",        type=str,   default=None,
                      help="Path to CLIP checkpoint to resume from.")
    clip.add_argument("--clip-epochs",        type=int,   default=None)
    clip.add_argument("--clip-lr",            type=float, default=None)
    clip.add_argument("--clip-weight-decay",  type=float, default=None)
    clip.add_argument("--clip-warmup-epochs", type=int,   default=None)
    clip.add_argument("--clip-env",           type=str,   default=None,
                      help="Environment key: local_omen | local_mac | kaggle | hpc.")
    clip.add_argument("--clip-batch-size",    type=int,   default=None,
                      help="Override batch size (env config value used otherwise).")
    clip.add_argument("--clip-checkpoint-dir",type=str,   default=None)
    clip.add_argument("--clip-wandb-project", type=str,   default=None)
    clip.add_argument("--clip-wandb-run-name",type=str,   default=None)
    clip.add_argument("--clip-wandb-run-id",  type=str,   default=None,
                      help="W&B run ID to resume a crashed run.")
    clip.add_argument("--clip-wandb-offline", action="store_true",
                      help="Force W&B offline mode for CLIP run.")
    clip.add_argument("--clip-eval-every",    type=int,   default=None,
                      help="Run CLIP val retrieval every N epochs.")

    dino = p.add_argument_group("DINO")
    dino.add_argument("--dino-resume",         type=str,   default=None,
                      help="Path to DINO checkpoint to resume from.")
    dino.add_argument("--dino-epochs",         type=int,   default=None)
    dino.add_argument("--dino-lr",             type=float, default=None)
    dino.add_argument("--dino-weight-decay",   type=float, default=None)
    dino.add_argument("--dino-warmup-epochs",  type=int,   default=None)
    dino.add_argument("--dino-env",            type=str,   default=None)
    dino.add_argument("--dino-checkpoint-dir", type=str,   default=None)
    dino.add_argument("--dino-wandb-project",  type=str,   default=None)
    dino.add_argument("--dino-wandb-run-name", type=str,   default=None)
    dino.add_argument("--dino-wandb-run-id",   type=str,   default=None)
    dino.add_argument("--dino-wandb-offline",  action="store_true")
    dino.add_argument("--dino-eval-every",     type=int,   default=None)
    dino.add_argument("--dino-local-crops",    type=int,   default=None,
                      help="Number of local DINO crops (default 8).")
    dino.add_argument("--dino-student-temp",   type=float, default=None)

    return p.parse_args()


def _build_clip_cfg(args: argparse.Namespace):
    from configs.clip_config import get_clip_config
    overrides = {}
    if args.seed              is not None: overrides["seed"]              = args.seed
    if args.device            is not None: overrides["device"]            = args.device
    if args.clip_resume       is not None: overrides["resume_checkpoint"] = args.clip_resume
    if args.clip_epochs       is not None: overrides["epochs"]            = args.clip_epochs
    if args.clip_lr           is not None: overrides["lr"]                = args.clip_lr
    if args.clip_weight_decay is not None: overrides["weight_decay"]      = args.clip_weight_decay
    if args.clip_warmup_epochs is not None: overrides["warmup_epochs"]    = args.clip_warmup_epochs
    if args.clip_env          is not None: overrides["env"]               = args.clip_env
    if args.clip_checkpoint_dir  is not None: overrides["checkpoint_dir"]    = args.clip_checkpoint_dir
    if args.clip_wandb_project   is not None: overrides["wandb_project"]      = args.clip_wandb_project
    if args.clip_wandb_run_name  is not None: overrides["wandb_run_name"]     = args.clip_wandb_run_name
    if args.clip_wandb_run_id    is not None: overrides["wandb_run_id"]       = args.clip_wandb_run_id
    if args.clip_wandb_offline:               overrides["wandb_offline"]      = True
    if args.clip_eval_every      is not None: overrides["eval_every_n_epochs"] = args.clip_eval_every
    return get_clip_config(**overrides)


def _build_dino_cfg(args: argparse.Namespace):
    from configs.dino_config import get_dino_config
    overrides = {}
    if args.seed               is not None: overrides["seed"]              = args.seed
    if args.device             is not None: overrides["device"]            = args.device
    if args.dino_resume        is not None: overrides["resume_checkpoint"] = args.dino_resume
    if args.dino_epochs        is not None: overrides["epochs"]            = args.dino_epochs
    if args.dino_lr            is not None: overrides["lr"]                = args.dino_lr
    if args.dino_weight_decay  is not None: overrides["weight_decay"]      = args.dino_weight_decay
    if args.dino_warmup_epochs is not None: overrides["warmup_epochs"]     = args.dino_warmup_epochs
    if args.dino_env           is not None: overrides["env"]               = args.dino_env
    if args.dino_checkpoint_dir  is not None: overrides["checkpoint_dir"]     = args.dino_checkpoint_dir
    if args.dino_wandb_project   is not None: overrides["wandb_project"]       = args.dino_wandb_project
    if args.dino_wandb_run_name  is not None: overrides["wandb_run_name"]      = args.dino_wandb_run_name
    if args.dino_wandb_run_id    is not None: overrides["wandb_run_id"]        = args.dino_wandb_run_id
    if args.dino_wandb_offline:               overrides["wandb_offline"]       = True
    if args.dino_eval_every      is not None: overrides["eval_every_n_epochs"] = args.dino_eval_every
    if args.dino_local_crops     is not None: overrides["local_crops_number"]  = args.dino_local_crops
    if args.dino_student_temp    is not None: overrides["student_temp"]        = args.dino_student_temp
    return get_dino_config(**overrides)


def main() -> None:
    args = parse_args()
    mode = args.mode

    if mode in ("clip", "both"):
        from engine.trainer_clip import train_clip
        clip_cfg = _build_clip_cfg(args)
        logger.info("=" * 60)
        logger.info("  Starting CLIP training")
        logger.info("=" * 60)
        train_clip(clip_cfg)

    if mode in ("dino", "both"):
        from engine.trainer_dino import train_dino
        dino_cfg = _build_dino_cfg(args)
        logger.info("=" * 60)
        logger.info("  Starting DINO training")
        logger.info("=" * 60)
        train_dino(dino_cfg)


if __name__ == "__main__":
    main()
