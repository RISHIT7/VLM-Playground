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

# Run all 8 linear probes (CLIP + DINO × CLS + GAP × count + color)
python main.py --mode linear_probe \
    --probe-clip-ckpt checkpoints/clip/checkpoint_best.pt \
    --probe-dino-ckpt checkpoints/dino/checkpoint_best.pt

# Run a single probe
python main.py --mode linear_probe \
    --probe-backbone clip --probe-repr cls --probe-task count \
    --probe-clip-ckpt checkpoints/clip/checkpoint_best.pt
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
        "--mode", required=True,
        choices=["clip", "dino", "both", "linear_probe"],
        help="Which model(s) to train or evaluate.",
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

    probe = p.add_argument_group("Linear Probe")
    probe.add_argument("--probe-clip-ckpt",     type=str, default=None,
                       help="Path to trained CLIP checkpoint for probing.")
    probe.add_argument("--probe-dino-ckpt",     type=str, default=None,
                       help="Path to trained DINO checkpoint for probing.")
    probe.add_argument("--probe-backbone",      type=str, default=None,
                       choices=["clip", "dino", "dino_teacher"],
                       help="Run probe for a single backbone (default: all three).")
    probe.add_argument("--probe-repr",          type=str, default=None,
                       choices=["cls", "gap"],
                       help="Run probe for a single representation (default: both).")
    probe.add_argument("--probe-task",          type=str, default=None,
                       choices=["count", "color"],
                       help="Run probe for a single task (default: both).")
    probe.add_argument("--probe-epochs",        type=int, default=None)
    probe.add_argument("--probe-lr",            type=float, default=None)
    probe.add_argument("--probe-batch-size",    type=int, default=None)
    probe.add_argument("--probe-env",           type=str, default=None)
    probe.add_argument("--probe-checkpoint-dir",type=str, default=None)
    probe.add_argument("--probe-wandb-offline", action="store_true")

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


def _run_linear_probe(args: argparse.Namespace) -> None:
    from configs.linear_probe_config import get_linear_probe_config
    from engine.trainer_linear_probe import train_linear_probe

    # Collect base overrides shared across all runs
    base: dict = {}
    if args.seed                is not None: base["seed"]           = args.seed
    if args.device              is not None: base["device"]         = args.device
    if args.probe_epochs        is not None: base["epochs"]         = args.probe_epochs
    if args.probe_lr            is not None: base["lr"]             = args.probe_lr
    if args.probe_batch_size    is not None: base["batch_size"]     = args.probe_batch_size
    if args.probe_env           is not None: base["env"]            = args.probe_env
    if args.probe_checkpoint_dir is not None: base["checkpoint_dir"] = args.probe_checkpoint_dir
    if args.probe_wandb_offline:              base["wandb_offline"]  = True

    # If all three selectors are given, run a single experiment
    if args.probe_backbone and args.probe_repr and args.probe_task:
        overrides = {
            **base,
            "backbone": args.probe_backbone,
            "representation": args.probe_repr,
            "task": args.probe_task,
            "clip_checkpoint": args.probe_clip_ckpt,
            "dino_checkpoint": args.probe_dino_ckpt,
        }
        cfg = get_linear_probe_config(**overrides)
        train_linear_probe(cfg)
    else:
        # Run all 12 combinations (or a filtered subset)
        backbones = [args.probe_backbone] if args.probe_backbone else ["clip", "dino", "dino_teacher"]
        representations = [args.probe_repr] if args.probe_repr else ["cls", "gap"]
        tasks = [args.probe_task] if args.probe_task else ["count", "color"]

        for backbone in backbones:
            for representation in representations:
                for task in tasks:
                    tag = f"{backbone}_{representation}_{task}"
                    logger.info("=" * 60)
                    logger.info(f"  Starting linear probe: {tag}")
                    logger.info("=" * 60)
                    overrides = {
                        **base,
                        "backbone": backbone,
                        "representation": representation,
                        "task": task,
                        "clip_checkpoint": args.probe_clip_ckpt,
                        "dino_checkpoint": args.probe_dino_ckpt,
                        "wandb_run_name": f"probe-{tag}",
                    }
                    cfg = get_linear_probe_config(**overrides)
                    train_linear_probe(cfg)


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

    if mode == "linear_probe":
        _run_linear_probe(args)


if __name__ == "__main__":
    main()
