from dataclasses import dataclass
from typing import Optional


@dataclass
class LinearProbeConfig:
    # Backbone
    backbone: str = "clip"          # "clip" | "dino"
    representation: str = "cls"     # "cls" | "gap"
    task: str = "count"             # "count" | "color"

    # Architecture (must match pretrained backbone)
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    vit_depth: int = 12
    vit_num_heads: int = 6
    vit_mlp_dim: int = 1536

    # DINO-specific (only used when backbone == "dino")
    dino_out_dim: int = 4096
    dino_head_hidden_dim: int = 2048
    dino_head_bottleneck_dim: int = 256
    dino_use_bn_in_head: bool = False

    # Task-specific
    num_count_classes: int = 10     # 0..9 objects (or adjust to dataset)
    num_color_classes: int = 8      # CLEVR has 8 colours

    # Pretrained checkpoint paths
    clip_checkpoint: Optional[str] = None
    dino_checkpoint: Optional[str] = None

    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 256           # overrides env batch_size if set

    # Environment
    env: str = "local_omen"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/linear_probe"
    resume_checkpoint: Optional[str] = None

    # Logging
    wandb_project: str = "col775-a2-linear-probe"
    wandb_run_name: str = "probe-default"
    wandb_offline: bool = False
    wandb_run_id: Optional[str] = None
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1

    # Misc
    seed: int = 42
    device: str = "cuda"


def get_linear_probe_config(**overrides) -> LinearProbeConfig:
    """
    Returns a LinearProbeConfig, optionally overriding any field.

    Example:
        cfg = get_linear_probe_config(backbone="dino", task="color", representation="gap")
    """
    cfg = LinearProbeConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"LinearProbeConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
