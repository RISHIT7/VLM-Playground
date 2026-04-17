from dataclasses import dataclass
from typing import Optional


@dataclass
class DINOTrainConfig:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    vit_depth: int = 12
    vit_num_heads: int = 6
    vit_mlp_dim: int = 1536
    dino_out_dim: int = 4096  # projection head output dimension
    head_hidden_dim: int = 2048  # DINO head hidden layer
    head_bottleneck_dim: int = 256  # DINO head bottleneck layer
    use_bn_in_head: bool = False  # whether to insert BN in projection head

    local_crops_number: int = 8  # V local crops per image

    student_temp: float = 0.1
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.04
    teacher_temp_warmup_epochs: int = 0  # ramp teacher temp for first N epochs

    momentum_teacher_start: float = 0.996
    momentum_teacher_end: float = 1.0  # cosine schedule for teacher momentum
    center_momentum: float = 0.9

    lr: float = 5e-4
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6

    epochs: int = 100
    warmup_epochs: int = 5

    env: str = "local_omen"

    checkpoint_dir: str = "checkpoints/dino"
    save_every_n_epochs: int = 1

    wandb_project: str = "col775-a2-dino"
    wandb_run_name: str = "dino-vit-s-clevr"
    wandb_offline: bool = False
    wandb_run_id: Optional[str] = None

    resume_checkpoint: Optional[str] = None

    eval_every_n_epochs: int = 1

    seed: int = 42
    device: str = "cuda"
    log_every_n_steps: int = 50


def get_dino_config(**overrides) -> DINOTrainConfig:
    """
    Returns a DINOTrainConfig, optionally overriding any field.

    Example:
        cfg = get_dino_config(epochs=50, env="kaggle", local_crops_number=6)
    """

    cfg = DINOTrainConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"DINOTrainConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
