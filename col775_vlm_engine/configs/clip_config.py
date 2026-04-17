from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIPTrainConfig:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    vit_depth: int = 12
    vit_num_heads: int = 6
    vit_mlp_dim: int = 1536
    text_depth: int = 6
    text_num_heads: int = 6
    text_mlp_dim: int = 1536
    proj_dim: int = 512  # shared CLIP projection dimension
    max_seq_len: int = 77  # tokenizer max length

    lr: float = 5e-4
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6

    epochs: int = 100
    warmup_epochs: int = 5  # linear warm-up for the first N epochs

    env: str = "local_omen"  # which EnvConfig to use (see data_config.py)

    checkpoint_dir: str = "checkpoints/clip"
    save_every_n_epochs: int = 1  # also saves latest each epoch

    wandb_project: str = "col775-a2-clip"
    wandb_run_name: str = "clip-vit-s-clevr"
    wandb_offline: bool = False
    # supply this to resume a previous W&B run
    wandb_run_id: Optional[str] = None

    # Path to a checkpoint to resume from; None means start fresh
    resume_checkpoint: Optional[str] = None

    eval_every_n_epochs: int = 1  # run val retrieval every N epochs

    seed: int = 42
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    log_every_n_steps: int = 5  # console + wandb step logging


def get_clip_config(**overrides) -> CLIPTrainConfig:
    """
    Returns a CLIPTrainConfig, optionally overriding any field.

    Example:
        cfg = get_clip_config(epochs=50, lr=1e-4, env="kaggle")
    """

    cfg = CLIPTrainConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"CLIPTrainConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
