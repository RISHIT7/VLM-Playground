from dataclasses import dataclass
from typing import Optional


@dataclass
class VAEConfig:
    # Input
    img_size: int = 128
    in_channels: int = 3
    latent_channels: int = 4        # z is (B, 4, 16, 16)

    # Encoder/Decoder channel progression
    base_channels: int = 32         # 32 → 64 → 128 across 3 stages
    channel_multipliers: tuple = (1, 2, 4)
    num_res_blocks: int = 2

    # Loss
    kl_weight: float = 1e-6

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 100
    batch_size: int = 64
    warmup_epochs: int = 5

    # Environment
    env: str = "local_omen"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/vae"
    resume_checkpoint: Optional[str] = None

    # Logging
    wandb_project: str = "col775-a2-vae"
    wandb_run_name: str = "vae-default"
    wandb_offline: bool = False
    wandb_run_id: Optional[str] = None
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 5

    # Misc
    seed: int = 42
    device: str = "cuda"


def get_vae_config(**overrides) -> VAEConfig:
    cfg = VAEConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"VAEConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
