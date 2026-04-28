from dataclasses import dataclass
from typing import Optional


@dataclass
class LDMConfig:
    # VAE (frozen during LDM training)
    vae_checkpoint: Optional[str] = None
    latent_size: int = 16           # spatial dim of VAE latent
    latent_channels: int = 4

    # U-Net architecture
    base_channels: int = 128
    channel_multipliers: tuple = (1, 2, 4)
    num_res_blocks: int = 2
    attention_levels: tuple = (1, 2)    # which downsampling levels get SpatialTransformers
    num_heads: int = 8

    # Text conditioning (frozen CLIP text encoder)
    clip_model_name: str = "openai/clip-vit-base-patch32"
    context_dim: int = 512          # CLIP text embedding dim

    # Diffusion
    num_timesteps: int = 500
    noise_schedule: str = "cosine"
    cfg_scale: float = 4.0
    cfg_drop_rate: float = 0.1      # probability of replacing text with null embedding

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 100
    batch_size: int = 32
    warmup_epochs: int = 5

    # Environment
    env: str = "local_omen"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/ldm"
    resume_checkpoint: Optional[str] = None

    # Logging
    wandb_project: str = "col775-a2-ldm"
    wandb_run_name: str = "ldm-default"
    wandb_offline: bool = False
    wandb_run_id: Optional[str] = None
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 10

    # Misc
    seed: int = 42
    device: str = "cuda"


def get_ldm_config(**overrides) -> LDMConfig:
    cfg = LDMConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"LDMConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
