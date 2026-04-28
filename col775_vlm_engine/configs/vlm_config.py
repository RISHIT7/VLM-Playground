from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMConfig:
    # Stage
    stage: int = 1                  # 1 = alignment, 2 = fine-tune

    # Vision encoder (frozen, loaded from Part A checkpoint)
    vision_backbone: str = "clip"   # best performing from Part A
    vision_checkpoint: Optional[str] = None
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    vit_depth: int = 12
    vit_num_heads: int = 6
    vit_mlp_dim: int = 1536

    # Projector MLP (reverse-bottleneck)
    projector_hidden_dim: int = 256
    llm_hidden_dim: int = 3584      # Qwen3-4B hidden size

    # LLM
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 256
    max_seq_len: int = 512

    # LoRA (Stage-2 only)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Mixed precision / memory
    use_fp16: bool = True
    gradient_checkpointing: bool = True

    # Environment
    env: str = "local_omen"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/vlm"
    resume_checkpoint: Optional[str] = None

    # Logging
    wandb_project: str = "col775-a2-vlm"
    wandb_run_name: str = "vlm-default"
    wandb_offline: bool = False
    wandb_run_id: Optional[str] = None
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1

    # Misc
    seed: int = 42
    device: str = "cuda"


def get_vlm_config(**overrides) -> VLMConfig:
    cfg = VLMConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"VLMConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg
