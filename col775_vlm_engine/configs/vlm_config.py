from dataclasses import dataclass
import torch

@dataclass
class VLMConfig:
    data_root: str = "../data/A2_dataset/"

    num_gpus: int = 1
    device: str = "cuda"
    llm_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    vit_dim: int = 384
    expansion_factor: int = 2 
    
    weight_decay: float = 0.0      
    warmup_ratio: float = 0.03     
    
    # --- STAGE 1 TARGETS ---
    stage1_epochs: int = 2
    stage1_lr: float = 2e-3
    stage1_target_bs: int = 128
    stage1_per_device_bs: int = 32  # (4 fits 16/24GB GPUs)
    
    # --- STAGE 2 TARGETS ---
    stage2_epochs: int = 3
    stage2_lr: float = 2e-5
    stage2_target_bs: int = 32
    stage2_per_device_bs: int = 4
    lora_rank: int = 16            
    lora_alpha: int = 32
    
    # Infrastructure
    num_workers: int = 8
    wandb_project: str = "col775-a2-vlm"
    checkpoint_dir: str = "checkpoints/vlm"
    log_every_n_steps: int = 1

    @property
    def stage1_grad_accum(self) -> int:
        """Dynamically calculates accumulation steps to hit 128 Effective BS"""
        return max(1, self.stage1_target_bs // (self.stage1_per_device_bs * self.num_gpus))

    @property
    def stage2_grad_accum(self) -> int:
        """Dynamically calculates accumulation steps to hit 32 Effective BS"""
        return max(1, self.stage2_target_bs // (self.stage2_per_device_bs * self.num_gpus))