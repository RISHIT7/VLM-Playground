from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvConfig:
    env_name: str
    base_dir_part_a: str
    base_dir_part_aa: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: Optional[int] = None
    drop_last: bool = False

def get_config(env: str = "local") -> EnvConfig:
    if env == "local_mac":
        return EnvConfig(
            env_name="local",
            base_dir_part_a="../../data/A2_dataset/Part_A",
            base_dir_part_aa="../../data/A2_dataset/Part_Aa",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None,
            drop_last=False,
        )
    elif env == "local_omen":
        return EnvConfig(
            env_name="local",
            base_dir_part_a="data/Part_A",        # change to actual path
            base_dir_part_aa="data/Part_Aa",      # change to actual path
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None,
            drop_last=False,
        )
    elif env == "kaggle":
        return EnvConfig(
            env_name="kaggle",
            base_dir_part_a="/kaggle/input/clevr/PartA",
            base_dir_part_aa="/kaggle/input/clevr/PartAa",
            batch_size=64,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=False,
        )
    elif env == "hpc":
        # Update these paths to match your HPC scratch space
        return EnvConfig(
            env_name="hpc",
            base_dir_part_a="/scratch/username/clevr/Part_A",
            base_dir_part_aa="/scratch/username/clevr/Part_Aa",
            batch_size=256,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")