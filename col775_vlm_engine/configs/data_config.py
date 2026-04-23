from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os

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
    file_path = os.path.abspath(__file__)
    if env == "local_mac":
        # check kr lena bhai, I'm not sure what the exact path will be on your machine, but it should be something like this: reply "sahi hai :)"
        part_a_path = Path(file_path).parent.parent.parent / "data" / "A2_dataset" / "Part_A"
        part_aa_path = Path(file_path).parent.parent.parent / "data" / "A2_dataset" / "Part_Aa"
        return EnvConfig(
            env_name="local",
            base_dir_part_a=str(part_a_path),
            base_dir_part_aa=str(part_aa_path),
            batch_size=64,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None,
            drop_last=False,
        )
    elif env == "local_omen":
        part_a_path = Path(file_path).parent.parent.parent / "Dataset" / "Part_A"
        part_aa_path = Path(file_path).parent.parent.parent / "Dataset" / "Part_Aa"
        return EnvConfig(
            env_name="local",
            base_dir_part_a=str(part_a_path),
            base_dir_part_aa=str(part_aa_path),
            batch_size=32,
            num_workers=6,
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
            base_dir_part_a="/scratch/scai/phd/aiz228170/COL775-A2-2026/dataset/A2_dataset/Part_A",
            base_dir_part_aa="/scratch/scai/phd/aiz228170/COL775-A2-2026/dataset/A2_dataset/Part_Aa",
            batch_size=256,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")