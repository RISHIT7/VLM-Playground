import modal
import os

# 1. Define the Environment (Image)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "peft", 
        "accelerate", "wandb", "tqdm", "pillow", "pyyaml"
    )
)

app = modal.App("vlm-training")

# 2. Define Persistent Storage for Data and Checkpoints
# Run 'modal volume create vlm-data' and 'modal volume create vlm-checkpoints' first
data_volume = modal.Volume.from_name("vlm-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("vlm-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100", # Or A100, L4, T4
    timeout=3600 * 48, # 48 hours
    volumes={
        "/data": data_volume,
        "/checkpoints": ckpt_volume
    },
    mounts=[modal.Mount.from_local_dir("./col775_vlm_engine", remote_path="/root/col775_vlm_engine")],
    secrets=[modal.Secret.from_dict({"WANDB_API_KEY": "your_actual_wandb_key"})]
)
def train():
    # Set PYTHONPATH so remote imports work
    os.environ["PYTHONPATH"] = "/root/col775_vlm_engine"
    
    # Change to the directory containing main.py
    os.chdir("/root/col775_vlm_engine")
    
    # Construct the command (using actual paths on Modal volumes)
    command = (
        "python main.py "
        "--mode vlm_stage1 "
        "--data-root /data "
        "--captions-json /data/train.jsonl "
        "--vlm-device cuda "
        "--num-gpus 1 "
        "--vlm-batch-size 32 "
        "--vlm-epochs 1 "
        "--vlm-vit-ckpt /data/checkpoint_best.pt "
        "--vlm-checkpoint-dir /checkpoints "
        "--vlm-llm-id 'Qwen/Qwen3-4B-Instruct-2507-FP8'"
    )
    
    print(f"Executing: {command}")
    os.system(command)
    
    # Commit changes to volume so checkpoints persist
    ckpt_volume.commit()

@app.local_entrypoint()
def main():
    train.remote()
