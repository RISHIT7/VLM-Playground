import modal
import os
import sys

app = modal.App("col775-a2-dino")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=2.2.6",
        "pillow>=12.2.0",
        "scikit-learn>=1.7.2",
        "torch==2.5.0",
        "torchvision>=0.20.0",
        "tqdm>=4.67.3",
        "wandb>=0.26.0",
        "huggingface_hub",
    )
    .add_local_dir(
        local_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        remote_path="/workspace/col775_vlm_engine",
    )
    .add_local_dir(
        local_path=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "Dataset",
                "Part_Aa",
                "Probe-Datasets",
            )
        ),
        remote_path="/workspace/Probe-Datasets",
    )
)

dataset_volume = modal.Volume.from_name("col775-a2-dataset", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("checkpoints", create_if_missing=True)


def download_dataset():
    from huggingface_hub import snapshot_download
    import shutil
    import subprocess
    import glob
    import os

    if not os.path.exists("/data/A2_dataset/Part_A/train/clevr_train_captions.json"):
        if not glob.glob("/data/archive.tar.part*"):
            print("Downloading dataset aggr8/COL775-A2-Clevr-Extended-100k to /data...")
            snapshot_download(
                repo_id="aggr8/COL775-A2-Clevr-Extended-100k",
                repo_type="dataset",
                local_dir="/data",
            )
        else:
            print("Tarballs already present in /data, skipping download...")

        print("Extracting dataset from tarballs...")
        subprocess.run(
            "cat /data/archive.tar.part* | tar -xf - -C /data", shell=True, check=True
        )
        print("Extraction complete. Cleaning up tarballs...")
        subprocess.run(
            "rm -f /data/archive.tar.part* .gitattributes README.md",
            shell=True,
            check=True,
        )
    else:
        print("Dataset already present and extracted in /data.")

    os.makedirs("/data/A2_dataset/Part_Aa/Probe-Datasets", exist_ok=True)

    for filepath in glob.glob("/workspace/Probe-Datasets/*.json"):
        filename = os.path.basename(filepath)
        dst_dir = f"/data/A2_dataset/Part_Aa/Probe-Datasets/{filename}"
        if not os.path.exists(dst_dir):
            shutil.copy2(filepath, dst_dir)

        if "captions" in filename:
            dst_root = f"/data/A2_dataset/Part_Aa/{filename}"
            if not os.path.exists(dst_root):
                shutil.copy2(filepath, dst_root)


@app.function(
    cpu=8.0,
    image=image,
    gpu="L40S",
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("col775")],
    timeout=60 * 60 * 24,  # 24 hours timeout to ensure completion
)
def train_dino_remote(args: list[str]):
    download_dataset()

    os.chdir("/workspace/col775_vlm_engine")
    sys.path.insert(0, "/workspace/col775_vlm_engine")

    cli_args = [
        "main.py",
        "--mode",
        "dino",
        "--dino-checkpoint-dir",
        "/checkpoints/dino",
        "--dino-env",
        "modal",
    ] + args

    print(f"Running DINO training with args: {cli_args}")
    sys.argv = cli_args

    from main import main

    main()

    dataset_volume.commit()
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(*args: str):
    """
    Run locally to trigger Modal remote training.
    e.g. `modal run scripts/modal_train_dino.py -- --dino-epochs 10`
    """
    train_dino_remote.remote(list(args))
