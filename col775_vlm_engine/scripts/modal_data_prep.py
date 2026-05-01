import modal
import os

app = modal.App("col775-a2-data-prep")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub")
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


@app.function(
    cpu=8.0,
    image=image,
    volumes={"/data": dataset_volume},
    timeout=60 * 60 * 12,  # 12 hours timeout to ensure completion
)
def prepare_dataset_remote():
    from huggingface_hub import snapshot_download
    import shutil
    import subprocess
    import glob

    if not os.path.exists("/data/Part_A/train/clevr_train_captions.json"):
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

    os.makedirs("/data/Part_Aa/Probe-Datasets", exist_ok=True)

    for filepath in glob.glob("/workspace/Probe-Datasets/*.json"):
        filename = os.path.basename(filepath)
        dst_dir = f"/data/Part_Aa/Probe-Datasets/{filename}"
        if not os.path.exists(dst_dir):
            shutil.copy2(filepath, dst_dir)

        if "captions" in filename:
            dst_root = f"/data/Part_Aa/{filename}"
            if not os.path.exists(dst_root):
                shutil.copy2(filepath, dst_root)

    dataset_volume.commit()
    print("Dataset perfectly extracted and committed to Volume!")


@app.local_entrypoint()
def main():
    """
    Run locally to trigger Modal remote data prep on a cheap CPU.
    e.g. `modal run scripts/modal_data_prep.py`
    """
    prepare_dataset_remote.remote()
