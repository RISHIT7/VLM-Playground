from huggingface_hub import snapshot_download

repo_id = "aggr8/COL775-A2-Clevr-Extended-100k"
local_dir = "./Clevr"

snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

