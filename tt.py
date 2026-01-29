from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bezdarnost/DF2K-bicubic",
    repo_type="dataset",
    local_dir="./datasets/DF2K-bicubic",
    local_dir_use_symlinks=False,
)
