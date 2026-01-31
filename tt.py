import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bezdarnost/DF2K-bicubic",
    repo_type="dataset",
    local_dir="./datasets/DF2K-bicubic",
    local_dir_use_symlinks=False,
)
df_path = os.path.abspath("./datasets/DF2K-bicubic/x4")
for filename in os.listdir(df_path):
    file_path = os.path.join(df_path, filename)
    split_filename = filename.split("x")
    if len(split_filename) != 1:
        out_filename = split_filename[0] + ".png"
        out_path = os.path.join(df_path, out_filename)
        os.rename(file_path, out_path)
