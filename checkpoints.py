from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil


def retrieve_checkpoints():
    """
    With this function, all weights required for models are going to be downloaded
    """
    repo_id = "franciszzj/Leffa"
    exclude_list = [
        "pose_transfer.pth",
        "examples/",
        "assets/",
        "README.md",
        "schp",
        ".gitattributes",
        "virtual_tryon.pth",
    ]
    destination_folder = "./checkpoints"

    os.makedirs(destination_folder, exist_ok=True)

    repo_files = list_repo_files(repo_id)

    for file in repo_files:
        if any(file.startswith(exclude) for exclude in exclude_list):
            continue

        file_path = hf_hub_download(repo_id=repo_id, filename=file)

        subdir = os.path.join(destination_folder, os.path.dirname(file))
        os.makedirs(subdir, exist_ok=True)

        destination_path = os.path.join(destination_folder, file)

        shutil.copy2(file_path, destination_path)

    print(
        f"Downloaded all files except those in the exclude list into '{destination_folder}'"
    )


retrieve_checkpoints()
