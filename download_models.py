from huggingface_hub import snapshot_download

def download_models():
    # Download CatVTON model
    print("Downloading CatVTON model...")
    catvton_path = snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir="Models/CatVTON",
        local_dir_use_symlinks=False
    )
    print(f"Downloaded CatVTON to {catvton_path}")

    # Download DensePose model
    print("Downloading DensePose model...")
    densepose_path = snapshot_download(
        repo_id="Hoang-Anh-Pham/densepose",
        local_dir="Models/DensePose",
        local_dir_use_symlinks=False
    )
    print(f"Downloaded DensePose to {densepose_path}")

    # Download SCHP model
    print("Downloading SCHP model...")
    schp_path = snapshot_download(
        repo_id="Hoang-Anh-Pham/schp",
        local_dir="Models/SCHP",
        local_dir_use_symlinks=False
    )
    print(f"Downloaded SCHP to {schp_path}")

if __name__ == "__main__":
    download_models() 