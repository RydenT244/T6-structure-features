from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "xX-its-amit-Xx/cyp-cofold-poses"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


FILES_TO_DOWNLOAD = [
    "t5_manifest.csv",
    "compounds.csv",
]


for filename in FILES_TO_DOWNLOAD:

    print(f"Downloading {filename}...")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )

    print(f"Saved to: {downloaded_path}")


print("\nFinished downloading required CSV files.")