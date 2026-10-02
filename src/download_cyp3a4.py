from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


MANIFEST_PATH = Path("data/t5_manifest.csv")
OUTPUT_DIR = Path("data/poses/CYP3A4")

REPO_ID = "xX-its-amit-Xx/cyp-cofold-poses"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    # Keep CYP3A4 only
    cyp3a4 = manifest[
        manifest["isoform"] == "CYP3A4"
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CYP3A4 rows:", len(cyp3a4))

    for _, row in cyp3a4.iterrows():
        inchikey = row["inchikey_block"]

        # Get CIF filename from manifest path
        cif_name = Path(
            str(row["structure_path"]).replace("\\", "/")
        ).name

        remote_path = f"t5_bakeoff/poses/CYP3A4/{cif_name}"

        local_path = OUTPUT_DIR / cif_name

        # Skip files already downloaded
        if local_path.exists():
            print(f"Already exists: {cif_name}")
            continue

        print(f"Downloading {inchikey}: {cif_name}")

        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_path,
            )

            # Copy downloaded cached file into our project
            local_path.write_bytes(
                Path(downloaded_path).read_bytes()
            )

        except Exception as e:
            print(f"FAILED {cif_name}: {e}")

    print("\nFinished downloading CYP3A4 poses.")


if __name__ == "__main__":
    main()