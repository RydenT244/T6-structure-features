from pathlib import Path

import pandas as pd

from inspect_cif import load_structure, extract_basic_features


MANIFEST_PATH = Path("data/t5_manifest.csv")
OUTPUT_PATH = Path("data/cyp3a4_structure_features.csv")


def main():
    # Load manifest
    manifest = pd.read_csv(MANIFEST_PATH)

    # Keep CYP3A4 structures only
    cyp3a4 = manifest[
        manifest["isoform"] == "CYP3A4"
    ].copy()

    # Start with valid, completed structures
    cyp3a4 = cyp3a4[
        (cyp3a4["completed"] == True)
        & (cyp3a4["valid"] == True)
    ].copy()

    print("CYP3A4 valid structures:", len(cyp3a4))

    rows = []

    for _, row in cyp3a4.iterrows():
        inchikey = row["inchikey_block"]

        # The manifest contains a Windows-style path.
        # We only need the CIF filename locally.
        cif_name = Path(
            str(row["structure_path"]).replace("\\", "/")
        ).name

        cif_path = Path("data/poses/CYP3A4") / cif_name

        print(f"Processing {inchikey}: {cif_name}")

        if not cif_path.exists():
            print("  Missing CIF, skipping.")
            continue

        try:
            structure = load_structure(cif_path)

            features = extract_basic_features(structure)

            result = {
                "inchikey_block": inchikey,
                "isoform": row["isoform"],
                "method": row["method"],
                "confidence": row["confidence"],
                "manifest_fe_min_dist": row["fe_min_dist"],
                "manifest_n_contacts": row["n_contacts"],
                "manifest_is_coordinated": row["is_coordinated"],
                **features,
            }

            rows.append(result)

        except Exception as e:
            print(f"  Failed: {e}")

    results = pd.DataFrame(rows)

    results.to_csv(OUTPUT_PATH, index=False)

    print("\nFinished.")
    print("Extracted structures:", len(results))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()