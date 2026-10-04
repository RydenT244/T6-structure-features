from pathlib import Path

import pandas as pd

from inspect_cif import (
    load_structure,
    get_heme_iron,
    get_ligand,
    extract_basic_features,
)


# ============================================================
# Configuration
# ============================================================

# T5 manifest containing metadata for each generated pose
MANIFEST_PATH = Path("data/t5_manifest.csv")

# Folder containing downloaded CYP3A4 CIF structures
CIF_DIR = Path("data/poses/CYP3A4")

# Output QC / inspection table
OUTPUT_PATH = Path("data/cyp3a4_cif_qc.csv")


# ============================================================
# Main inspection function
# ============================================================

def main():
    """
    Inspect all CYP3A4 structures listed in the T5 manifest.

    This script combines:
        1. Original T5 metadata
        2. Basic CIF quality checks
        3. Structure-derived features extracted from each pose

    The goal is to create one table that can be used to:
        - check whether structures loaded successfully
        - preserve which poses were already marked valid/invalid
        - compare extracted features across structures
        - identify unusual or failed structures
        - prepare the data for later modeling
    """

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest = pd.read_csv(MANIFEST_PATH)

    # Keep only CYP3A4 structures
    cyp3a4 = manifest[
        manifest["isoform"] == "CYP3A4"
    ].copy()

    print("CYP3A4 manifest rows:", len(cyp3a4))

    rows = []

    # --------------------------------------------------------
    # Loop through every CYP3A4 manifest row
    # --------------------------------------------------------

    for _, manifest_row in cyp3a4.iterrows():

        inchikey = manifest_row["inchikey_block"]

        # structure_path in the manifest may contain Windows-style
        # backslashes, so convert them before extracting the filename
        cif_name = Path(
            str(manifest_row["structure_path"]).replace("\\", "/")
        ).name

        cif_path = CIF_DIR / cif_name

        print(f"Inspecting {inchikey}: {cif_name}")

        # ----------------------------------------------------
        # Start with metadata from T5
        # ----------------------------------------------------

        result = {
            "inchikey_block": inchikey,
            "isoform": manifest_row["isoform"],
            "method": manifest_row["method"],
            "structure_path": manifest_row["structure_path"],

            # T5 quality / validity metadata
            "completed": manifest_row["completed"],
            "valid": manifest_row["valid"],
            "invalid_reason": manifest_row["invalid_reason"],
            "confidence": manifest_row["confidence"],
            "is_coordinated": manifest_row["is_coordinated"],

            # Existing T5 measurements for comparison
            "manifest_fe_min_dist": manifest_row["fe_min_dist"],
            "manifest_n_contacts": manifest_row["n_contacts"],

            # Local CIF information
            "cif_file": cif_name,
        }

        # ----------------------------------------------------
        # Check whether the CIF exists locally
        # ----------------------------------------------------

        if not cif_path.exists():
            result["status"] = "MISSING_CIF"
            result["error"] = "CIF file not found locally"

            rows.append(result)
            continue

        # ----------------------------------------------------
        # Attempt to load and inspect structure
        # ----------------------------------------------------

        try:
            structure = load_structure(cif_path)

            # Basic structure components
            fe = get_heme_iron(structure)
            ligand = get_ligand(structure)

            # Extract validated T6 features
            #
            # This now includes:
            #   - fe_min_dist
            #   - closest_atom_name
            #   - closest_atom_element
            #   - fe_n_min_dist
            #   - closest_n_name
            #   - fe_n_angle
            #   - n_ligand_atoms
            #   - n_ligand_nitrogens
            #   - n_contacts
            #   - n_contact_residues
            features = extract_basic_features(structure)

            # ------------------------------------------------
            # Add general CIF / structural QC information
            # ------------------------------------------------

            result.update({
                "n_atoms_total": structure.array_length(),

                # Save chain IDs as a compact string such as A,H,L
                "chains": ",".join(
                    sorted(set(structure.chain_id))
                ),

                "has_heme": "HEM" in set(structure.res_name),

                # get_heme_iron() already guarantees one Fe
                "n_fe_atoms": 1,

                "ligand_residue_names": ",".join(
                    sorted(set(ligand.res_name))
                ),

                # ------------------------------------------------
                # Extracted structure-derived features
                # ------------------------------------------------

                "fe_min_dist": features["fe_min_dist"],
                "closest_atom_name": features["closest_atom_name"],
                "closest_atom_element": features["closest_atom_element"],

                "fe_n_min_dist": features["fe_n_min_dist"],
                "closest_n_name": features["closest_n_name"],
                "fe_n_angle": features["fe_n_angle"],

                "n_ligand_atoms": features["n_ligand_atoms"],
                "n_ligand_nitrogens": features["n_ligand_nitrogens"],

                "n_contacts": features["n_contacts"],
                "n_contact_residues": features["n_contact_residues"],
                "contacted_residues": features["contacted_residues"],
                
                "ligand_centroid_fe_dist": features["ligand_centroid_fe_dist"],
                "ligand_atoms_within_4A_fe": features["ligand_atoms_within_4A_fe"],
                "ligand_atoms_within_5A_fe": features["ligand_atoms_within_5A_fe"],
                "ligand_atoms_within_6A_fe": features["ligand_atoms_within_6A_fe"],

                "mean_protein_neighbors_4A": features["mean_protein_neighbors_4A"],
                "mean_protein_neighbors_5A": features["mean_protein_neighbors_5A"],
                "mean_protein_neighbors_6A": features["mean_protein_neighbors_6A"],
                "max_protein_neighbors_5A": features["max_protein_neighbors_5A"],

                "hydrophobic_contact_residues": features["hydrophobic_contact_residues"],
                "aromatic_contact_residues": features["aromatic_contact_residues"],
                "polar_contact_residues": features["polar_contact_residues"],
                "positive_contact_residues": features["positive_contact_residues"],
                "negative_contact_residues": features["negative_contact_residues"],
                "other_contact_residues": features["other_contact_residues"],

                "status": "OK",
                "error": None,
            })

        except Exception as e:
            # Keep the row even when feature extraction fails.
            # This is important so failed poses are not silently removed.
            result["status"] = "FAILED"
            result["error"] = str(e)

        rows.append(result)

    # --------------------------------------------------------
    # Create inspection / QC table
    # --------------------------------------------------------

    results = pd.DataFrame(rows)

    # Save all structures, including failed and invalid poses
    results.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nFinished.")
    print("Rows written:", len(results))
    print("Saved to:", OUTPUT_PATH)

    print("\nStatus counts:")
    print(results["status"].value_counts(dropna=False))

    print("\nManifest validity counts:")
    print(results["valid"].value_counts(dropna=False))

    print("\nCoordination counts:")
    print(results["is_coordinated"].value_counts(dropna=False))


if __name__ == "__main__":
    main()