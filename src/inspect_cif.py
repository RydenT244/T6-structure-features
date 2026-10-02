from pathlib import Path

import numpy as np
import biotite.structure.io.pdbx as pdbx


CIF_PATHS = [
    Path("data/ZDVHZKHFMHUZDU__rt5s0.cif"),
    Path("data/SBNKFTQSBPKMBZ__rt5s0.cif"),
    Path("data/BZAIRDCOXHAADH__rt5s0.cif"),
]


def load_structure(cif_path):
    """Load the first model from a CIF file."""
    cif_file = pdbx.CIFFile.read(cif_path)
    return pdbx.get_structure(cif_file, model=1)


def get_heme_iron(structure):
    """Return the single heme Fe atom."""
    fe_atoms = structure[
        (structure.res_name == "HEM")
        & (structure.element == "FE")
    ]

    if fe_atoms.array_length() == 0:
        raise ValueError("No heme Fe atom found.")

    if fe_atoms.array_length() > 1:
        raise ValueError("More than one heme Fe atom found.")

    return fe_atoms[0]


def get_ligand(structure):
    """Return ligand atoms from chain L."""
    ligand = structure[structure.chain_id == "L"]

    if ligand.array_length() == 0:
        raise ValueError("No ligand found in chain L.")

    return ligand


def count_protein_ligand_contacts(structure, cutoff=4.5):
    """
    Count protein-ligand heavy-atom pairs within cutoff Angstrom.
    """
    protein = structure[structure.chain_id == "A"]
    ligand = structure[structure.chain_id == "L"]

    protein = protein[protein.element != "H"]
    ligand = ligand[ligand.element != "H"]

    diff = (
        protein.coord[:, np.newaxis, :]
        - ligand.coord[np.newaxis, :, :]
    )

    distances = np.linalg.norm(diff, axis=2)

    return int(np.sum(distances <= cutoff))

def get_residue_contact_counts(structure, cutoff=4.5):
    """
    Count ligand contacts for each protein residue.

    Returns
    -------
    dict
        Keys are residue labels such as "PHE_108".
        Values are the number of protein-ligand atom-pair contacts
        involving that residue.
    """

    protein = structure[structure.chain_id == "A"]
    ligand = structure[structure.chain_id == "L"]

    # Keep only heavy atoms
    protein = protein[protein.element != "H"]
    ligand = ligand[ligand.element != "H"]

    # Pairwise protein-ligand distances
    diff = (
        protein.coord[:, np.newaxis, :]
        - ligand.coord[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        diff,
        axis=2
    )

    contact_mask = distances <= cutoff

    residue_contacts = {}

    for atom_index in range(protein.array_length()):

        atom_contact_count = int(
            np.sum(contact_mask[atom_index])
        )

        if atom_contact_count > 0:

            residue_name = str(
                protein.res_name[atom_index]
            )

            residue_id = int(
                protein.res_id[atom_index]
            )

            residue_label = (
                f"{residue_name}_{residue_id}"
            )

            residue_contacts[residue_label] = (
                residue_contacts.get(residue_label, 0)
                + atom_contact_count
            )

    return residue_contacts

def get_heme_plane_normal(structure):
    heme = structure[structure.res_name == "HEM"]
    heme_n = heme[heme.element == "N"]

    if heme_n.array_length() < 3:
        raise ValueError("Not enough heme nitrogen atoms to define a plane.")

    coords = heme_n.coord
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    _, _, vh = np.linalg.svd(centered)

    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    return normal

def calculate_fe_n_angle(structure):
    fe = get_heme_iron(structure)
    ligand = get_ligand(structure)

    ligand_n = ligand[ligand.element == "N"]

    if ligand_n.array_length() == 0:
        return np.nan

    n_distances = np.linalg.norm(
        ligand_n.coord - fe.coord,
        axis=1
    )

    closest_n_index = np.argmin(n_distances)
    closest_n = ligand_n[closest_n_index]

    fe_n_vector = closest_n.coord - fe.coord
    fe_n_vector = fe_n_vector / np.linalg.norm(fe_n_vector)

    plane_normal = get_heme_plane_normal(structure)

    cos_angle = np.abs(
        np.dot(fe_n_vector, plane_normal)
    )

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_angle))

    return float(angle)

def extract_basic_features(structure):
    """
    Extract basic T6 structure-derived features.
    """

    fe = get_heme_iron(structure)
    ligand = get_ligand(structure)
    fe_n_angle = calculate_fe_n_angle(structure)

    residue_contacts = get_residue_contact_counts(structure)

    n_contact_residues = len(residue_contacts)

    contacted_residues = ";".join(
        sorted(residue_contacts.keys())
    )   

    fe_coord = fe.coord

    # Fe to all ligand atoms
    all_distances = np.linalg.norm(
        ligand.coord - fe_coord,
        axis=1
    )

    closest_index = np.argmin(all_distances)
    closest_atom = ligand[closest_index]

    fe_min_dist = all_distances[closest_index]

    # Fe to ligand nitrogen atoms
    ligand_n = ligand[ligand.element == "N"]

    if ligand_n.array_length() > 0:
        n_distances = np.linalg.norm(
            ligand_n.coord - fe_coord,
            axis=1
        )

        min_n_index = np.argmin(n_distances)
        closest_n = ligand_n[min_n_index]

        fe_n_min_dist = n_distances[min_n_index]
        closest_n_name = str(closest_n.atom_name)

    else:
        fe_n_min_dist = np.nan
        closest_n_name = None

    # Protein-ligand contacts
    n_contacts = count_protein_ligand_contacts(structure)

    return {
        "fe_min_dist": float(fe_min_dist),
        "closest_atom_name": str(closest_atom.atom_name),
        "closest_atom_element": str(closest_atom.element),
        "fe_n_min_dist": float(fe_n_min_dist),
        "closest_n_name": closest_n_name,
        "fe_n_angle": fe_n_angle,
        "n_ligand_atoms": ligand.array_length(),
        "n_ligand_nitrogens": ligand_n.array_length(),
        "n_contacts": n_contacts,
        "n_contact_residues": n_contact_residues,
        "contacted_residues": contacted_residues,
    }


if __name__ == "__main__":

    for cif_path in CIF_PATHS:

        # Load first so any Biotite warnings appear before the label
        structure = load_structure(cif_path)

        print("\n" + "=" * 60)
        print("Structure:", cif_path.name)
        print("=" * 60)

        # Normal features
        features = extract_basic_features(structure)

        for feature, value in features.items():
            print(f"{feature}: {value}")

