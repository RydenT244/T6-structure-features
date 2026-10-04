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

def calculate_heme_proximity_features(structure):
    """
    Calculate ligand proximity features relative to the heme iron.

    Returns:
        ligand_centroid_fe_dist:
            Distance from the ligand centroid to heme Fe.

        ligand_atoms_within_4A_fe:
            Number of ligand heavy atoms within 4 Å of Fe.

        ligand_atoms_within_5A_fe:
            Number of ligand heavy atoms within 5 Å of Fe.

        ligand_atoms_within_6A_fe:
            Number of ligand heavy atoms within 6 Å of Fe.
    """

    fe = get_heme_iron(structure)
    ligand = get_ligand(structure)

    # Keep only ligand heavy atoms
    ligand = ligand[ligand.element != "H"]

    # --------------------------------------------------------
    # Distance from every ligand atom to Fe
    # --------------------------------------------------------

    distances = np.linalg.norm(
        ligand.coord - fe.coord,
        axis=1
    )

    # --------------------------------------------------------
    # Ligand centroid
    # --------------------------------------------------------

    ligand_centroid = np.mean(
        ligand.coord,
        axis=0
    )

    centroid_distance = np.linalg.norm(
        ligand_centroid - fe.coord
    )

    # --------------------------------------------------------
    # Count ligand atoms near Fe
    # --------------------------------------------------------

    atoms_within_4A = int(
        np.sum(distances <= 4.0)
    )

    atoms_within_5A = int(
        np.sum(distances <= 5.0)
    )

    atoms_within_6A = int(
        np.sum(distances <= 6.0)
    )

    return {
        "ligand_centroid_fe_dist": float(centroid_distance),
        "ligand_atoms_within_4A_fe": atoms_within_4A,
        "ligand_atoms_within_5A_fe": atoms_within_5A,
        "ligand_atoms_within_6A_fe": atoms_within_6A,
    }

def calculate_ligand_buriedness(structure):
    """
    Estimate how surrounded the ligand is by protein atoms.

    For each ligand heavy atom, count how many protein heavy atoms
    are within 4, 5, and 6 Angstroms.

    Returns the average number of nearby protein atoms per ligand atom.
    Larger values mean the ligand is more buried/surrounded by protein.
    """

    protein = structure[structure.chain_id == "A"]
    ligand = structure[structure.chain_id == "L"]

    # Heavy atoms only
    protein = protein[protein.element != "H"]
    ligand = ligand[ligand.element != "H"]

    # Pairwise distances:
    # rows = ligand atoms
    # columns = protein atoms
    diff = (
        ligand.coord[:, np.newaxis, :]
        - protein.coord[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        diff,
        axis=2
    )

    # Number of nearby protein atoms for each ligand atom
    neighbors_4A = np.sum(
        distances <= 4.0,
        axis=1
    )

    neighbors_5A = np.sum(
        distances <= 5.0,
        axis=1
    )

    neighbors_6A = np.sum(
        distances <= 6.0,
        axis=1
    )

    return {
        "mean_protein_neighbors_4A": float(
            np.mean(neighbors_4A)
        ),

        "mean_protein_neighbors_5A": float(
            np.mean(neighbors_5A)
        ),

        "mean_protein_neighbors_6A": float(
            np.mean(neighbors_6A)
        ),

        "max_protein_neighbors_5A": int(
            np.max(neighbors_5A)
        ),
    }

def calculate_contact_composition(structure):
    """
    Count how many contacted protein residues belong to each
    chemical residue category.

    A residue is considered contacted if at least one protein
    heavy atom is within 4.5 Angstroms of a ligand heavy atom.
    """

    residue_contacts = get_residue_contact_counts(
        structure,
        cutoff=4.5
    )

    # Residue chemistry groups
    hydrophobic = {
        "ALA", "VAL", "LEU", "ILE", "MET"
    }

    aromatic = {
        "PHE", "TYR", "TRP"
    }

    polar = {
        "SER", "THR", "ASN", "GLN"
    }

    positive = {
        "ARG", "LYS", "HIS"
    }

    negative = {
        "ASP", "GLU"
    }

    counts = {
        "hydrophobic_contact_residues": 0,
        "aromatic_contact_residues": 0,
        "polar_contact_residues": 0,
        "positive_contact_residues": 0,
        "negative_contact_residues": 0,
        "other_contact_residues": 0,
    }

    for residue_label in residue_contacts.keys():

        # Example:
        # "PHE_215" -> "PHE"
        residue_name = residue_label.split("_")[0]

        if residue_name in hydrophobic:
            counts["hydrophobic_contact_residues"] += 1

        elif residue_name in aromatic:
            counts["aromatic_contact_residues"] += 1

        elif residue_name in polar:
            counts["polar_contact_residues"] += 1

        elif residue_name in positive:
            counts["positive_contact_residues"] += 1

        elif residue_name in negative:
            counts["negative_contact_residues"] += 1
        
        else:
            counts["other_contact_residues"] += 1

    return counts

def extract_basic_features(structure):
    """
    Extract basic T6 structure-derived features.
    """

    fe = get_heme_iron(structure)
    ligand = get_ligand(structure)
    fe_n_angle = calculate_fe_n_angle(structure)

    heme_proximity = calculate_heme_proximity_features(structure)

    buriedness = calculate_ligand_buriedness(structure)

    residue_contacts = get_residue_contact_counts(structure)

    contact_composition = calculate_contact_composition(structure)

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
        "ligand_centroid_fe_dist":
            heme_proximity["ligand_centroid_fe_dist"],
        "ligand_atoms_within_4A_fe":
            heme_proximity["ligand_atoms_within_4A_fe"],
        "ligand_atoms_within_5A_fe":
            heme_proximity["ligand_atoms_within_5A_fe"],
        "ligand_atoms_within_6A_fe":
            heme_proximity["ligand_atoms_within_6A_fe"],
        "mean_protein_neighbors_4A":
            buriedness["mean_protein_neighbors_4A"],
        "mean_protein_neighbors_5A":
            buriedness["mean_protein_neighbors_5A"],
        "mean_protein_neighbors_6A":
            buriedness["mean_protein_neighbors_6A"],
        "max_protein_neighbors_5A":
            buriedness["max_protein_neighbors_5A"],
        "hydrophobic_contact_residues":
            contact_composition["hydrophobic_contact_residues"],
        "aromatic_contact_residues":
            contact_composition["aromatic_contact_residues"],
        "polar_contact_residues":
            contact_composition["polar_contact_residues"],
        "positive_contact_residues":
            contact_composition["positive_contact_residues"],
        "negative_contact_residues":
            contact_composition["negative_contact_residues"],
        "other_contact_residues":
            contact_composition["other_contact_residues"],
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

