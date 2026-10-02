from pathlib import Path

from inspect_cif import load_structure


# ============================================================
# Configuration
# ============================================================

CIF_PATH = Path(
    "data/poses/CYP3A4/ZDVHZKHFMHUZDU__rt5s0.cif"
)

# Residues we want to verify from the contact analysis
CHECK_RESIDUES = [
    105,
    108,
    119,
    120,
    212,
    213,
    215,
    241,
    301,
    304,
    305,
    309,
    369,
    370,
    372,
    374,
    442,
    482,
]


# ============================================================
# Load structure
# ============================================================

structure = load_structure(CIF_PATH)

# Keep only protein chain A
protein = structure[
    structure.chain_id == "A"
]


# ============================================================
# Print basic protein numbering range
# ============================================================

unique_residue_ids = sorted(
    set(int(res_id) for res_id in protein.res_id)
)

print("=" * 60)
print("PROTEIN RESIDUE NUMBERING CHECK")
print("=" * 60)

print("CIF file:", CIF_PATH.name)

print(
    "Residue number range:",
    unique_residue_ids[0],
    "to",
    unique_residue_ids[-1]
)

print(
    "Number of unique protein residues:",
    len(unique_residue_ids)
)


# ============================================================
# Check selected residues
# ============================================================

print("\n" + "=" * 60)
print("SELECTED RESIDUES")
print("=" * 60)

for res_id in CHECK_RESIDUES:

    residue = protein[
        protein.res_id == res_id
    ]

    if residue.array_length() == 0:

        print(
            f"{res_id}: NOT FOUND"
        )

    else:

        residue_names = sorted(
            set(str(name) for name in residue.res_name)
        )

        print(
            f"{res_id}: {', '.join(residue_names)}"
        )


# ============================================================
# Special check for heme-binding cysteine
# ============================================================

print("\n" + "=" * 60)
print("HEME-BINDING CYSTEINE CHECK")
print("=" * 60)

residue_442 = protein[
    protein.res_id == 442
]

if residue_442.array_length() == 0:

    print("Residue 442 was not found.")

else:

    residue_names = sorted(
        set(str(name) for name in residue_442.res_name)
    )

    print(
        "Residue 442 name:",
        residue_names
    )

    if "CYS" in residue_names:

        print(
            "PASS: residue 442 is CYS."
        )

    else:

        print(
            "WARNING: residue 442 is not CYS."
        )


# ============================================================
# Print selected residue labels
# ============================================================

print("\n" + "=" * 60)
print("CONTACT RESIDUE LABELS")
print("=" * 60)

for res_id in CHECK_RESIDUES:

    residue = protein[
        protein.res_id == res_id
    ]

    if residue.array_length() > 0:

        residue_name = str(
            residue.res_name[0]
        )

        print(
            f"{residue_name}_{res_id}"
        )