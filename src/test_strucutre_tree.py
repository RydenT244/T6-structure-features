from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Configuration
# ============================================================

STRUCTURE_PATH = Path("data/cyp3a4_cif_qc.csv")
COMPOUNDS_PATH = Path("data/compounds.csv")

TARGET_COLUMN = "CYP3A4_pIC50_direct_inhibition"


# ============================================================
# Feature sets
# ============================================================

SELECTED_RESIDUES = [
    "PHE_215",
    "CYS_442",
    "ARG_212",
    "ILE_120",
    "PHE_213",
    "ILE_369",
    "PHE_241",
    "ARG_372",
]

# Geometry only
GEOMETRY_FEATURES = [
    "fe_n_min_dist",
    "fe_n_angle",
]

# Core T6 feature set
CORE_FEATURES = [
    "fe_n_min_dist",
    "fe_n_angle",
    "n_contacts",
]

# Core plus contact residues
CORE_CONTACT_RESIDUES = [
        "fe_n_min_dist",
        "fe_n_angle",
        "n_contacts",
        "n_contact_residues",
    ]

RESIDUE_FEATURES = [
    f"contact_{residue}"
    for residue in SELECTED_RESIDUES
]

CORE_PLUS_RESIDUES_ID = (
    CORE_FEATURES + RESIDUE_FEATURES
)

HEME_PROXIMITY_FEATURES = [
    "ligand_centroid_fe_dist",
    "ligand_atoms_within_4A_fe",
    "ligand_atoms_within_5A_fe",
    "ligand_atoms_within_6A_fe",
]

CORE_HEME_PROXIMITY = (
    CORE_FEATURES
    + HEME_PROXIMITY_FEATURES
)

BURIEDNESS_FEATURES = [
    "mean_protein_neighbors_4A",
    "mean_protein_neighbors_5A",
    "mean_protein_neighbors_6A",
    "max_protein_neighbors_5A",
]

CORE_PLUS_BURIEDNESS = (
    CORE_FEATURES
    + BURIEDNESS_FEATURES
)

CONTACT_COMPOSITION_FEATURES = [
    "hydrophobic_contact_residues",
    "aromatic_contact_residues",
    "polar_contact_residues",
    "positive_contact_residues",
    "negative_contact_residues",
    "other_contact_residues",
]

CORE_PLUS_CONTACT_COMPOSITION = (
    CORE_FEATURES
    + CONTACT_COMPOSITION_FEATURES
)

feature_sets = {
    "geometry_only": GEOMETRY_FEATURES,
    "core_structure": CORE_FEATURES,
    "core_plus_contact_residues": CORE_CONTACT_RESIDUES,
    "core_plus_residue_identity": CORE_PLUS_RESIDUES_ID,
    "core_plus_heme_proximity": CORE_HEME_PROXIMITY,
    "core_plus_buriedness": CORE_PLUS_BURIEDNESS,
    "core_plus_contact_composition": CORE_PLUS_CONTACT_COMPOSITION,
}


# ============================================================
# Load data
# ============================================================

structures = pd.read_csv(STRUCTURE_PATH)
compounds = pd.read_csv(COMPOUNDS_PATH)


# Keep usable structures
structures = structures[
    (structures["valid"] == True)
    & (structures["is_coordinated"] == True)
    & (structures["status"] == "OK")
].copy()


# Merge with experimental inhibition values
df = structures.merge(
    compounds[
        [
            "inchikey_block",
            TARGET_COLUMN,
        ]
    ],
    on="inchikey_block",
    how="inner"
)


# Keep rows with all features needed for comparison
all_features = [
    "fe_n_min_dist",
    "fe_n_angle",
    "n_contacts",
    "n_contact_residues",
    "ligand_centroid_fe_dist",
    "ligand_atoms_within_4A_fe",
    "ligand_atoms_within_5A_fe",
    "ligand_atoms_within_6A_fe",
    "mean_protein_neighbors_4A",
    "mean_protein_neighbors_5A",
    "mean_protein_neighbors_6A",
    "max_protein_neighbors_5A",
]


for residue in SELECTED_RESIDUES:

    column_name = f"contact_{residue}"

    df[column_name] = (
        df["contacted_residues"]
        .fillna("")
        .apply(
            lambda x: 1 if residue in x.split(";") else 0
        )
    )

df = df.dropna(
    subset=all_features + [TARGET_COLUMN]
).copy()

print("Rows used:", len(df))


# ============================================================
# Multiple random seeds
# ============================================================

# More seeds gives a more stable estimate.
# 20 is a reasonable starting point for this small dataset.

SEEDS = range(20)

results = []


for model_name, features in feature_sets.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    X = df[features]
    y = df[TARGET_COLUMN]

    for seed in SEEDS:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed
        )

        model = DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=5,
            random_state=seed
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "model": model_name,
            "seed": seed,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        })


# ============================================================
# Summarize performance across seeds
# ============================================================

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("model")
    .agg(
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),

        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std"),

        mean_R2=("R2", "mean"),
        std_R2=("R2", "std"),
    )
    .reset_index()
)


print("\n" + "=" * 70)
print("AVERAGE PERFORMANCE ACROSS RANDOM SEEDS")
print("=" * 70)

print(
    summary.to_string(index=False)
)