import pandas as pd

df = pd.read_csv("data/cyp3a4_cif_qc.csv")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nStatus counts:")
print(df["status"].value_counts(dropna=False))

print("\nValid counts:")
print(df["valid"].value_counts(dropna=False))

print("\nCoordinated counts:")
print(df["is_coordinated"].value_counts(dropna=False))

print("\nFailed or missing structures:")
print(
    df[df["status"] != "OK"][
        ["inchikey_block", "status", "error"]
    ]
)

print("\nInvalid structures:")
print(
    df[df["valid"] == False][
        ["inchikey_block", "invalid_reason"]
    ]
)

print("\nExtracted feature summary:")
print(
    df[
        [
            "fe_min_dist",
            "fe_n_min_dist",
            "n_ligand_atoms",
            "n_ligand_nitrogens",
            "n_contacts",
        ]
    ].describe()
)

print("\nFe distance agreement:")

df["fe_diff"] = (
    df["fe_min_dist"]
    - df["manifest_fe_min_dist"]
).abs()

print(df["fe_diff"].describe())


print("\nContact count agreement:")

df["contact_diff"] = (
    df["n_contacts"]
    - df["manifest_n_contacts"]
).abs()

print(df["contact_diff"].describe())

#confim extraced values match t5 ref values

print("\nFe distance agreement:")

df["fe_diff"] = (
    df["fe_min_dist"]
    - df["manifest_fe_min_dist"]
).abs()

print(df["fe_diff"].describe())


print("\nContact count agreement:")

df["contact_diff"] = (
    df["n_contacts"]
    - df["manifest_n_contacts"]
).abs()

print(df["contact_diff"].describe())

#inspect extreme poses

print("\nSmallest Fe-N distances:")

print(
    df.sort_values("fe_n_min_dist")[
        [
            "inchikey_block",
            "fe_n_min_dist",
            "closest_atom_element",
            "is_coordinated",
            "valid",
        ]
    ].head(10)
)

#comapre fe_n_min_dist by coordination status
print("\nFe-N distance by coordination status:")
print(
    df.groupby("is_coordinated")["fe_n_min_dist"].describe()
)

#check closest atom element dist and cord status 
print("\nClosest atom element counts:")
print(
    df["closest_atom_element"].value_counts(dropna=False)
)

print("\nClosest atom element by coordination status:")
print(
    pd.crosstab(
        df["closest_atom_element"],
        df["is_coordinated"]
    )
)