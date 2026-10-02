import pandas as pd
from collections import Counter


df = pd.read_csv(
    "data/cyp3a4_cif_qc.csv"
)

# Keep the same usable structures as your modeling
df = df[
    (df["valid"] == True)
    & (df["is_coordinated"] == True)
    & (df["status"] == "OK")
].copy()


residue_frequency = Counter()


for residues in df["contacted_residues"].dropna():

    residue_list = residues.split(";")

    for residue in residue_list:
        residue_frequency[residue] += 1


print("\nMost commonly contacted residues:")
print("=" * 50)

for residue, count in residue_frequency.most_common(20):
    print(
        f"{residue}: {count}"
    )