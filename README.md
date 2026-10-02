
Current structure-derived features include:

- Fe-to-closest-ligand-N distance
- Fe-N approach angle relative to the heme plane
- total protein-ligand heavy-atom contacts
- number of contacted protein residues
- identities of contacted residues

These features are tested against:

`CYP3A4_pIC50_direct_inhibition`

using decision-tree regression.


Expected local data layout:

```text
data/
├── compounds.csv
├── t5_manifest.csv
└── poses/
    └── CYP3A4/
        ├── *.cif



What to run:


pip install -r requirements.txt

feature extraction:
python src/inspect_all_cifs.py

analyze residue contacts:
python src/analyze_residue_contacts.py

run decsion tree experiments:
python src/test_structure_tree.py


Measureables: 

From the structure dataset / feature extraction
- total number of CYP3A4 CIF files available
- total number of CIF files successfully processed
- number of rows in data/cyp3a4_cif_qc.csv
- number of usable structures after filtering
- number of failed or errored structures
- any error messages encountered

From analyze_residue_contacts.py
- the most commonly contacted residues and their counts

From test_structure_tree.py
Please send the full performance table containing:
- mean RMSE
- standard deviation RMSE
- mean MAE
- standard deviation MAE
- mean R²
- standard deviation R²







