## Prepare ADP dataset files (rename + place)

After downloading the three XTC trajectories, rename them to the filenames expected by the code and place them under `mint/data/ADP/` alongside the PDB.

### Required renames

Move/copy your downloaded files into `mint/data/ADP/` and rename as follows:

- `alanine-dipeptide-nowater.pdb` → `mint/data/ADP/alanine-dipeptide-nowater.pdb`
- `alanine-dipeptide-0-250ns-nowater.xtc` → `mint/data/ADP/alanine-dipeptide-250ns-nowater_train.xtc`
- `alanine-dipeptide-1-250ns-nowater.xtc` → `mint/data/ADP/alanine-dipeptide-250ns-nowater_valid.xtc`
- `alanine-dipeptide-2-250ns-nowater.xtc` → `mint/data/ADP/alanine-dipeptide-250ns-nowater_test.xtc`

### Example command sequence

From the folder where you downloaded the files:
```bash
mkdir -p /path/to/mint/data/ADP

cp alanine-dipeptide-nowater.pdb /path/to/mint/data/ADP/

cp alanine-dipeptide-0-250ns-nowater.xtc /path/to/mint/data/ADP/alanine-dipeptide-250ns-nowater_train.xtc
cp alanine-dipeptide-1-250ns-nowater.xtc /path/to/mint/data/ADP/alanine-dipeptide-250ns-nowater_valid.xtc
cp alanine-dipeptide-2-250ns-nowater.xtc /path/to/mint/data/ADP/alanine-dipeptide-250ns-nowater_test.xtc
```

Sanity check:
```bash
ls -1 mint/data/ADP
```
Expected:
- `alanine-dipeptide-nowater.pdb`
- `alanine-dipeptide-250ns-nowater_train.xtc`
- `alanine-dipeptide-250ns-nowater_valid.xtc`
- `alanine-dipeptide-250ns-nowater_test.xtc`
