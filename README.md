# JPLE: Joint Protein-Ligand Embedding
JPLE is a Python package for modeling protein-nucleic acid interactions using 
peptide k-mer and nucleic acid k-mer representations. It supports **training**,
**protein queries**, and **nucleic acid queries**.

## Installation
JPLE can be installed as a Python package via `pip`. It is recommended to use a
dedicated Python environment (e.g., Conda).

```bash
# Create a Conda environment (optional)
conda create -n jple python=3.8 -y
conda activate jple

# Install JPLE from the local directory
pip install .
```
Dependencies are installed automatically, including:
- logomaker
- matplotlib
- numpy
- pandas
- pyhmmer
- scipy

After installation, the `jple` command will be available:
```bash
jple --help
```

## Data
Raw and processed data for JPLE is under [`data`](https://github.com/morrislab/jple/blob/main/data). 
You can regenerate the processed data using the Jupyter notebook 
[`notebooks/preprocess_data.ipynb`](https://github.com/morrislab/jple/blob/main/notebooks/preprocess_data.ipynb).

### Raw data
Located in [`data/raw`](https://github.com/morrislab/jple/blob/main/data/raw):
- [`hmm_acc.txt`](https://github.com/morrislab/jple/blob/main/data/raw/hmm_acc.txt): 
Accessions of the 10 selected Pfam profile HMMs.
- [`rnacompete_metadata_eupri.tsv`](https://github.com/morrislab/jple/blob/main/data/raw/rnacompete_metadata_eupri.tsv):
Metadata for the 420 RBP constructs in EuPRI.
- [`RRMs_3D_hmm_extended.hmm`](https://github.com/morrislab/jple/blob/main/data/raw/RRMs_3D_hmm_extended.hmm):
Custom RRM_1 profile HMM.
- [`zscore_eupri.tsv`](https://github.com/morrislab/jple/blob/main/data/raw/zscore_eupri.tsv):
Binding profiles of the 420 RBP constructs in EuPRI (requires unzipping).

### Processed data
Located in [`data/processed`](https://github.com/morrislab/jple/blob/main/data/processed):
- [`zscore_train.tsv`](https://github.com/morrislab/jple/blob/main/data/processed/zscore_train.tsv):
Binding profiles of the 348 RBP constructs used for training (requires unzipping).
- [`seq_train.fasta`](https://github.com/morrislab/jple/blob/main/data/processed/seq_train.fasta):
Sequences of the 348 RBP constructs used for training.
- [`param_train.npz`](https://github.com/morrislab/jple/blob/main/data/processed/param_train.npz):
Saved JPLE model parameters (requires generation by [training](#Training)).
- [`domain_rbp.hmm`](https://github.com/morrislab/jple/blob/main/data/processed/domain_rbp.hmm):
Profile HMMs of the 10 selected domains plus the custom RRM_1 profile.

## Usage
JPLE can be run via a **Command-Line Interface (CLI)** or a Python **API**.

It supports three modes:
1. **Training**
2. **Protein query**
3. **Nucleic acid query**

Example input and output files are provided in [`test`](https://github.com/morrislab/jple/blob/main/test).

### Training
Train JPLE on protein sequences and their binding profiles.

Parameters are saved to a `.npz` file.

#### CLI
```bash
jple \
--mode train \
--param data/processed/param_train.npz \
--fasta data/processed/seq_train.fasta \
--hmm data/processed/domain_rbp.hmm \
--zscore data/processed/zscore_train.tsv
```

#### API
```python
from jple import run_jple

protein_dict_dict = run_jple.main(
    mode='train',
    param_path='data/processed/param_train.npz',
    fasta_path='data/processed/seq_train.fasta',
    hmm_path='data/processed/domain_rbp.hmm',
    y_path='data/processed/zscore_train.tsv'
)
```
The returned `protein_dict_dict` is a dictionary keyed by protein name (FASTA
header).

Each inner dictionary contains:
1. `protein_seq`: Protein sequence.
2. `domain_df`: Domains boundaries (if present).

### Protein query
Predict binding profiles for protein sequences.

#### CLI
```bash
jple \
--mode predict_protein \
--param data/processed/param_train.npz \
--fasta test/data/seq_test.fasta \
--hmm data/processed/domain_rbp.hmm \
--output test/output_protein
```

#### API
```python
from jple import run_jple

protein_dict_dict = run_jple.main(
    mode='predict_protein',
    param_path='data/processed/param_train.npz',
    fasta_path='test/data/seq_test.fasta',
    hmm_path='data/processed/domain_rbp.hmm',
)
```

#### Outputs
Results are saved to the `--output` directory (**CLI**) or returned in
`protein_dict_dict` (**API**).

| Output            | CLI file       | API key       |
|-------------------|----------------|---------------|
| Protein sequence  | –              | `protein_seq` |
| Domain boundaries | `domain.tsv`   | `domain_df`   |
| Binding profiles  | `zscore.tsv`   | `zscore_df`   |
| JPLE e-dist       | `dist.txt`     | `dist`        |
| Nearest neighbors | `neighbor.tsv` | `neighbor_df` |
| PWM               | `pwm.txt`      | `pwm`         |
| IUPAC motif       | `iupac.txt`    | `iupac`       |
| Sequence logo     | `logo.png`     | –             |

> **Notes:**
> - Domain boundaries are only reported if the protein contains selected domains.
> - Everything downstream are produced only if the protein contains an RRM or KH domain.

### Nucleic acid query
Predict residue importance profiles given binding profiles.

#### CLI
```bash
jple \
--mode predict_na \
--param data/processed/param_train.npz \
--zscore test/data/zscore_test.tsv \
--output test/output_na
```

#### API
```python
from jple import run_jple

protein_dict_dict = run_jple.main(
    mode='predict_na',
    param_path='data/processed/param_train.npz',
    y_path='test/data/zscore_test.tsv'
)
```

#### Outputs
Results are saved to the `--output` directory (**CLI**) or returned in
`protein_dict_dict` (**API**).

| Output             | CLI file        | API key         |
|--------------------|-----------------|-----------------|
| Residue importance | `importance.tsv`| `importance_df` |
| JPLE e-dist        | `dist.txt`      | `dist`          |

## License
This project is licensed under the BSD 3-Clause License. See
[LICENSE](https://github.com/morrislab/jple/blob/main/LICENSE) for details.

## References
If you use JPLE in publications, please cite:

Sasse, A., Ray, D., Laverty, K.U. et al.
**A resource of RNA-binding protein motifs across eukaryotes reveals
evolutionary dynamics and gene-regulatory function.**

*Nat Biotechnol* (2025). https://doi.org/10.1038/s41587-025-02733-6
