"""
parser.py

Argument parser for running Joint Protein-Ligand Embedding (JPLE).

"""
import argparse

def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the JPLE program.


    Returns
    -------
    argparse.ArgumentParser
        Argument parser for JPLE.

    """
    parser = argparse.ArgumentParser(
        prog='jple',
        description='Run Joint Protein-Ligand Embedding (JPLE).'
    )

    # Mode
    parser.add_argument(
        '--mode',
        choices=['train', 'predict_protein', 'predict_na'],
        default='predict_protein',
        help='JPLE Execution mode. Choose "train" to train a model, '
             '"predict_protein" (default) to perform a protein query, or '
             '"predict_na" to perform a nucleic acid query.',
    )

    # Model
    parser.add_argument(
        '--param',
        dest='param_path',
        default='data/processed/param_train.npz',
        help='Path to the JPLE model parameter file in NPZ format. '
             'Model parameters will be saved here in "train" mode. '
             '(default: data/processed/param_train.npz)',
    )

    # Input
    parser.add_argument(
        '--fasta',
        dest='fasta_path',
        default=None,
        help='Path to the input query FASTA file (required in "train" and '
             '"predict_protein" mode).',
    )
    parser.add_argument(
        '--hmm',
        dest='hmm_path',
        default='data/processed/domain_rbp.hmm',
        help='Path to the profile HMM model file '
             '(default: data/processed/domain_rbp.hmm).',
    )
    parser.add_argument(
        '--zscore',
        dest='y_path',
        default=None,
        help='Path to the binding profiles (required in "train" and '
             '"predict_na" mode). The file should be tab-delimited, which each '
             'row representing a protein and each column representing a '
             'nucleic acid k-mer.',
    )

    # Output
    parser.add_argument(
        '--output',
        dest='output_path',
        default='output',
        help='Path to the output folder for JPLE results. (default: output)',
    )
    return parser
