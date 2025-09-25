"""
io_utils.py

Utilities for reading FASTA files and saving JPLE results (domain annotations,
PWM, motifs, etc.).

"""
import os
from typing import Dict

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_fasta(fasta_path: str) -> Dict[str, Dict]:
    """
    Read sequences from a FASTA File.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.

    Returns
    -------
    protein_dict_dict : Dict[str, Dict]
        Dictionary of protein dictionaries. Each dictionary is indexed by
        its protein ID and contains:

        (1) ``protein_seq`` (str): Protein sequence.

    """

    # Open file
    with open(fasta_path) as f:
        line_list = f.read().splitlines()

    # Read lines
    current_id = ''
    current_seq = ''
    protein_dict_dict = {}
    for line in line_list:

        # Line is empty
        if not line:
            continue

        # Header line
        if line.startswith('>'):
            if current_seq:
                protein_dict_dict[current_id] = {
                    'protein_seq': current_seq
                }
                current_seq = ''
            current_id = line[1:].strip()
            if current_id in protein_dict_dict:
                raise ValueError('FASTA headers must be unique. ',
                                 f'Duplicate found: {current_id}')

        # Sequence line
        else:
            current_seq += line.strip()

    # Add final record
    if current_seq:
        protein_dict_dict[current_id] = {
            'protein_seq': current_seq
        }
    return protein_dict_dict


def save_pwm(pwm_path: str, protein_id: str, pwm: np.ndarray) -> None:
    """
    Save a position weight matrix (PWM) in MEME format.

    Parameters
    ----------
    pwm_path : str
        Path to the output MEME file.
    protein_id : str
        Protein ID.
    pwm : np.ndarray
        PWM matrix of shape (L, 4), where L is the motif length and columns
        correspond to A, C, G, and U.

    """

    # Write the header
    line_list = [
        'MEME version 5.5.8\n\n',
        'ALPHABET= ACGU\n\n',
        'Background letter frequencies (from uniform background):\n',
        'A 0.25000 C 0.25000 G 0.25000 U 0.25000 \n\n',
        f'MOTIF {protein_id}\n\n',
        f'letter-probability matrix: alength= 4 w= {len(pwm)} nsites= 1 E= 0\n'
    ]

    # Write the PWM
    for pwm_list in pwm:
        line_list.append('  ' +
                         '\t'.join(f'{val:.6f}' for val in pwm_list) + '\n')

    # Save the file
    with open(pwm_path, 'w') as f:
        f.writelines(line_list)


def save_logo(logo_path: str, pwm: np.ndarray) -> None:
    """
    Plot and save a sequence logo for a PWM.

    Parameters
    ----------
    logo_path : str
        Path to the output motif logo.
    pwm : np.ndarray
        PWM matrix of shape (L, 4), where L is the motif length and columns
        correspond to A, C, G, and T.

    """

    # Define the color scheme
    color_scheme = {
        'A': '#00CC00', # Green
        'C': '#0000CC', # Blue
        'G': '#FFB302', # Orange
        'U': '#CC0001', # Red
    }

    # Convert per-position information content (in bits)
    bit_list = np.clip(2 + np.sum(pwm * np.log2(pwm), axis=1), 0, None)
    logo_mat = pwm * bit_list[:, None]

    # Plot the logo
    logomaker.Logo(pd.DataFrame(logo_mat, columns=['A', 'C', 'G', 'U']),
                   color_scheme=color_scheme,
                   show_spines=False)
    plt.xticks([])
    plt.yticks([])

    # Save the logo
    plt.savefig(logo_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(mode: str, output: str,
                 protein_dict_dict: Dict[str, Dict]) -> None:
    """
    Save JPLE results to the specified output folder.

    For each protein with domain annotations, a subfolder is created containing:

    (1) ``domain.tsv``: Domain boundaries.
    (2) ``zscore.tsv``: Predicted binding profiles.
    (3) ``dist.txt``: JPLE e-dist.
    (4) ``neighbor.tsv``: Neighboring training-set proteins and their
        contributions.
    (5) ``pwm.txt``: Predicted PWM in MEME format.
    (6) ``iupac.txt``: Predicted IUPAC motif.
    (7) ``logo.png``: Motif logo generated from the PWM.
    (8) ``importance.tsv``: Predicted residue importance profiles.

    Parameters
    ----------
    mode : str
        JPLE Execution mode.
    output : str
        Path to the output directory.
    protein_dict_dict : Dict[str, Dict]
        Dictionary of protein dictionaries with results.

    """
    # Create the output folder
    os.makedirs(output, exist_ok=True)

    # Iterate over the proteins
    for protein_id, protein_dict in protein_dict_dict.items():

        # Skip if the not performing NA query and protein has no domains
        if mode != 'predict_na' and 'domain_df' not in protein_dict:
            continue

        # Create the output protein folder
        protein_folder = os.path.join(output, protein_id)
        os.makedirs(protein_folder, exist_ok=True)

        # Save the domains
        if 'domain_df' in protein_dict:
            domain_path = os.path.join(protein_folder, 'domain.tsv')
            protein_dict['domain_df'].to_csv(domain_path, index=False, sep='\t')

        # Save the predicted binding profiles
        if 'zscore_df' in protein_dict:
            zscore_path = os.path.join(protein_folder, 'zscore.tsv')
            protein_dict['zscore_df'].to_csv(zscore_path, sep='\t')

        # Save the e-dist
        if 'dist' in protein_dict:
            dist_path = os.path.join(protein_folder, 'dist.txt')
            np.savetxt(dist_path, [protein_dict['dist']])

        # Save the neighbors
        if 'neighbor_df' in protein_dict:
            neighbor_path = os.path.join(protein_folder, 'neighbor.tsv')
            protein_dict['neighbor_df'].to_csv(neighbor_path, index=False,
                                               sep='\t')

        # Save the PWM
        if 'pwm' in protein_dict:
           pwm_path = os.path.join(protein_folder, 'pwm.txt')
           save_pwm(pwm_path, protein_id, protein_dict['pwm'])

        # Save the IUPAC motif
        if 'iupac' in protein_dict:
           iupac_path = os.path.join(protein_folder, 'iupac.txt')
           with open(iupac_path, 'w') as f:
               f.write(protein_dict['iupac'] + '\n')

        # Generate and save the logo
        if 'pwm' in protein_dict:
           logo_path = os.path.join(protein_folder, 'logo.png')
           save_logo(logo_path, protein_dict['pwm'])

        # Save the predicted residue importance profile
        if 'importance_df' in protein_dict:
            importance_path = os.path.join(protein_folder, 'importance.tsv')
            protein_dict['importance_df'].to_csv(importance_path, sep='\t')
