"""
rep_utils.py

Utilities for extracting protein regions and generating peptide k-mer
representations.

"""
from typing import Dict, List, Tuple

import numpy as np


def get_region(protein_dict_dict: Dict[str, Dict], domain_list: List[str],
               flank: int=15) -> Tuple[List[str], List[List[str]]]:
    """
    Extract protein regions corresponding to selected domains.

    Each domain is extended in both directions by `flank` amino acids, and
    overlapping regions are merged. Only proteins with at least one selected
    domain are returned.

    Parameters
    ----------
    protein_dict_dict: Dict[str, Dict]
        Dictionary of protein dictionaries with results.
    domain_list : List[str]
        List of selected domains.
    flank : int, optional
        Number of amino acids to extend on each side of a domain (default 15).

    Returns
    -------
    jple_protein_id_list : List[str]
        IDs of the proteins with at least one selected domain.
    region_seq_list_list : List[List[str]]
        List of lists of region sequences for each protein.

    """

    # Iterate over the proteins
    jple_protein_id_list = []
    region_seq_list_list = []
    for protein_id, protein_dict in protein_dict_dict.items():

        # Skip if the protein has no domains
        if 'domain_df' not in protein_dict:
            continue

        # Filter for the domains of interest
        protein_seq = protein_dict['protein_seq']
        domain_df = protein_dict['domain_df']
        filtered_domain_df = domain_df[domain_df['domain'].isin(domain_list)]

        # Skip if the protein has no selected domains
        if len(filtered_domain_df) == 0:
            continue

        # Extend the domains by 15 AAs in both directions
        domain_mat = filtered_domain_df[['from', 'to']].to_numpy()
        domain_mat = domain_mat[np.argsort(domain_mat[:, 0])]
        domain_mat[:, 0] = domain_mat[:, 0] - 1 - flank
        domain_mat[:, 1] = domain_mat[:, 1] - 1 + flank
        domain_mat = domain_mat.clip(0, len(protein_seq) - 1)

        # Iterate over the extended domains and merge them into regions
        region_seq_list = []
        current_start, current_end = domain_mat[0]
        for start, end in domain_mat[1:]:
            if start <= current_end + 1:
                current_end = end
            else:
                region_seq_list.append(
                    protein_seq[current_start:current_end + 1])
                current_start, current_end = start, end
        region_seq_list.append(protein_seq[current_start:current_end + 1])

        # Save the protein IDs and region sequences
        jple_protein_id_list.append(protein_id)
        region_seq_list_list.append(region_seq_list)
    return jple_protein_id_list, region_seq_list_list


def generate_rep(region_seq_list_list: List[List[str]],
                 x_kmer_list: List[str]=None, k: int=5, g: int=1) -> \
        Tuple[np.ndarray, List[str]]:
    """
    Generate  a peptide k-mer count matrix for a set of protein regions.

    K-mers may include gaps ('X') of length `g`. If `x_kmer_list` is not
    provided, all observed k-mers are used.

    Parameters
    ----------
    region_seq_list_list : List[List[str]]
        List of lists of region sequences for each protein.
    x_kmer_list : List[str], optional
        Predefined list of k-mers to use. If None, all observed k-mers will be
        included.
    k : int, optional
        Length of peptide k-mers (default 5).
    g : int, optional
        Length of gap within peptide k-mers (default 1).

    Returns
    -------
    rep_mat : np.ndarray
        Peptide K-mer count matrix of shape (n_proteins, n_kmers). Rows
        correspond to proteins, columns correspond to k-mers.
    x_kmer_list : List[str]
        List of peptide k-mers corresponding to the columns of `rep_mat`.

    """

    def extract_kmers(seq: str) -> List[str]:
        """
        Generate k-mers (with optional gaps) from a sequence.

        Parameters
        ----------
        seq : str
            Sequence to extract k-mers from.

        Returns
        -------
        kmer_all_list : List[str]
            List of all k-mers.

        """
        kmer_all_list = []
        gap = 'X' * g

        # Iterate over the k-mer starting indices
        for kmer_start_idx in range(len(seq) - k + 1):
            kmer = seq[kmer_start_idx:kmer_start_idx + k]

            # If there is a gap
            if g > 0:

                # Iterate over the gap positions
                for gap_idx in range(1, k - g):
                    kmer_all_list.append(
                        kmer[:gap_idx] + gap + kmer[gap_idx + g:])

            # If there is no gap
            else:
                kmer_all_list.append(kmer)
        return kmer_all_list

    # Generate the k-mer list if not provided
    if x_kmer_list is None:
        kmer_set = set()
        for region_seq_list in region_seq_list_list:
            for region_seq in region_seq_list:
                kmer_set.update(extract_kmers(region_seq))
        x_kmer_list = sorted(kmer_set)

    # Map k-mers to indices
    kmer_dict = {kmer: idx for idx, kmer in enumerate(x_kmer_list)}

    # Initialize the matrix
    rep_mat = np.zeros((len(region_seq_list_list), len(x_kmer_list)), dtype=int)

    # Iterate over the proteins
    for protein_idx, region_seq_list in enumerate(region_seq_list_list):

        # Iterate over the regions
        for region_seq in region_seq_list:

            # Iterate over the k-mers
            for kmer in extract_kmers(region_seq):
                if kmer in kmer_dict:
                    rep_mat[protein_idx, kmer_dict[kmer]] += 1
    return rep_mat, x_kmer_list
