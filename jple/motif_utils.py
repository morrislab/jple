"""
motif_utils.py

Utilities for aligning k-mers, computing PWMs, and deriving IUPAC motifs.

"""
from typing import List, Tuple

import numpy as np
import pandas as pd


def align(kmer_1: str, kmer_2: str) -> Tuple[int, int]:
    """
    Align two k-mers.

    The two k-mers are shifted left and right against one another to find the
    optimal offset to maximize matching nucleotides.

    Parameters
    ----------
    kmer_1 : str
        First k-mer.
    kmer_2 : str
        Second k-mer.

    Returns
    -------
    max_match : int
        Maximum number of matching nucleotides.
    min_offset : int
        Offset required for maximum matching.

    """
    max_match = 0
    min_offset = 0
    k = len(kmer_1)

    # Shift kmer 1 left
    for i in range(k):
        match = sum(a == b for a, b in zip(kmer_1[i:], kmer_2))
        if match > max_match:
            max_match = match
            min_offset = i

    # Shift kmer 2 left
    for i in range(k):
        match = sum(a == b for a, b in zip(kmer_2[i:], kmer_1))
        if match > max_match:
            max_match = match
            min_offset = -i
    return max_match, min_offset


def align_top_kmer(kmer_list: List[str]) -> List[str]:
    """
    Perform pairwise alignment for all k-mers against the seed k-mer

    The seed k-mer is selected as the k-mer with the most average matches
    against the other k-mers, with the least average offset.

    Parameters
    ----------
    kmer_list : List[str]
        List of the k-mers.

    Returns
    -------
    aligned_kmer_list : List[str]
        List of the aligned k-mers.

    """

    # Find the seed k-mer
    match_sum_list = []
    offset_sum_list = []

    # Iterate over the k-mers
    for kmer_1 in kmer_list:
        match_sum = 0
        offset_sum = 0

        # Iterate over the k-mers
        for kmer_2 in kmer_list:

            # Perform alignment and get total sum and offset
            match, offset = align(kmer_1, kmer_2)
            match_sum += match
            offset_sum += abs(offset)
        match_sum_list.append(match_sum)
        offset_sum_list.append(offset_sum)
    match_sum_list = np.array(match_sum_list)
    offset_sum_list = np.array(offset_sum_list)

    # Select the seed k-mer
    best_idx_list = np.argwhere(
        match_sum_list == match_sum_list.max()).flatten()
    best_idx = best_idx_list[np.argmin(offset_sum_list[best_idx_list])]

    # Perform alignment using the seed k-mer
    offset_list = []
    for kmer in kmer_list:
        _, offset = align(kmer_list[best_idx], kmer)
        offset_list.append(offset)

    # Format the alignments
    aligned_kmer_list = []
    min_offset = min(offset_list)
    max_offset = max(offset_list)
    for kmer, offset in zip(kmer_list, offset_list):
        aligned_kmer_list.append(
            '-' * (offset - min_offset) + kmer + '-' * (max_offset - offset)
        )
    return aligned_kmer_list


def get_pwm(kmer_list: List[str], zscore_list: List[float]) -> np.ndarray:
    """
    Compute and save the PWM.

    The PWMs only contain positions where no less than half of the k-mers are
    represented.

    Parameters
    ----------
    kmer_list : List[str]
        List of the top 10 aligned k-mers.
    zscore_list : List[float]
        List of the Z-scores of the top 10 k-mers.

    Returns
    -------
    pwm : np.ndarray
        PWM.

    """

    # Initialize the variables
    mean_zscore = np.mean(zscore_list)
    num_pos = len(kmer_list[0])
    pfm = np.zeros((num_pos, 4))
    pwm = np.zeros((num_pos, 4))

    # Iterate over the positions
    for pos in range(num_pos):

        # Iterate over the k-mers
        for kmer_idx, kmer in enumerate(kmer_list):

            # Gap
            if kmer[pos] == '-':
                pwm[pos] += mean_zscore / 4
                continue

            # Iterate over the nucleotides
            for nt_idx, nt in enumerate(['A', 'C', 'G', 'U']):
                if kmer[pos] == nt:
                    pfm[pos, nt_idx] += 1
                    pwm[pos, nt_idx] += zscore_list[kmer_idx]

    # Trim the PWM
    coverage = pfm.sum(axis=1) / len(kmer_list)
    start = np.argmax(coverage >= 0.5)
    end = len(coverage) - np.argmax(coverage[::-1] >= 0.5)
    pwm = pwm[start:end]

    # Add pseudo-count and convert to fraction
    pwm += 1
    pwm /= pwm.sum(axis=1)[:, None]
    return pwm


def compute_pwm(zscore_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the PWM by aligning the top k-mers by Z-scores.

    Parameters
    ----------
    zscore_df : pd.DataFrame
        Predicted k-mer Z-scores.

    Returns
    -------
    pwm : np.ndarray
        Predicted PWM.

    """

    # Get the top 10 k-mers
    sorted_df = zscore_df.sort_values('zscore', ascending=False)[:10]
    top_kmer_list = sorted_df.index.to_list()
    top_zscore_list = sorted_df['zscore'].to_list()

    # Align the top 10 k-mers
    aligned_top_kmer_list = align_top_kmer(top_kmer_list)

    # Get the PWM
    pwm = get_pwm(aligned_top_kmer_list, top_zscore_list)
    return pwm

def compute_iupac(pwm: np.ndarray) -> str:
    """
    Compute the closest IUPAC motif from a PWM.

    Each column is compared to IUPAC nucleotide fractions, and the nearest
    symbol (Euclidean distance) is chosen. Leading/trailing 'N's are stripped.

    Parameters
    ----------
    pwm : np.ndarray
        PWM matrix of shape (L, 4), where L is the motif length and columns
        correspond to A, C, G, and U.

    Returns
    -------
    iupac : str
        iupac motif.

    """

    # Define the IUPAC letters
    iupac_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1],
        'R': [0.5, 0, 0.5, 0],
        'Y': [0, 0.5, 0, 0.5],
        'S': [0, 0.5, 0.5, 0],
        'W': [0.5, 0, 0, 0.5],
        'K': [0, 0, 0.5, 0.5],
        'M': [0.5, 0.5, 0, 0],
        'B': [0, 1 / 3, 1 / 3, 1 / 3],
        'D': [1 / 3, 0, 1 / 3, 1 / 3],
        'H': [1 / 3, 1 / 3, 0, 1 / 3],
        'V': [1 / 3, 1 / 3, 1 / 3, 0],
        'N': [0.25, 0.25, 0.25, 0.25],
    }
    letter_list = np.array(list(iupac_dict.keys()))
    frac_mat = np.array(list(iupac_dict.values()))

    # Compute the IUPAC motif
    iupac = ''
    for pwm_row in pwm:
        diff_list = np.linalg.norm(frac_mat - pwm_row, axis=1)
        iupac += letter_list[diff_list.argmin()]
    iupac = iupac.strip('N')
    return iupac
