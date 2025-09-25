"""
domain_utils.py

Utilities for scanning protein sequences against profile HMMs using pyhmmer.

This module provides functions to scan protein sequences for Pfam domains,
resolve nested and overlapping domains, and rescan trimmed domains to confirm
their presence.

Main function
-------------
scan_hmm : Scan a list of protein sequences and return a dictionary of resolved
           domain assignment per protein.

"""
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pyhmmer


def _scan_hmm_pyhmmer(protein_id_list: List[str], seq_list: List[str],
                     hmm_path: str, E: float=0.01, domE: float=0.01) \
        -> pd.DataFrame:
    """
    Scan protein sequences against profile HMMs using pyhmmer.hmmscan.

    Parameters
    ----------
    protein_id_list : List[str]
        List of protein IDs.
    seq_list : List[str]
        List of protein sequences.
    hmm_path : str
        Path to the profile HMM file.
    E : float, optional
        Sequence-level E-value threshold. Default is 0.01.
    domE : float, optional
        Domain-level E-value threshold. Default is 0.01.

    Returns
    -------
    scan_df : pd.DataFrame
        Dataframe with one row per reported hit, containing:
        (1) ``protein_id`` (str): Protein ID.
        (2) ``domain`` (str): Name of the profile HMM.
        (3) ``env_from`` (int): Start index of the domain (1-based, inclusive).
        (4) ``env_to`` (int): End index of the domain (1-based, inclusive).
        (5) ``domain_evalue`` (float): conditional E-value of the domain.

    """

    # Format the sequences
    hmmer_seq_list = []
    for protein_id, seq in zip(protein_id_list, seq_list):
        tmp = pyhmmer.easel.TextSequence(name=protein_id.encode(), sequence=seq)
        hmmer_seq_list.append(tmp.digitize(pyhmmer.easel.Alphabet.amino()))

    # Read the HMMs
    with pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
        hmm_list = list(hmm_file)

    # Scan the sequences
    hits_list = list(
        pyhmmer.hmmscan(hmmer_seq_list, hmm_list, E=E, domE=domE))

    # Parse the hmmscan output
    scan_list_list = []
    for hits in hits_list:
        for hit in hits:
            for domain in hit.domains:
                if domain.reported:
                    domain_name = domain.alignment.hmm_name.decode('utf-8')
                    scan_list = [domain.alignment.target_name.decode('utf-8'),
                                 domain_name,
                                 domain.env_from, domain.env_to,
                                 domain.c_evalue]
                    scan_list_list.append(scan_list)
    scan_df = pd.DataFrame(scan_list_list,
                           columns=['protein_id', 'domain', 'env_from',
                                    'env_to',
                                    'domain_evalue']).sort_values(
        ['protein_id', 'env_from'])
    return scan_df


def _resolve_nested_domains(scan_protein_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve for nested domains.

    If domain j is fully nested within domain i:
        - Remove domain j if its E-value >= domain i
        - Otherwise, remove domain i

    Parameters
    ----------
    scan_protein_df : pd.DataFrame
        Dataframe with one row per domain hit.

    Returns
    -------
    scan_protein_df : pd.DataFrame
        Filtered Dataframe with non-nested domains, indexed by domain name.

    """

    # Iterate over all pairs of domains
    nonnested_idx_list = np.arange(len(scan_protein_df))
    for i in range(len(scan_protein_df)):
        for j in range(i + 1, len(scan_protein_df)):

            # If domain j is fully nested within domain i
            if scan_protein_df.iloc[j]['env_to'] <= scan_protein_df.iloc[i][
                'env_to']:

                # Remove domain j if its E-value is higher than or equal to that
                # of domain i
                if scan_protein_df.iloc[j]['domain_evalue'] >= \
                        scan_protein_df.iloc[i]['domain_evalue']:
                    nonnested_idx_list = nonnested_idx_list[
                        nonnested_idx_list != j]

                # Remove domain i if its E-value is higher
                else:
                    nonnested_idx_list = nonnested_idx_list[
                        nonnested_idx_list != i]

    # Filter for the retained domains
    scan_protein_df = scan_protein_df.iloc[nonnested_idx_list].sort_values(
        'domain_evalue')
    scan_protein_df['domain'] = [f'{e}.{i}' for i, e in
                                 enumerate(scan_protein_df['domain'])]
    scan_protein_df = scan_protein_df.set_index('domain')
    return scan_protein_df


def _assign_overlap(scan_protein_df: pd.DataFrame) -> List[str]:
    """
    Assign each residue to the domain with the lowest E-value in overlapping
    regions.

    Parameters
    ----------
    scan_protein_df : pd.DataFrame
        Dataframe indexed by domain name.

    Returns
    -------
    label_list : List[str]
        List of per-residue domain assignments.

    """
    label_list = [''] * (scan_protein_df['env_to'].max() + 1)
    for domain, row in scan_protein_df.iterrows():
        for idx in range(row['env_from'], row['env_to'] + 1):
            if not label_list[idx]:
                label_list[idx] = domain
    return label_list


def _per_residue_to_ranges(label_list: List[str]) -> \
        Tuple[List[str], List[int], List[int]]:
    """
    Convert per-residue labels into domain ranges.

    Parameters
    ----------
    label_list : List[str]
        List of per-residue domain assignments.

    Returns
    -------
    domain_list : List[str]
        Domain names for each range.
    start_list : List[int]
        Start indices of the domain ranges (1-based).
    end_list : List[int]
        End indices of the domain ranges(1-based, inclusive).

    """
    domain_list = []
    start_list = []
    end_list = []

    cur_start = None
    cur_label = None

    # Iterate over the residues
    for idx, label in enumerate(label_list):

        # Residue is part of a domain
        if label:
            if cur_start is None:
                cur_start = idx
                cur_label = label
            elif label != cur_label:
                domain_list.append(cur_label)
                start_list.append(cur_start)
                end_list.append(idx - 1)
                cur_start = idx
                cur_label = label

        # Residue is not part of a domain
        else:
            if cur_start is not None:
                domain_list.append(cur_label)
                start_list.append(cur_start)
                end_list.append(idx - 1)
                cur_start = None
                cur_label = None

    # Save the final domain
    if cur_start is not None:
        domain_list.append(cur_label)
        start_list.append(cur_start)
        end_list.append(len(label_list) - 1)
    return domain_list, start_list, end_list


def _rescan_trimmed_domains(domain_df: pd.DataFrame, seq: str,
                            hmm_path: str) -> pd.DataFrame:
    """
    Rescan domains that were trimmed to confirm presence.

    Parameters
    ----------
    domain_df : pd.DataFrame
        The DataFrame, with each row indexed by the domain name, contains the
        following columns:

        (1) ``env_from_before`` (int): Start index of the domain before trimming
            (1-based).
        (2) ``env_to_before`` (int): End index of the domain before trimming
            (1-based, inclusive).
        (3) ``env_from_after`` (int): Start index of the domain after trimming
            (1-based).
        (4) ``env_to_after`` (int): End index of the domain after trimming
            (1-based, inclusive).
        (5) ``trimmed`` (bool): Whether the domain has been trimmed.
    seq : str
        Protein sequence.
    hmm_path : str
        Path to the profile HMM file.

    Returns
    -------
    domain_df: pd.DataFrame
        Filtered DataFrame with domains confirmed present.

    """

    # Iterate over the domains
    flag_list = []
    for domain_id, row in domain_df.iterrows():

        # The domain is trimmed
        if row['trimmed']:

            # Rescan the trimmed sequence
            trimmed_seq = seq[row['env_from_after'] - 1:row['env_to_after']]
            trimmed_scan_df = _scan_hmm_pyhmmer([domain_id], [trimmed_seq],
                                               hmm_path)

            # Check whether the domain is retained
            flag_list.append(str(row.name).split('.')[0] in trimmed_scan_df[
                'domain'].to_list())

        # The domain is not trimmed
        else:
            flag_list.append(True)
    domain_df = domain_df[flag_list]
    return domain_df


def scan_hmm(protein_id_list: List[str], seq_list: List[str], hmm_path: str) ->\
        Dict[str, pd.DataFrame]:
    """
    Scan protein sequences for Pfam domains and resolve overlapping domains.

    Each protein is scanned against HMM profiles. Overlapping domains are
    resolved as follows:

    (1) Nested domains: the nested domain is removed if its E-value is
        higher or equal; otherwise, the larger domain is removed.
    (2) Partially overlapping domains: each residue is assigned to the domain
        with the lower E-value.
    (3) Domains that are trimmed due to overlap are rescanned to confirm their
        presence.

    Parameters
    ----------
    protein_id_list : List[str]
        List of protein IDs.
    seq_list : List[str]
        List of protein sequences.
    hmm_path : str
        Path to the profile HMM file.

    Returns
    -------
    domain_df_dict : Dict[str, pd.DataFrame]
        Dictionary mapping protein_id to DataFrame of resolved domains with
        columns:

        (1) ``domain`` (str): Name of the profile HMM
        (2) ``from`` (int): Start index of the domain (1-based)
        (3) ``to`` (int): End index of the domain (1-based, inclusive)
        (4) ``seq`` (str): Protein sequence corresponding to the domain

    """

    # (1) Scan protein sequences
    scan_df = _scan_hmm_pyhmmer(protein_id_list, seq_list, hmm_path)

    domain_df_dict = {}
    for protein_id in scan_df['protein_id'].unique():
        scan_protein_df = scan_df[scan_df['protein_id'] == protein_id]
        seq = seq_list[np.where(np.array(protein_id_list) == protein_id)[0][0]]

        # (2) Resolve nested domains
        scan_protein_df = _resolve_nested_domains(scan_protein_df)

        # (3) Resolve partial overlaps
        label_list = _assign_overlap(scan_protein_df)

        # (4) Convert per-residue labels to ranges
        domain_list, start_list, end_list = _per_residue_to_ranges(label_list)
        domain_df = pd.DataFrame({'env_from': start_list, 'env_to': end_list},
                              index=domain_list)

        # (5) Rescan trimmed domains
        domain_df = domain_df.join(scan_protein_df, lsuffix='_after',
                                   rsuffix='_before')
        length_before = domain_df['env_to_before'] - domain_df[
            'env_from_before'] + 1
        length_after = domain_df['env_to_after'] - domain_df[
            'env_from_after'] + 1
        domain_df['trimmed'] = length_after < length_before
        domain_df = _rescan_trimmed_domains(domain_df, seq, hmm_path)

        # (6) Build final DataFrame
        domain_df = pd.DataFrame({
            'domain': [e.split('.')[0] for e in domain_df.index],
            'from': domain_df['env_from_after'],
            'to': domain_df['env_to_after'],
            'seq': [seq[i - 1:j] for (i, j) in zip(domain_df['env_from_after'],
                                                   domain_df['env_to_after'])]
        })
        domain_df_dict[protein_id] = domain_df
    return domain_df_dict
