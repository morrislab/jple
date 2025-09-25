"""
run_jple.py

Main entry point for running Joint Protein-Ligand Embedding (JPLE).

"""
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from jple import domain_utils, io_utils, motif_utils, rep_utils
from jple.model import JPLE


def prepare_protein_reps(protein_dict_dict: Dict[str, Dict],
                         x_kmer_list: List[str]=None) -> Tuple[
    List[str], np.ndarray, List[str]
]:
    """
    Extract protein regions and generate protein k-mer representations.

    Parameters
    ----------
    protein_dict_dict: Dict[str, Dict]
        Dictionary of protein dictionaries with results.
    x_kmer_list : List[str], optional
        Predefined list of k-mers to use. If None, all observed k-mers will be
        included.

    Returns
    -------
    jple_protein_id_list : List[str]
        IDs of the proteins with at least one selected domain.
    x : np.ndarray
        Protein representations.
    x_kmer_list : List[str]
        List of peptide k-mers corresponding to the columns of `x`.

    """

    # Extract the region sequences
    jple_protein_id_list, region_seq_list_list = \
        rep_utils.get_region(protein_dict_dict,
                             ['RRMs_3D_hmm_extended', 'KH_1'])

    # Generate the protein representations
    x, x_kmer_list = \
        rep_utils.generate_rep(region_seq_list_list, x_kmer_list=x_kmer_list)

    # Normalize the protein representations
    x = (x.T / np.sqrt(np.sum(x * x, axis=1))).T
    return jple_protein_id_list, x, x_kmer_list


def prepare_binding_profiles(y_path: str, jple_protein_id_list: List[str],
                             mode: str) -> Tuple[
    List[str], np.ndarray, List[str]
]:
    """
    Load and normalize binding profiles.

    Parameters
    ----------
    y_path : str
        Path to the binding profiles.
    jple_protein_id_list : List[str]
        IDs of the proteins with at least one selected domain.
    mode : str
        JPLE execution mode.

    Returns
    -------
    jple_protein_id_list : List[str]
        IDs of the proteins.
    y : np.ndarray
        Binding profiles.
    y_kmer_list : List[str]
        List of nucleic acid k-mers corresponding to the columns of `y`.

    """

    # Read the binding profiles
    y_df = pd.read_csv(y_path, sep='\t', index_col=0).dropna()

    # Check if the protein IDs are unique
    seen_set, duplicate_set = set(), set()
    for protein_id in y_df.columns:
        if protein_id in seen_set:
            duplicate_set.add(protein_id)
        seen_set.add(protein_id)
    if duplicate_set:
        raise ValueError('Z-score table indices must be unique. ',
                         f'Duplicates found: {", ".join(duplicate_set)}')

    # Check if the protein IDs matches those in the FASTA file
    if mode == 'train':
        missing_set = set(jple_protein_id_list) - set(y_df.columns)
        if missing_set:
            raise ValueError('The following protein IDs are not present in '
                             'the Z-score table: '
                             f'{", ".join(map(str, missing_set))}')
        y_df = y_df.loc[:, jple_protein_id_list]

    # Convert the dataframe to matrix
    y = y_df.to_numpy().T
    jple_protein_id_list = y_df.columns.to_list()
    y_kmer_list = y_df.index.to_list()

    # Normalize the binding profiles
    y = (y.T / np.sqrt(np.sum(y * y, axis=1))).T
    return jple_protein_id_list, y, y_kmer_list

def jple_train(jple: JPLE, x: np.ndarray, y: np.ndarray) -> JPLE:
    """
    Perform JPLE training.

    Parameters
    ----------
    jple : JPLE
        JPLE instance.
    x : np.ndarray
        Training-set representations.
    y : np.ndarray
        Training-set binding profiles.

    Returns
    -------
    jple : JPLE
        Trained JPLE instance.

    """
    jple.fit(x, y)
    return jple


def jple_predict_protein(jple: JPLE, x: np.ndarray,
                         test_protein_id_list: List[str],
                         train_protein_id_list: List[str],
                         y_train_kmer_list: List[str],
                         protein_dict_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Perform JPLE protein query.

    Parameters
    ----------
    jple : JPLE
        Trained JPLE instance.
    x : np.ndarray
        Test-set representations.
    test_protein_id_list : List[str]
        IDs of the test-set proteins with at least one selected domain.
    train_protein_id_list : List[str]
        IDs of the training-set proteins with at least one selected domain.
    y_train_kmer_list : List[str]
        List of training-set nucleic acid k-mers.
    protein_dict_dict : List[str, Dict]
        Dictionary of protein dictionaries with results.

    Returns
    -------
    protein_dict_dict : Dict[str, Dict]
        Dictionary of protein dictionaries with the following added results:

        (1) zscore_df: Predicted binding profiles.
        (2) dist: JPLE e-dist.
        (3) neighbor_df: Neighboring training-set proteins and their
            contributions.
        (4) pwm: Predicted PWM.
        (5) iupac: Predicted IUPAC motif.

    """

    # Perform prediction
    y_pred_all, dist_list, neighbor_all_df = jple.predict_protein(x)

    # Normalize the predicted Z-scores
    y_pred_all /= np.std(y_pred_all, axis=1)[:, None]

    # Split the results
    for idx, test_protein_id in enumerate(test_protein_id_list):
        protein_dict = protein_dict_dict[test_protein_id]

        # Predicted Z-scores
        zscore_df = pd.DataFrame(
            y_pred_all[idx].T,
            index=pd.Index(y_train_kmer_list, name='kmer'),
            columns=['zscore']
        )
        protein_dict['zscore_df'] = zscore_df

        # e-dist
        protein_dict['dist'] = dist_list[idx]

        # Neighbors
        neighbor_df = \
            neighbor_all_df[neighbor_all_df['test_idx'] == idx].copy()
        neighbor_df['train_protein_id'] = neighbor_df['train_idx'].map(
            dict(enumerate(train_protein_id_list))
        )
        neighbor_df = neighbor_df[neighbor_df['contribution'] >= 10][
            ['train_protein_id', 'dist', 'contribution']
        ]
        protein_dict['neighbor_df'] = neighbor_df

        # Generate the PWM
        pwm = motif_utils.compute_pwm(zscore_df)
        protein_dict['pwm'] = pwm

        # Generate the IUPAC motif
        protein_dict['iupac'] = motif_utils.compute_iupac(pwm)
    return protein_dict_dict


def jple_predict_na(jple: JPLE, y: np.ndarray,
                    test_protein_id_list: List[str],
                    x_train_kmer_list: List[str],
                    protein_dict_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Perform JPLE nucleic acid query.

    Parameters
    ----------
    jple : JPLE
        Trained JPLE instance.
    y : np.ndarray
        Test-set binding profiles.
    test_protein_id_list : List[str]
        IDs of the test-set proteins with at least one selected domain.
    x_train_kmer_list : List[str]
        List of training-set peptide k-mers.
    protein_dict_dict : Dict[str, Dict]
        Dictionary of protein dictionaries with results.

    Returns
    -------
    protein_dict_dict : Dict[str, Dict]
        Dictionary of protein dictionaries with the following added results:

        (1) importance_df: Predicted residue importance profile.
        (2) dist: JPLE e-dist.

    """

    # Perform prediction
    x_pred_all, dist_list = jple.predict_na(y)

    # Normalize the residue importance profiles
    x_pred_all = (x_pred_all - x_pred_all.mean()) / x_pred_all.std()

    # Split the results
    for idx, test_protein_id in enumerate(test_protein_id_list):

        # Predicted residue importance profiles
        importance_df = pd.DataFrame(
            x_pred_all[idx].T,
            index=pd.Index(x_train_kmer_list, name='kmer'),
            columns=['importance']
        )
        protein_dict_dict[test_protein_id] = {}
        protein_dict_dict[test_protein_id]['importance_df'] = importance_df

        # e-dist
        protein_dict_dict[test_protein_id]['dist'] = dist_list[idx]
    return protein_dict_dict


def main(mode: str='predict_protein',
         param_path: str='data/processed/param_train.npz',
         fasta_path:str=None,
         hmm_path:str='data/processed/domain_rbp.hmm',
         y_path:str=None,
         output_path:str=None,
         return_results: bool=True) -> Dict[str, Dict]:
    """
    Entry point of the program.

    Parameters
    ----------
    mode : str
        JPLE Execution mode. Choose "train" to train a model, "predict_protein"
        (default) to perform a protein query, or "predict_na" to perform a
        nucleic acid query.
    param_path : str
        Path to the JPLE model parameter file in NPZ format. Model parameters
        will be saved here in "train" mode.
        (default: data/processed/param_train.npz)
    fasta_path : str
        Path to the input query FASTA file (required in "train" and
        "predict_protein" mode).
    hmm_path : str
        Path to the profile HMM model file
        (default: data/processed/domain_rbp.hmm).
    y_path : str
        Path to the binding profiles (required in "train" and  "predict_na"
        mode). The file should be tab-delimited, which each row representing a
        protein and each column representing a nucleic acid k-mer.
    output_path : str
        Path to the output folder for JPLE results. Results are not saved if not
        provided.
    return_results : bool
        Whether to return results or not (default: True).

    Returns
    -------
    protein_dict_dict : Dict[str, Dict]
        Dictionary of protein dictionaries. Each dictionary is indexed by
        its protein ID and may contain the following keys:

        (1) domain_df: Domain boundaries.
        (2) zscore_df: Predicted binding profiles.
        (3) dist: JPLE e-dist.
        (4) neighbor_df: Neighboring training-set proteins and their
            contributions.
        (5) pwm: Predicted PWM.
        (6) iupac: Predicted IUPAC motif.
        (7) importance_df: Predicted residue importance profiles.

    """

    # Validation
    if mode != 'predict_na' and fasta_path is None:
        raise ValueError('fasta_path is required when mode is "train" or '
                         '"predicted_protein".')
    if mode != 'predict_protein' and y_path is None:
        raise ValueError('y_path is required when mode is "train" or '
                         '"predicted_na".')

    # Load the model parameters
    if mode in ['predict_protein', 'predict_na']:
        logger.info(f'Loading the model parameters from {param_path}')
        param_dict = np.load(param_path, allow_pickle=True)
    else:
        param_dict = {'x_train_kmer_list': None}

    # Process the protein sequences
    if mode in ['train', 'predict_protein']:

        # Read the input protein sequences
        logger.info(f'Reading the input protein sequences from {fasta_path}')
        protein_dict_dict = io_utils.read_fasta(fasta_path)

        # Extract the protein domains
        logger.info('Extracting the protein domains...')
        domain_df_dict = \
            domain_utils.scan_hmm(
                list(protein_dict_dict.keys()),
                [e['protein_seq'] for e in protein_dict_dict.values()],
                hmm_path)
        for protein_id, domain_df in domain_df_dict.items():
            protein_dict_dict[protein_id]['domain_df'] = domain_df

        # Prepare the protein representations
        logger.info('Preparing the protein representations...')
        jple_protein_id_list, x, x_kmer_list = \
            prepare_protein_reps(protein_dict_dict,
                                 param_dict['x_train_kmer_list'])
    else:
        protein_dict_dict = {}
        jple_protein_id_list, x, x_kmer_list = None, None, None

    # Process the binding profiles
    if mode in ['train', 'predict_na']:

        # Prepare the binding profiles
        logger.info(f'Preparing the binding profiles from {y_path}')
        jple_protein_id_list, y, y_kmer_list = \
            prepare_binding_profiles(y_path, jple_protein_id_list, mode)
    else:
        y, y_kmer_list = None, None

    # Initialize JPLE
    jple = JPLE()

    # Load the JPLE parameters
    if mode in ['predict_protein', 'predict_na']:
        jple.load(y_train=param_dict['y_train'],
                  x_train_mean=param_dict['x_train_mean'],
                  y_train_mean=param_dict['y_train_mean'],
                  w_train=param_dict['w_train'],
                  v_train=param_dict['v_train'])

    # Run JPLE: train
    if mode == 'train' and x is not None and len(x) > 0:
        logger.info('JPLE: training the model...')
        jple_train(jple, x, y)
        param_dict = {
            'train_protein_id_list': jple_protein_id_list,
            'x_train_kmer_list': x_kmer_list,
            'y_train_kmer_list': y_kmer_list,
            'y_train': jple.y_train,
            'x_train_mean': jple.x_train_mean,
            'y_train_mean': jple.y_train_mean,
            'w_train': jple.w_train,
            'v_train': jple.v_train
        }

    # Run JPLE: predict_protein
    elif mode == 'predict_protein' and x is not None and len(x) > 0:
        logger.info('JPLE: performing protein query...')
        protein_dict_dict = \
            jple_predict_protein(jple, x, jple_protein_id_list,
                                 param_dict['train_protein_id_list'],
                                 param_dict['y_train_kmer_list'],
                                 protein_dict_dict)

    # Run JPLE: predict_na
    else:
        logger.info('JPLE: performing nucleic acid query...')
        protein_dict_dict = \
            jple_predict_na(jple, y, jple_protein_id_list,
                            param_dict['x_train_kmer_list'],
                            protein_dict_dict)

    # Save results
    if mode == 'train':
        logger.info(f'Saving the parameters to {param_path}')
        np.savez_compressed(param_path, **param_dict)
    elif output_path is not None:
        logger.info(f'Saving the results to {output_path}')
        io_utils.save_results(mode, output_path, protein_dict_dict)
    logging.info('Done.')

    # Return results only when called via API
    if return_results:
        return protein_dict_dict
