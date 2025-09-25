"""
model.py

Implementation of Joint Protein-Ligand Embedding (JPLE).

"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class JPLE:
    """
    Implementation of JPLE.

    """

    def __init__(self, num_eigenvector: int=122, threshold: float=0.01,
                 std: float=0.2) -> None:
        """
        Initialize the predictor.

        Parameters
        ----------
        num_eigenvector : int
            Number of retained eigenvectors.
        threshold: float
            Training-set proteins that have higher similarity than this
            threshold will be used for reconstruction.
        std : float
            Standard deviation of the RBF.

        """
        self.num_eigenvector = num_eigenvector
        self.threshold = threshold
        self.std = std

        self.y_train = None
        self.x_train_mean = None
        self.y_train_mean = None
        self.w_train = None
        self.v_train = None

    def load(self, y_train: np.ndarray, x_train_mean: np.ndarray,
             y_train_mean: np.ndarray, w_train: np.ndarray,
             v_train: np.ndarray) -> None:
        """
        Load the model parameters.

        Parameters
        ----------
        y_train : np.ndarray
            Training-set binding profiles.
        x_train_mean : np.ndarray
            Column means of the training-set representations.
        y_train_mean : np.ndarray
            Column means of the training-set binding profiles.
        w_train : np.ndarray
            Training-set embeddings.
        v_train : np.ndarray
            Training-set eigenvectors.

        """
        self.y_train = y_train
        self.x_train_mean = x_train_mean
        self.y_train_mean = y_train_mean
        self.w_train = w_train
        self.v_train = v_train

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Compute the latent embeddings of the training-set proteins


        Parameters
        ----------
        x_train : np.ndarray
            Training-set representations.
        y_train : np.ndarray
            Training-set binding profiles.

        """

        # Initialization
        self.y_train = y_train

        # Subtract each column by mean
        self.x_train_mean = x_train.mean(axis=0)
        self.y_train_mean = y_train.mean(axis=0)
        x_train = x_train - self.x_train_mean
        y_train = y_train - self.y_train_mean

        # Perform SVD
        xy_train = np.append(x_train, y_train, axis=1)
        u, s, v = np.linalg.svd(xy_train, full_matrices=False)
        num_nt = y_train.shape[1]

        # Compute the variance explained by each singular vector
        sig = np.sum(v[:, -num_nt:] ** 2, axis=1) * s ** 2

        # Select the most important singular vectors from V
        sorted_vector_idx = np.argsort(-sig)[:self.num_eigenvector]

        # Save the latent embeddings of the training-set proteins
        self.w_train = u[:, sorted_vector_idx] * s[sorted_vector_idx]
        self.v_train = v[sorted_vector_idx]

    def predict_protein(self, x_test: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Predict the binding profiles of the test-set proteins.

        Parameters
        ----------
        x_test : np.ndarray
            Representations of the test-set proteins.

        Returns
        -------
        y_pred : np.ndarray
            Predicted binding profiles of the test-set proteins.
        min_dist_list : np.array
            Distance between each test-set protein and its nearest training-set
            protein.
        neighbor_df : pd.DataFrame
            Neighboring training-set proteins and their contributions to the
            predicted binding profiles of the test-set proteins.

        """

        # Subtract each column by its mean in the training set
        x_test = x_test - self.x_train_mean

        # Compute the latent embeddings of the test-set proteins
        v_train_x = self.v_train[:, :x_test.shape[1]]
        w_test, _, _, _ = np.linalg.lstsq(v_train_x.T, x_test.T, rcond=None)
        w_test = w_test.T

        # Compute the similarities between the test-set and training-set
        # proteins in the latent space
        dist_mat = cdist(w_test, self.w_train, 'cosine')
        sim_mat = np.exp(-dist_mat ** 2 / self.std ** 2)

        # Iterate over the test-set proteins
        y_pred = []
        min_dist_list = []
        protein_idx_list_list = []
        sim_idx_list_list = []
        dist_subset_list_list = []
        prop_subset_list_list = []
        for protein_idx, (dist_list, sim_list) in \
                enumerate(zip(dist_mat, sim_mat)):

            # Filter for the training-set proteins that are not too distant
            sim_idx_list = np.argwhere(sim_list >= self.threshold).flatten()
            sim_idx_list = sim_idx_list[sim_list[sim_idx_list].argsort()][::-1]
            if len(sim_idx_list) == 0:
                sim_idx_list = [np.argmax(sim_list)]
            sim_idx_list_list.append(sim_idx_list)

            y_train_sim = self.y_train[sim_idx_list]
            sim_subset_list = sim_list[sim_idx_list]
            dist_subset_list = dist_list[sim_idx_list]

            # Compute the predicted binding profiles
            denom = np.sum(sim_subset_list)
            y_pred.append(
                np.sum(sim_subset_list[:, None] * y_train_sim, axis=0) / denom
            )
            min_dist_list.append(np.min(dist_list))

            # Get the distances
            dist_subset_list_list.append(dist_subset_list)

            # Get the proportions
            prop_subset_list_list.append(sim_subset_list / denom * 100)
            protein_idx_list_list.append([protein_idx] * len(sim_idx_list))

        # Generate neighbors dataframe
        neighbor_df = \
            pd.DataFrame({'test_idx': np.concatenate(protein_idx_list_list),
                          'train_idx': np.concatenate(sim_idx_list_list),
                          'dist': np.concatenate(dist_subset_list_list),
                          'contribution': np.concatenate(
                              prop_subset_list_list)})
        return np.array(y_pred), np.array(min_dist_list), neighbor_df

    def predict_na(self, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the residue importance profiles of the test-set binding
        profiles.

        Parameters
        ----------
        y_test : np.ndarray
            Binding profiles of the test-set proteins.

        Returns
        -------
        x_pred : np.ndarray
            Predicted residue importance profiles of the test-set proteins.
        min_dist_list : np.array
            Distance between each test-set protein and its nearest training-set
            protein.

        """

        # Subtract each column by its mean in the training set
        y_test = y_test - self.y_train_mean

        # Compute the latent embeddings of the test-set proteins
        v_train_y = self.v_train[:, -y_test.shape[1]:]
        w_test, _, _, _ = np.linalg.lstsq(v_train_y.T, y_test.T, rcond=None)
        w_test = w_test.T

        # Compute the similarities between the test and training-set proteins in
        # the latent space
        min_dist_list = cdist(w_test, self.w_train, 'cosine').min(axis=1)

        # Compute the residue importance profiles
        x_pred = np.dot(w_test, self.v_train[:, :-y_test.shape[1]])
        x_pred += self.x_train_mean
        return x_pred, min_dist_list
