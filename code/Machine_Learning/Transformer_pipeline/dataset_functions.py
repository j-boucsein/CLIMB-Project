import torch
import numpy as np
import csv
from astropy.io import fits
from torch.utils.data import Dataset
from astropy.table import Table
import tqdm

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.spectra_helpers import SpectraCustomHDF5


def build_dataset_for_gridpoints(gridpoints, suite_of_spectra, shuffle_and_split_type, reduced_dataset=False):
    """Collects the Flux data from the gridpoint spectra files and returns them as X and y data np.arrays

    Args:
        gridpoints (list): list with the grid point numbers to use
        suite_of_spectra (string): suite of spectra to use
        shuffle_and_split_type (string): type of file can be one of ("train", "eval", "test")
        reduced_dataset (bool): if True, only 1000 spectra are used per file (= 10% of the dataset)

    Returns:
        np.array(X): Array of spectra (flux values)
        np.array(y): Array of cosmo parameters (for each spectrum)
    """
    X, X_mask, y = [], [], []

    for i in gridpoints:
        path_to_file = f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/{suite_of_spectra}/gridpoint{i}_{shuffle_and_split_type}.hdf5"
        spec_file = SpectraCustomHDF5(path_to_file)
        _, flux, mask = spec_file.get_all_spectra_with_mask()  

        assert mask is not None, f"No masks found in {path_to_file=}"
        
        if reduced_dataset:
            flux = flux[:1000]  # Train on only 10% of the available data
            mask = mask[:1000]

        metadata = spec_file.get_header()
        params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

        for j in range(len(flux)):
            X.append(flux[j])
            X_mask.append(mask[j])
            y.append(params)

    return np.array(X), np.array(X_mask, dtype=bool), np.array(y)


class SpectraCosmoDataset(Dataset):
    """Class that defines a custom dataset for the spectra data

    Args:
        X (np.array): Array of spectra (flux values)
        mask (np.array): Boolean array, True for valid pixels. Same shape as X
        y (np.array): Array of cosmo parameters (for each spectrum)
        dtype (torch.dtype, optional): dtype used for X and y
    """
    def __init__(self, X, mask, y, dtype=torch.float32):
        mask = np.asarray(mask, dtype=bool)  # integer masks would index rows instead of pixels
        assert np.isfinite(X[mask]).all(), "X contains NaN or Inf values in the valid pixels"
        self.X = torch.tensor(X, dtype=dtype)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=dtype)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]
    

def _normalize(dataset, y_mean, y_std):
    """Normalize the y data of the given dataset using the other given parameters

    Args:
        dataset (torch.utils.data.Dataset): dataset to be normalized
        y_mean (torch.Tensor): mean of the training y data
        y_std (torch.Tensor): std of the training y data
    """
    dataset.y = (dataset.y - y_mean) / y_std


def get_shuffled_and_split_datasets(suite_of_spectra, reduced_dataset=False):
    """Create training dataset where all boxes are included but split spacially into train, eval, test sets

    Args:
        suite_of_spectra (string): suite of spectra to use
        reduced_dataset (bool, optional): set this to true to only retrieve 10% of the dataset

    Returns:
        train_dataset (SpectraCosmoDataset): Dataset with the training data
        eval_dataset (SpectraCosmoDataset): Dataset with the evaluation data
        test_dataset (SpectraCosmoDataset): Dataset with the test data
        y_mean (torch.Tensor): mean of the y data from the training set
        y_std (torch.Tensor): std of the y data from the training set
    """

    gps_list = [i for i in range(50)]

    X_train, X_mask_train, y_train = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "train", reduced_dataset)
    X_eval, X_mask_eval, y_eval = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "eval", reduced_dataset)
    X_test, X_mask_test, y_test = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "test", reduced_dataset)

    train_dataset = SpectraCosmoDataset(X_train, X_mask_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  X_mask_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  X_mask_test,  y_test)

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, y_mean, y_std)
    _normalize(eval_dataset, y_mean, y_std)
    _normalize(test_dataset, y_mean, y_std)

    return train_dataset, eval_dataset, test_dataset, y_mean, y_std


def get_sdss_file_ids(path, snr_lowe_edge, snr_upper_edge=np.inf):
    """ Get a list of (PLATE, MJD, FIBERID) from the SDSS BOSS Lyman alpha forest catalogue
      of spectra with S/N in given range.

    Args:
        path (string): path to the catalogue file
        snr_lowe_edge (float): lower filter for the signa to noise ratio of the spectra
        snr_upper_edge (float, optional): upper filter for the signa to noise ratio of the spectra

    Returns:
        list: list of (PLATE, MJD, FIBERID) tuples for spectra in the SDSS BOSS Lyman alpha forest catalogue
    """
    lya_cat = Table.read(path)
    pmf_list = []

    for row in lya_cat:
        if row["SNR"] > snr_lowe_edge and row["SNR"] < snr_upper_edge:
            pmf_list.append((row["PLATE"], row["MJD"], row["FIBERID"]))

    return pmf_list


def get_spectra_reference_point(suite_of_spectra):
    """ Load the spectra and cosmo pars from the reference box of a suit of spectra

    Args:
        suite_of_spectra (string): suite of spectra to use
        n_spectra (int, optional): number of spectra to load

    Returns:
        X (np.array): Array of arrays of flux values of all spectra
        y (np.array): Array of arrays of cosmo params of all spectra (Note: this will be highly
                      redundent here and is only in this form to match the usual data structure)
    """
    X, X_mask, y = [], [], []

    path_to_file_base = f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/{suite_of_spectra}/reference"
    train_path = path_to_file_base + "_train.hdf5"
    test_path = path_to_file_base + "_test.hdf5"
    eval_path = path_to_file_base + "_eval.hdf5"

    spec_file_train = SpectraCustomHDF5(train_path)
    spec_file_test = SpectraCustomHDF5(test_path)
    spec_file_eval = SpectraCustomHDF5(eval_path)

    _, flux_train, mask_train = spec_file_train.get_all_spectra_with_mask()  
    _, flux_test, mask_test = spec_file_test.get_all_spectra_with_mask()  
    _, flux_eval, mask_eval = spec_file_eval.get_all_spectra_with_mask()  

    assert (mask_train is not None) and (mask_test is not None) and (mask_eval is not None), f"No masks found in {path_to_file_base=}"

    metadata = spec_file_train.get_header()
    params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

    # not my proudest work, but the simplest solution atm...
    for i in range(len(flux_train)):
        X.append(flux_train[i])
        X_mask.append(mask_train[i])
        y.append(params)

    for i in range(len(flux_test)):
        X.append(flux_test[i])
        X_mask.append(mask_test[i])
        y.append(params)

    for i in range(len(flux_eval)):
        X.append(flux_eval[i])
        X_mask.append(mask_eval[i])
        y.append(params)

    X, X_mask, y = np.array(X), np.array(X_mask, dtype=bool), np.array(y)

    assert np.isfinite(X[X_mask]).all(), "X contains NaN or Inf values in the valid pixels"

    return X, X_mask, y


if __name__ == "__main__":
    ...