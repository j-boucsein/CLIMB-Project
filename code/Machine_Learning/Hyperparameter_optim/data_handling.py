import torch
import numpy as np
from torch.utils.data import Dataset

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../../../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.spectra_helpers import SpectraCustomHDF5


def write_to_log_file(file_path, message, mode="a"):
    """Function to write content to the log file

    Args:
        file_path (string): path to the log file
        message (string): message to write to the log file
        mode (str, optional): Mode of opening the file. Defaults to "a".
    """
    with open(file_path, mode) as file:
        file.write(message)
        file.write("\n")


def build_dataset_for_gridpoints(gridpoints, suite_of_spectra, shuffle_and_split_type=None):
    """Collects the Flux data from the gridpoint spectra files and returns them as X and y data np.arrays

    Args:
        gridpoints (list): list with the grid point numbers to use
        suite_to_use (string): simulation suite to use

    Returns:
        np.array(X): Array of spectra (flux values)
        np.array(y): Array of cosmo parameters (for each spectrum)
    """
    X, y = [], []

    for i in gridpoints:
        if shuffle_and_split_type is None:
            path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_of_spectra}/gp{i}_spectra.hdf5"
        else:
            path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_of_spectra}/gp{i}_spectra_{shuffle_and_split_type}.hdf5"
        spec_file = SpectraCustomHDF5(path_to_file)
        _, flux = spec_file.get_all_spectra()  
        
        np.random.shuffle(flux)
        flux = flux[:1000]  # (1000, 468) Train on only 10% of the available data

        metadata = spec_file.get_header()
        params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

        for spec in flux:
            X.append(spec)
            y.append(params)

    return np.array(X), np.array(y)


class SpectraCosmoDataset(Dataset):
    """Class that defines a custom dataset for the spectra data
    """
    def __init__(self, X, y, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def _normalize(dataset, y_mean, y_std):
    """Normalize the given dataset using the other given parameters

    Args:
        dataset (torch.utils.data.Dataset): dataset to be normalized
        y_mean (_type_): mean of the training y data
        y_std (_type_): std of the training y data
    """
    dataset.y = (dataset.y - y_mean) / y_std


def get_shuffled_and_split_datasets(suite_of_spectra):

    gps_list = [i for i in range(50)]

    X_train, y_train = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "train")
    X_eval, y_eval = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "eval")
    X_test, y_test = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "test")

    train_dataset = SpectraCosmoDataset(X_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  y_test)

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, y_mean, y_std)
    _normalize(eval_dataset, y_mean, y_std)
    _normalize(test_dataset, y_mean, y_std)

    return train_dataset, eval_dataset, test_dataset, y_mean, y_std