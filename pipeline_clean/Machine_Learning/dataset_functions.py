import torch
import numpy as np
from torch.utils.data import Dataset

import sys, os
# This is not super pretty, but I think this is the best way to import stuff from ../util?
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.spectra_io import SpectraCustomHDF5


def read_hash_mapping(boxes_dir):
    """Read the hash_mapping.txt that the data pipeline writes into training_boxes and reference_boxes

    Args:
        boxes_dir (string): folder that contains hash_mapping.txt and the spectra files

    Returns:
        list: one (box path, file hash) tuple per box, in the order of the file
    """
    boxes = []

    with open(os.path.join(boxes_dir, "hash_mapping.txt")) as f:
        for line in f:
            box_path, hash_name = line.strip().split(" -> ")
            boxes.append((box_path, hash_name))

    return boxes


def build_dataset_for_boxes(boxes_dir, shuffle_and_split_type, reduced_dataset=False):
    """Collects the Flux data from the spectra files of all boxes in a folder and returns them as X and y data np.arrays

    Args:
        boxes_dir (string): folder with the spectra files and their hash_mapping.txt
        shuffle_and_split_type (string): type of file can be one of ("training", "evaluation", "testing")
        reduced_dataset (bool): if True, only the first 10% of the spectra of every file are used

    Returns:
        np.array(X): Array of spectra (flux values)
        np.array(y): Array of cosmo parameters (for each spectrum)
    """
    X, X_mask, y = [], [], []
    wavelengths_first_file = None

    for _, hash_name in read_hash_mapping(boxes_dir):
        path_to_file = os.path.join(boxes_dir, f"{hash_name}_{shuffle_and_split_type}.hdf5")
        spec_file = SpectraCustomHDF5(path_to_file)
        wavelengths, flux, mask = spec_file.get_all_spectra_with_mask()

        assert mask is not None, f"No masks found in {path_to_file=}"

        # the model treats pixel i as the same wavelength in every spectrum, so all files need the same grid
        if wavelengths_first_file is None:
            wavelengths_first_file = wavelengths
        assert np.array_equal(wavelengths, wavelengths_first_file), f"Wavelength grid of {path_to_file} differs from the first file in {boxes_dir}"

        if reduced_dataset:
            n_reduced = len(flux) // 10  # Train on only 10% of the available data
            flux = flux[:n_reduced]
            mask = mask[:n_reduced]

        metadata = spec_file.get_header()
        params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

        # One array per file, they are joined after the loop. float32 is what the model works with
        X.append(flux.astype(np.float32))
        X_mask.append(mask)
        y.append(np.tile(params, (len(flux), 1)))  # the same parameters for every spectrum of the file

    return np.concatenate(X), np.concatenate(X_mask).astype(bool), np.concatenate(y)


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
        # the model would return a number for such a spectrum, but it would carry no information
        assert mask.any(axis=1).all(), "There are spectra without a single valid pixel"
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


# Ways of normalising the input flux, searched over by the hyperparameter optimisation.
# The raw flux is badly behaved: the valid pixels have a median of 0.77 and a 99th percentile of
# 2.4, but a maximum of about 1.6e4, so its standard deviation (7.5) is set almost entirely by a
# ~1e-4 tail. Feeding that straight into the embedding swamps the positional encoding and lets a
# handful of pixels dominate the gradient. "none" reproduces the old behaviour.
X_TRANSFORMS = ("none", "standardize", "clip_standardize", "log1p_standardize", "asinh_standardize")


def fit_x_transform(train_dataset, kind, clip_quantile=0.999):
    """Fit a flux transform on the valid pixels of the training set

    Args:
        train_dataset (SpectraCosmoDataset): dataset the statistics are taken from
        kind (string): one of X_TRANSFORMS
        clip_quantile (float, optional): upper quantile used by "clip_standardize", the lower one
                                        is its mirror image (1 - clip_quantile)

    Returns:
        dict: the fitted parameters, to be handed to apply_x_transform
    """
    if kind not in X_TRANSFORMS:
        raise ValueError(f"Unknown x_transform {kind!r}, expected one of {X_TRANSFORMS}")

    if kind == "none":
        return {"kind": kind}

    values = train_dataset.X[train_dataset.mask].to(torch.float64)

    params = {"kind": kind}

    if kind == "clip_standardize":
        # The tail is a tiny fraction of the pixels but sets the scale of the whole input, so it is
        # clipped away before the statistics are taken. Both ends are clipped: the flux happens to
        # be non negative today, so the lower quantile sits at 0 and does nothing, but clipping only
        # the top would silently let a future negative tail set the scale again.
        lo = torch.quantile(values, 1.0 - clip_quantile)
        hi = torch.quantile(values, clip_quantile)
        params["clip_min"] = float(lo)
        params["clip_max"] = float(hi)
        values = values.clamp(min=lo, max=hi)

    elif kind == "log1p_standardize":
        # The flux is non negative (checked: min is exactly 0), so log1p is well defined
        values = torch.log1p(values.clamp(min=0.0))

    elif kind == "asinh_standardize":
        # Like log1p for the tail but linear near zero, so the bulk of the distribution is not
        # squashed
        values = torch.asinh(values)

    params["mean"] = float(values.mean())
    params["std"] = float(values.std()) + 1e-8

    return params


def transform_flux(X, mask, params):
    """Apply a fitted flux transform to a flux tensor

    Kept separate from apply_x_transform so inference can normalise SDSS or reference box spectra
    with the statistics the model was trained on. Feeding raw flux to a model trained on
    transformed flux produces silently wrong predictions.

    Invalid pixels are NaN in the raw data. They are set to 0 here, which is what the model does
    internally anyway, so no NaN can reach a statistic or a gradient.

    Args:
        X (torch.Tensor): flux, shape (n_spectra, len_spectra)
        mask (torch.Tensor): boolean, True for valid pixels, same shape as X
        params (dict): output of fit_x_transform

    Returns:
        torch.Tensor: the transformed flux, float32, zero on the invalid pixels
    """
    kind = params["kind"]

    if kind != "none":
        X = X.to(torch.float64)

        if kind == "clip_standardize":
            X = X.clamp(min=params["clip_min"], max=params["clip_max"])
        elif kind == "log1p_standardize":
            X = torch.log1p(X.clamp(min=0.0))
        elif kind == "asinh_standardize":
            X = torch.asinh(X)

        X = (X - params["mean"]) / params["std"]
        X = X.to(torch.float32)

    # NaN in the masked out pixels would survive a multiplication by a zero attention weight
    return torch.where(mask, X, torch.zeros_like(X))


def apply_x_transform(dataset, params):
    """Apply a fitted flux transform to a dataset, in place

    Args:
        dataset (SpectraCosmoDataset): dataset to transform
        params (dict): output of fit_x_transform
    """
    dataset.X = transform_flux(dataset.X, dataset.mask, params)


def get_shuffled_and_split_datasets(dataset_dir, reduced_dataset=False, x_transform="none"):
    """Create training dataset where all boxes are included but split spacially into train, eval, test sets

    Args:
        dataset_dir (string): folder of the dataset (the one that contains training_boxes and reference_boxes)
        reduced_dataset (bool, optional): set this to true to only retrieve 10% of the dataset
        x_transform (string, optional): flux normalisation, one of X_TRANSFORMS. Fitted on the
                                       training set and applied unchanged to eval and test

    Returns:
        train_dataset (SpectraCosmoDataset): Dataset with the training data
        eval_dataset (SpectraCosmoDataset): Dataset with the evaluation data
        test_dataset (SpectraCosmoDataset): Dataset with the test data
        y_mean (torch.Tensor): mean of the y data from the training set
        y_std (torch.Tensor): std of the y data from the training set
        x_params (dict): the flux transform fitted on the training set (output of fit_x_transform)
    """

    boxes_dir = os.path.join(dataset_dir, "training_boxes")

    X_train, X_mask_train, y_train = build_dataset_for_boxes(boxes_dir, "training", reduced_dataset)
    X_eval, X_mask_eval, y_eval = build_dataset_for_boxes(boxes_dir, "evaluation", reduced_dataset)
    X_test, X_mask_test, y_test = build_dataset_for_boxes(boxes_dir, "testing", reduced_dataset)

    train_dataset = SpectraCosmoDataset(X_train, X_mask_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  X_mask_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  X_mask_test,  y_test)

    y_mean = train_dataset.y.mean(dim=0)
    y_std  = train_dataset.y.std(dim=0) + 1e-8

    _normalize(train_dataset, y_mean, y_std)
    _normalize(eval_dataset, y_mean, y_std)
    _normalize(test_dataset, y_mean, y_std)

    # The flux transform is fitted on the training set only, so no information leaks from eval or
    # test into the normalisation
    x_params = fit_x_transform(train_dataset, x_transform)
    apply_x_transform(train_dataset, x_params)
    apply_x_transform(eval_dataset, x_params)
    apply_x_transform(test_dataset, x_params)

    return train_dataset, eval_dataset, test_dataset, y_mean, y_std, x_params


def get_spectra_reference_point(dataset_dir, hash_name):
    """ Load the spectra and cosmo pars from the reference box of a suit of spectra

    Args:
        dataset_dir (string): folder of the dataset (the one that contains training_boxes and reference_boxes)
        hash_name (string): file hash of the reference box, see reference_boxes/hash_mapping.txt

    Returns:
        X (np.array): Array of arrays of flux values of all spectra
        y (np.array): Array of arrays of cosmo params of all spectra (Note: this will be highly
                      redundent here and is only in this form to match the usual data structure)
    """
    path_to_file_base = os.path.join(dataset_dir, "reference_boxes", hash_name)
    train_path = path_to_file_base + "_training.hdf5"
    test_path = path_to_file_base + "_testing.hdf5"
    eval_path = path_to_file_base + "_evaluation.hdf5"

    spec_file_train = SpectraCustomHDF5(train_path)
    spec_file_test = SpectraCustomHDF5(test_path)
    spec_file_eval = SpectraCustomHDF5(eval_path)

    _, flux_train, mask_train = spec_file_train.get_all_spectra_with_mask()  
    _, flux_test, mask_test = spec_file_test.get_all_spectra_with_mask()  
    _, flux_eval, mask_eval = spec_file_eval.get_all_spectra_with_mask()  

    assert (mask_train is not None) and (mask_test is not None) and (mask_eval is not None), f"No masks found in {path_to_file_base=}"

    metadata = spec_file_train.get_header()
    params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

    # The three files of the box are joined to one array each. float32 is what the model works with
    X = np.concatenate([flux_train, flux_test, flux_eval]).astype(np.float32)
    X_mask = np.concatenate([mask_train, mask_test, mask_eval]).astype(bool)
    y = np.tile(params, (len(X), 1))  # the same parameters for every spectrum

    assert np.isfinite(X[X_mask]).all(), "X contains NaN or Inf values in the valid pixels"

    return X, X_mask, y
