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
    X, y = [], []

    for i in gridpoints:
        path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_of_spectra}/gp{i}_spectra_{shuffle_and_split_type}.hdf5"
        spec_file = SpectraCustomHDF5(path_to_file)
        _, flux = spec_file.get_all_spectra()  
        
        if reduced_dataset:
            np.random.shuffle(flux)
            flux = flux[:1000]  # Train on only 10% of the available data

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

    X_train, y_train = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "train", reduced_dataset)
    X_eval, y_eval = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "eval", reduced_dataset)
    X_test, y_test = build_dataset_for_gridpoints(gps_list, suite_of_spectra, "test", reduced_dataset)

    train_dataset = SpectraCosmoDataset(X_train, y_train)
    eval_dataset  = SpectraCosmoDataset(X_eval,  y_eval)
    test_dataset  = SpectraCosmoDataset(X_test,  y_test)

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


def get_sdss_spectra(resid_file_path, pmf_list, speclya_basepath="/virgotng/mpia/obs/SDSS/BOSSLyaDR9_spectra",
                      min_wavelength=3600.0, max_wavelength=3950.0, return_pmf_list=False):
    """ Calculates corrected SDSS BOSS Lyman alpha forest spectra from given pmf list and paths within the given
    wavelength range. Note: If pmf_list has thousands of entries, this function might take a couple of minutes to
    run, as it has to open a data file for every spectrum.

    Args:
        resid_file_path (string): path to the resid file
        pmf_list (list): list of (PLATE, MJD, FIBERID) tuples. These set which spectra to load from the survey
        speclya_basepath (str, optional): Basepath where the SDSS Survey data is. Defaults to "/virgotng/mpia/obs/SDSS/BOSSLyaDR9_spectra".
        min_wavelength (float, optional): Lower cutoff for the wavelengths. Defaults to 3600.0.
        max_wavelength (float, optional): Upper cutoff for the wavelengths. Defaults to 3950.0.
        return_pmf_list (bool, optional): Whether to return the pmf list of the used spectra. This can be used to make a custom catalogue

    Returns:
        snrs (list): List of np.arrays containing the SNR per pixel for every spectrum
        wavelengths_boss_specs (list): List of np.arrays of wavelengths for every spectrum 
        fluxes_boss_specs (list): List of np.arrays of fluxes for every spectrum
    """
    # -------- Read RESID File --------
    resid_file = resid_file_path
    resid = []
    resid_lam = []

    with open(resid_file) as rfile:
        reader = csv.reader(rfile, delimiter=" ")
        for i, row in enumerate(reader):
            if i != 0:
                resid.append(float(row[-1]))
                resid_lam.append(float(row[1]))

    resid_lam_file = np.array(resid_lam)
    resid_file = np.array(resid)
    # --------------------------------

    wavelengths_boss_specs = []
    fluxes_boss_specs = []
    snrs = []
    pmf_return_list = []

    for plate, mjd, fiber in tqdm.tqdm(pmf_list):
        # -------- Read Spectra file --------
        fiber_str = f"{fiber:04d}"
        filename = f"speclya-{plate}-{mjd}-{fiber_str}.fits"
        file_path = f"{speclya_basepath}/{plate}/" + filename

        with fits.open(file_path) as hdul:
            data = hdul[1].data

            loglam   = data["LOGLAM"]
            flux     = data["FLUX"]
            ivar     = data["IVAR"]
            cont     = data["CONT"]
            dla_corr = data["DLA_CORR"]
            mask_comb = data["MASK_COMB"]
            dla_corr = data["DLA_CORR"]
            noise_corr = data["NOISE_CORR"]

        # -------- Select correct range for Resids --------

        i0 = np.where(np.isclose(resid_lam_file, loglam[0]))[0][0]
        i1 = np.where(np.isclose(resid_lam_file, loglam[-1]))[0][0] + 1
        resid = resid_file[i0:i1]

        # -------------------------------------------------

        # Convert wavelengths to angstroms
        wavelength = 10**loglam

        # SDSS-recommended pixel mask
        good_pixel = (
            (mask_comb == 0) &
            (ivar > 0)
        )

        # Initialize F, SIGMA_F with NaNs
        F = np.full_like(flux, np.nan)
        SIGMA_F = np.full_like(flux, np.nan)

        # Compute F and SIGMA_F ONLY for good pixels
        F[good_pixel] = (flux[good_pixel] * dla_corr[good_pixel] / ((cont[good_pixel] * resid[good_pixel])))
        SIGMA_F[good_pixel] = np.sqrt(ivar[good_pixel] * resid[good_pixel]**2 * noise_corr[good_pixel]**2 * cont[good_pixel]**2 / dla_corr[good_pixel]**2)

        # -------- Select Lyα forest region --------
        forest_mask = (
            (wavelength >= min_wavelength) &
            (wavelength <= max_wavelength)
        )

        wave_sel = wavelength[forest_mask]
        F_sel    = F[forest_mask]
        Sigma_F_sel = SIGMA_F[forest_mask]

        # -------- Filter out spectra with Nans and Infs --------

        if np.any(np.isnan(F_sel)) or (not np.all(np.isfinite(F_sel))):
            continue

        # -------------------------------------------------------
        
        snrs.append(Sigma_F_sel)
        wavelengths_boss_specs.append(wave_sel)
        fluxes_boss_specs.append(F_sel)
        if return_pmf_list:
            pmf_return_list.append((plate, mjd, fiber))

    if return_pmf_list:
        return snrs, wavelengths_boss_specs, fluxes_boss_specs, pmf_return_list
    else:
        return snrs, wavelengths_boss_specs, fluxes_boss_specs


def get_spectra_reference_point(suite_of_spectra, n_spectra=None):
    """ Load the spectra and cosmo pars from the reference box of a suit of spectra

    Args:
        suite_of_spectra (string): suite of spectra to use
        n_spectra (int, optional): number of spectra to load

    Returns:
        X (np.array): Array of arrays of flux values of all spectra
        y (np.array): Array of arrays of cosmo params of all spectra (Note: this will be highly
                      redundent here and is only in this form to match the usual data structure)
    """
    X, y = [], []

    path_to_file = f"/vera/ptmp/gc/jerbo/training_data/{suite_of_spectra}/reference_point_spectra.hdf5"
    spec_file = SpectraCustomHDF5(path_to_file)
    _, flux = spec_file.get_all_spectra()  

    if n_spectra is not None:
        np.random.shuffle(flux)
        flux = flux[:n_spectra]  # Train on only 10% of the available data

    metadata = spec_file.get_header()
    params = metadata["Omega0"], metadata["OmegaBaryon"], metadata["OmegaLambda"], metadata["HubbleParam"]

    for spec in flux:
        X.append(spec)
        y.append(params)

    X, y = np.array(X), np.array(y)

    return X, y


if __name__ == "__main__":
    ...