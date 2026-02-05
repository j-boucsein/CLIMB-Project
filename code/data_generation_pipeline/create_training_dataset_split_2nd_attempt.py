import numpy as np
import random

# This is not super pretty, but I think this is the best way to import stuff from ..util?
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
from util.spectra_helpers import patch_spectra_together, add_noise_to_spectrum, SpectraCustomHDF5
from util.sim_data_helpers import get_cosmo_parameters, get_data_from_header


def make_dataset_spectra(list_of_sample_points, out_base_path, snr, min_wavelength=3550, max_wavelength=3950, n_spectra=10000):

    for i, current_sp in enumerate(list_of_sample_points):
        print(current_sp)
        base_spectra = np.arange(n_spectra)
        np.random.shuffle(base_spectra)

        wavelengths, augmented_spectra = patch_spectra_together(current_sp, base_spectra, min_wavelength, max_wavelength, n_spectra)
        
        augmented_spectra = np.stack(augmented_spectra, axis=0)

        # truncate the spectra to the given range
        mask = (wavelengths > min_wavelength) & (wavelengths < max_wavelength)
        wavelengths = wavelengths[mask]
        augmented_spectra = augmented_spectra[:, mask]

        noisy_spectra = []

        for spec in augmented_spectra:
            noisy_spectra.append(add_noise_to_spectrum(spec, snr=snr, distr="normal"))

        noisy_spectra = np.stack(noisy_spectra, axis=0)

        Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(current_sp)

        boxsize = get_data_from_header(current_sp, 3, "BoxSize")*1e-3  # convert to Mpc/h
        redshift = get_data_from_header(current_sp, 3, "Redshift")

        metadata_file = {
            "Omega0": Omega0,
            "OmegaLambda": OmegaLambda,
            "OmegaBaryon": OmegaBaryon,
            "HubbleParam": HubbleParam,
            "BoxSize": boxsize,
            "Redshift": redshift,
            "SNR": snr,
            "Noise_random_distr": "normal"
        }

        spec_file = SpectraCustomHDF5(out_base_path + f"spectra{i}.hdf5")
        spec_file.create_file(metadata_file, wavelengths, noisy_spectra)



def main():
    suite_to_use = "L25n256_suite"
    out_path = "/vera/ptmp/gc/jerbo/training_data/"
    snr = 100

    list_of_sample_points = []
    for i in range(50):
        list_of_sample_points.append(f"/vera/ptmp/gc/jerbo/{suite_to_use}/gridpoint{i}/")

    np.random.seed(123)
    np.random.shuffle(list_of_sample_points)

    train_split = 0.7

    n_train = int(len(list_of_sample_points)*train_split)
    n_eval = int(len(list_of_sample_points)*(1-train_split)/2)

    train_sample_points = list_of_sample_points[:n_train]
    eval_sample_points = list_of_sample_points[n_train : n_train + n_eval]
    test_sample_points = list_of_sample_points[n_train + n_eval:]
    
    out_path_train = out_path + f"{suite_to_use}_snr_{snr}/train_datasets/"
    out_path_eval = out_path + f"{suite_to_use}_snr_{snr}/eval_datasets/"
    out_path_test = out_path + f"{suite_to_use}_snr_{snr}/test_datasets/"
    
    make_dataset_spectra(train_sample_points, out_path_train, snr)
    make_dataset_spectra(eval_sample_points, out_path_eval, snr)
    make_dataset_spectra(test_sample_points, out_path_test, snr)


if __name__ == "__main__":
    main()