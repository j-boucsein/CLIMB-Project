import numpy as np
import random

# This is not super pretty, but I think this is the best way to import stuff from ..util?
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
from util.spectra_helpers import patch_spectra_together, add_noise_to_spectrum, SpectraCustomHDF5
from util.sim_data_helpers import get_cosmo_parameters, get_data_from_header


def make_training_spectra_one_box(gridpoint_path, outfile_path, n_spectra, snr,
                                   min_wavelength, max_wavelength,
                                   noise_random_distr="normal", total_num_spectra_per_file=10000):
    """
    Create training data for the Neural Nets. Load the spectra from one gridpoint, patch other spectra to the basespectrum
    to make it longer and truncate it in a range where there are features (the given min_wavelength and max_wavelength)
    then add noise. The noisy and augmented spectra are then saved to a new hdf5 file

    Args:
        gridpoint_path (string): path to one gridpoint
        outfile_path (string): full file path (including name of file) of the output file
        n_spectra (int): number of spectra to create for the new file (max total_num_spectra_per_file)
        snr (float): Signa-to-noise Ratio of the added noise
        min_wavelength (float): minimal wavelength of the created spectra
        max_wavelength (_type_): maximal wavelength of the created spectra
        noise_random_distr (str, optional): type of distribution to create the noise. Defaults to "normal".
        total_num_spectra_per_file (int, optional): total number of spectra in the files created by temet. Defaults to 10000.
    """
    base_spectra = random.sample(range(0, total_num_spectra_per_file), n_spectra)
    
    # make spectra longer by augmenting them
    wavelengths, patched_spectra = patch_spectra_together(gridpoint_path, base_spectra, min_wavelength, max_wavelength, total_num_spectra_per_file=total_num_spectra_per_file)
    
    wavelengths = np.array(wavelengths)
    patched_spectra = np.stack(patched_spectra, axis=0)

    # truncate the spectra to the given range
    mask = (wavelengths > min_wavelength) & (wavelengths < max_wavelength)
    wavelengths = wavelengths[mask]
    patched_spectra = patched_spectra[:, mask]

    # add noise to the spectra
    noisy_spectra = []
    for spec in patched_spectra:
        noisy_spectra.append(add_noise_to_spectrum(spec, snr, distr=noise_random_distr))

    noisy_spectra = np.stack(noisy_spectra, axis=0)
    
    # collect the metadata for the file
    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gridpoint_path)  # TODO: change the get method to take in gp path

    boxsize = get_data_from_header(gridpoint_path, 3, "BoxSize")*1e-3  # convert to Mpc/h
    redshift = get_data_from_header(gridpoint_path, 3, "Redshift")

    metadata_file = {
        "Omega0": Omega0,
        "OmegaLambda": OmegaLambda,
        "OmegaBaryon": OmegaBaryon,
        "HubbleParam": HubbleParam,
        "BoxSize": boxsize,
        "Redshift": redshift,
        "SNR": snr,
        "Noise_random_distr": noise_random_distr
    }

    # create the file and write the data to it
    spec_file = SpectraCustomHDF5(outfile_path)
    spec_file.create_file(metadata_file, wavelengths, noisy_spectra)


def main():
    suite_to_use = "L25n256_suite"
    gps_to_use = [i for i in range(50)]

    for i in gps_to_use:
        print(f"starting gp {i}")
        gp_path = f"/vera/ptmp/gc/jerbo/{suite_to_use}/gridpoint{i}/"
        out_file_path = f"/vera/ptmp/gc/jerbo/training_data/{suite_to_use}_no_noise/gp{i}_spectra.hdf5"

        n_spectra_to_make = 10000
        snr=0
        min_w = 3550
        max_w = 3950
        noise_random_distr = "normal"
        total_spectra_in_file = 10000

        make_training_spectra_one_box(gp_path, out_file_path, n_spectra_to_make, snr, 
                                  min_w, max_w, noise_random_distr=noise_random_distr,
                                  total_num_spectra_per_file=total_spectra_in_file)


if __name__ == "__main__":
    main()
