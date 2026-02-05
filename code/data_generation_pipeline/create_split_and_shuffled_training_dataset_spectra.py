import numpy as np
import random
import temet

# This is not super pretty, but I think this is the best way to import stuff from ..util?
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
from util.spectra_helpers import add_noise_to_spectrum, SpectraCustomHDF5, load_spectra_from_boxes
from util.sim_data_helpers import get_cosmo_parameters, get_data_from_header
from util.cosmo_helpers import redshift_wavelength_forward


def patch_spectra_together(gp_path, spectra_to_augment, min_wavelength, max_wavelength, augmentation_spectra_indices_list, total_num_spectra_per_file=10000, return_plot_vars=False):
    """make long spectra by stiching random spectra to the specified spectra_to_augment

    Args:
        gp_path (string): path to the gridpoint
        spectra_to_augment (list): index numbers of the (base) spectra that should be augmented
        min_wavelength (float): left edge of range that should be filled with random spectra
        max_wavelength (float): right edge of range that should be filled with random spectra
        total_num_spectra_per_file (int, optional): The total number of spectra in the spectra hdf5 file. Defaults to 10000.
        return_plot_vars (bool, optional): if true return additional plot variables. Defaults to False.

    Returns:
        wavelengths (np.array): the wavelengths of the spectra
        augmented_spectra (list): list of np.arrays containing the different augmented spectra. Returns one spectrum for each element in spectra_to_augmented
    """

    Ly_alpha_0 = 1215.67  # Lymann alpha wavelength at rest in Ängstrom

    # load the spectra (actually load wavelength and optical depth)
    wavelengths, taus, params = load_spectra_from_boxes([gp_path], [i for i in range(total_num_spectra_per_file)], y_param='tau_HI_1215')
    wavelengths, taus, params = wavelengths[0], taus[0], params[0]

    # calculate width of spectra
    sim = temet.sim(gp_path, redshift=2.0)
    dz = sim.dz
    z = get_data_from_header(gp_path, 3, param="Redshift")
    z = z + dz
    right_edge = redshift_wavelength_forward(z, Ly_alpha_0)
    left_edge = redshift_wavelength_forward(z - dz, Ly_alpha_0)
    spectrum_width = right_edge - left_edge

    # calculate wavelength bin width - note: the binning is not constant. However it is approximated to be constant
    #                                        because of this, the distance is taken from where the left cut is going
    #                                        to be made
    wavelength_bin_width = wavelengths[np.where(wavelengths > left_edge)][1] - wavelengths[np.where(wavelengths > left_edge)][0]

    index_shift_one_spec_width = round(spectrum_width/wavelength_bin_width)

    # check if min and max wavelengths are inside the single spectrum 
    if left_edge < min_wavelength:
        return None
    
    if right_edge > max_wavelength:
        return None
    
    # check if min and max wavelengths are within the instument range
    if left_edge < wavelengths[0]:
        left_edge = wavelengths[0]
    
    if right_edge > wavelengths[-1]:
        right_edge = wavelengths[-1]
    
    # calculate how many spectra to augment to the left and right
    spectra_to_add_to_left = int((left_edge - min_wavelength)/spectrum_width)+1  # int() truncates floats, so add 1 to make sure the full range is filled with spectra
    spectra_to_add_to_right = int((max_wavelength - right_edge)/spectrum_width)+1 # int() truncates floats, so add 1 to make sure the full range is filled with spectra

    number_of_augmentation_spectra = spectra_to_add_to_left + spectra_to_add_to_right

    plot_infos = []  # for plotting purposes only
    plot_specs = []  # for plotting purposes only
    augmented_spectra = []
    for spectrum_to_augment_number in spectra_to_augment:
        # choose the spectra to augment the main one with
        augmentation_spectra_numbers = random.sample(augmentation_spectra_indices_list, number_of_augmentation_spectra)
        
        augmentation_spectra_left = augmentation_spectra_numbers[:spectra_to_add_to_left]
        if spectra_to_add_to_right == 0:  # due to how slicing works in python I need to have this if condition here (and only for the right)
            augmentation_spectra_right = []
        else:
            augmentation_spectra_right = augmentation_spectra_numbers[-spectra_to_add_to_right:]

        plot_infos.append([*list(reversed(augmentation_spectra_left)), spectrum_to_augment_number, *augmentation_spectra_right])  # for plotting purposes only

        # get the base spectrum that is going to be augmented
        base_spectrum = taus[spectrum_to_augment_number]
        base_spectrum = np.array(base_spectrum)

        tmp_plot_spec = np.exp(-base_spectrum)  # for plotting purposes only
        tmp_plot_mem = []  # for plotting purposes only

        # add the spectra shifted to the left
        for i in range(len(augmentation_spectra_left)):
            cutoff = index_shift_one_spec_width*(i+1)
            shifted_aug_spec = np.concatenate((taus[augmentation_spectra_left[i]][cutoff:], np.zeros(cutoff)))
            base_spectrum = base_spectrum + shifted_aug_spec
            
            tmp_plot_mem.append(np.exp(-shifted_aug_spec))  # for plotting purposes only

        for i in list(reversed(tmp_plot_mem)):  # for plotting purposes only
            plot_specs.append(i)  # for plotting purposes only
        plot_specs.append(tmp_plot_spec)  # for plotting purposes only

        # add the spectra shifted to the right
        for i in range(len(augmentation_spectra_right)):
            cutoff = index_shift_one_spec_width*(i+1)
            shifted_aug_spec = np.concatenate((np.zeros(cutoff), taus[augmentation_spectra_right[i]][:-cutoff]))
            base_spectrum = base_spectrum + shifted_aug_spec
            
            plot_specs.append(np.exp(-shifted_aug_spec))  # for plotting purposes only

        base_spectrum = np.exp(-base_spectrum)  # convert from optical depth to flux

        augmented_spectra.append(base_spectrum)
    
    if return_plot_vars:  # for plotting purposes only
        return wavelengths, augmented_spectra, plot_infos, plot_specs
    else:
        return wavelengths, augmented_spectra


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

    n_train = int(total_num_spectra_per_file * 0.7)
    n_eval = int(total_num_spectra_per_file * 0.15)

    base_spectra = random.sample(range(0, total_num_spectra_per_file), total_num_spectra_per_file)
    # augmentation_spectra_indices_list = random.sample(range(0, total_num_spectra_per_file), total_num_spectra_per_file)
    
    base_spectra_train = base_spectra[:n_train]
    base_spectra_eval = base_spectra[n_train: n_train + n_eval]
    base_spectra_test = base_spectra[n_train + n_eval:]

    # make spectra longer by augmenting them
    wavelengths_train, patched_spectra_train = patch_spectra_together(gridpoint_path, base_spectra_train, min_wavelength, max_wavelength, base_spectra_train, total_num_spectra_per_file=total_num_spectra_per_file)
    wavelengths_eval, patched_spectra_eval = patch_spectra_together(gridpoint_path, base_spectra_eval, min_wavelength, max_wavelength, base_spectra_eval, total_num_spectra_per_file=total_num_spectra_per_file)
    wavelengths_test, patched_spectra_test = patch_spectra_together(gridpoint_path, base_spectra_test, min_wavelength, max_wavelength, base_spectra_test, total_num_spectra_per_file=total_num_spectra_per_file)


    wavelengths_train, wavelengths_eval, wavelengths_test = np.array(wavelengths_train), np.array(wavelengths_eval), np.array(wavelengths_test)
    patched_spectra_train, patched_spectra_eval, patched_spectra_test = np.stack(patched_spectra_train, axis=0), np.stack(patched_spectra_eval, axis=0), np.stack(patched_spectra_test, axis=0)

    # truncate the spectra to the given range
    mask = (wavelengths_train > min_wavelength) & (wavelengths_train < max_wavelength)  # this can be done for the train wavelengths for all, as they are the same anyways
    wavelengths_train, wavelengths_eval, wavelengths_test = wavelengths_train[mask], wavelengths_eval[mask], wavelengths_test[mask]
    patched_spectra_train, patched_spectra_eval, patched_spectra_test = patched_spectra_train[:, mask], patched_spectra_eval[:, mask], patched_spectra_test[:, mask]

    # add noise to the spectra
    noisy_spectra_train = []
    noisy_spectra_eval = []
    noisy_spectra_test = []
    for spec in patched_spectra_train:
        noisy_spectra_train.append(add_noise_to_spectrum(spec, snr, distr=noise_random_distr))
    for spec in patched_spectra_eval:
        noisy_spectra_eval.append(add_noise_to_spectrum(spec, snr, distr=noise_random_distr))
    for spec in patched_spectra_test:
        noisy_spectra_test.append(add_noise_to_spectrum(spec, snr, distr=noise_random_distr))

    noisy_spectra_train, noisy_spectra_eval, noisy_spectra_test = np.stack(noisy_spectra_train, axis=0), np.stack(noisy_spectra_eval, axis=0), np.stack(noisy_spectra_test, axis=0)
    
    # collect the metadata for the file
    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gridpoint_path)

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

    outfile_path_train = outfile_path + "_train.hdf5"
    outfile_path_eval = outfile_path + "_eval.hdf5"
    outfile_path_test = outfile_path + "_test.hdf5"

    # create the file and write the data to it
    spec_file_train = SpectraCustomHDF5(outfile_path_train)
    spec_file_eval = SpectraCustomHDF5(outfile_path_eval)
    spec_file_test = SpectraCustomHDF5(outfile_path_test)
    spec_file_train.create_file(metadata_file, wavelengths_train, noisy_spectra_train)
    spec_file_eval.create_file(metadata_file, wavelengths_eval, noisy_spectra_eval)
    spec_file_test.create_file(metadata_file, wavelengths_test, noisy_spectra_test)


def main():
    suite_to_use = "L25n256_suite"
    gps_to_use = [i for i in range(50)]

    for i in gps_to_use:
        print(f"starting gp {i}")
        gp_path = f"/vera/ptmp/gc/jerbo/{suite_to_use}/gridpoint{i}/"
        out_file_path = f"/vera/ptmp/gc/jerbo/training_data/L25n256snr10_6095_sas/gp{i}_spectra"

        n_spectra_to_make = 10000
        snr=10
        min_w = 3550
        max_w = 3950
        noise_random_distr = "normal"
        total_spectra_in_file = 10000

        make_training_spectra_one_box(gp_path, out_file_path, n_spectra_to_make, snr, 
                                  min_w, max_w, noise_random_distr=noise_random_distr,
                                  total_num_spectra_per_file=total_spectra_in_file)


if __name__ == "__main__":
    main()
