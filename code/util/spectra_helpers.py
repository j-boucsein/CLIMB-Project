import h5py
import numpy as np
import temet
import random
from .cosmo_helpers import redshift_wavelength_forward
from .sim_data_helpers import get_cosmo_parameters, get_data_from_header


def load_spectra_from_boxes(gp_paths, spectrumNums, y_param="flux"):
    """ Loads all spectra specified in spectrumNums for all boxes specified in gp_paths

    Args:
        gp_paths (list): List of Gridpoint folder paths to load spectra from.
        spectrumNums (list): List of indices for the spectra.
        y_param (str, optional): name of field to return data from. Defaults to 'flux'.

    Returns:
        wavelengths (list): one np.array containing the wavelengths of the spectra for each box
        y_param (list): len(spectrumNums) number of np.arrays for each box (each is one spectrum)
        params (list): [Omega0, OmegaBaryon, OmegaLambda, HubbleParam] for each box
    """
    
    wavelengths = []
    y_out = []
    params = []
    
    for gp_path in gp_paths:
        gp_name = gp_path.split("/")[-2]
        path = gp_path + f"data.files/spectra/spectra_{gp_name}_z2.0_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"

        with h5py.File(path, "r") as f:     
            # Zugriff auf einzelne Datasets
            wavelengths.append(f["wave"][:])  # Das [:] liest das gesamte Dataset in ein NumPy-Array ein
            y_out.append(f[y_param][:][spectrumNums])

        Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gp_path)

        params.append([Omega0, OmegaBaryon, OmegaLambda, HubbleParam])

    return wavelengths, y_out, params


def load_reference(path, spectrumNs):
    """load spectra from reference simulation

    Args:
        path (string): base path to reference simulation folder 
        spectrumNs (list): List of indices for the spectra.

    Returns:
        wavelengths (list): one np.array containing the wavelengths of the spectra for each box
        fluxes (list): len(spectrumNs) number of np.arrays for each box (each is one spectrum)
        params (list): [Omega0, OmegaBaryon, OmegaLambda, HubbleParam] for each box
    """
    path = path + f"data.files/spectra/spectra_reference_point_z1.5_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"
    wavelengths = None
    fluxes = None
    
    with h5py.File(path, "r") as f:     
        # Zugriff auf einzelne Datasets
        wavelengths = f["wave"][:]  # Das [:] liest das gesamte Dataset in ein NumPy-Array ein
        fluxes = f["flux"][:][spectrumNs]

    outdir_path = path + "output/"
    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(outdir_path)
    params = [Omega0, OmegaBaryon, OmegaLambda, HubbleParam]

    return wavelengths, fluxes, params


def patch_spectra_together(gp_path, spectra_to_augment, min_wavelength, max_wavelength, total_num_spectra_per_file=10000, return_plot_vars=False):
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
        augmentation_spectra_numbers = random.sample(range(0, total_num_spectra_per_file), number_of_augmentation_spectra)
        
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


def add_noise_to_spectrum(spec, snr, distr='uniform'):
    """ adds random noise to a spectrum

    Args:
        spec (np.array): flux values of a given spectrum
        snr (float or np.array): Signal-to-noise ratio of the added noise
        distr (str, optional): type of the random distibution. Defaults to 'uniform'.

    Returns:
        np.array: the spectrum with added noise
    """
    
    if distr == 'uniform':
        n = len(spec)
        noise_array = np.random.rand(n)
        noise_array -= 0.5
        noise_array *= 2*snr

        noisy_spec = noise_array + spec

        # noisy_spec[np.where(noisy_spec>1)] = 1
        noisy_spec[np.where(noisy_spec<0)] = 0

        return noisy_spec
    
    if distr == 'normal':
        n = len(spec)
        mu = 0
        sigma = 1/snr  # as the signal strength is normalized to 1
        noise_array = np.random.normal(mu, sigma, n)

        noisy_spec = noise_array + spec

        # noisy_spec[np.where(noisy_spec>1)] = 1
        noisy_spec[np.where(noisy_spec<0)] = 0

        return noisy_spec
    
    if distr == "realistic":
        assert len(snr) == len(spec), f"shape of noise array with {snr.shape=} doesnt match shape of spectrum with {spec.shape=}"

        mu = 0
        sigma = 1/snr
        noise_array = np.random.normal(mu, sigma)

        noisy_spec = noise_array + spec
        noisy_spec[np.where(noisy_spec<0)] = 0

        return noisy_spec
    
    return spec


class SpectraCustomHDF5:
    """ 
    Wrapper class for hdf5 files that I use to store the finished spectra to train the NNs with.
    The file structure is:
      /metadata
      /data/flux
      /data/wavelengths
    """

    def __init__(self, file_name):
        """
        Args:
            file_name (string): full file path (including name of file)
        """
        self.file_name = file_name

    
    def create_file(self, meta_data_dict, wavelengths, spectra):
        """
        Creates an hdf5 file storing the metadata and wavelengths and fluxes of spectra 

        Args:
            meta_data_dict (dict): dict containing the metadata
            wavelengths (np.array): array with the wavelengths of the spectrum
            spectra (np.array): the fluxes of spectra (2d np.array if multiple spectra)
        """
        with h5py.File(self.file_name, "w") as f:
            
            # -------- write meta data --------------
            meta = f.create_group("metadata")
            for keys in meta_data_dict.keys():
                meta.attrs[keys] = meta_data_dict[keys]
            
            # -------- write data ------------
            data = f.create_group("data")
            data.create_dataset("wavelengths", data=wavelengths)
            data.create_dataset("flux", data=spectra)


    def get_all_spectra(self):
        """
        Returns wavelengths and all spectra from file

        Returns:
            wavelengths (np.array): wavelengths of the spectrum
            fluxes (np.array): array containing the fluxes of the spectra
        """
        with h5py.File(self.file_name, "r") as f:
            fluxes = f["data/flux"][:]
            wavelengths = f["data/wavelengths"][:]

            return wavelengths, fluxes
        

    def get_single_spectrum(self, index):
        """
        Returns a single spectrum from file

        Args:
            index (int): the index of the spectrum

        Returns:
            wavelengths (np.array): wavelengths of the spectrum
            fluxes (np.array): array containing the fluxes of the spectrum
        """
        with h5py.File(self.file_name, "r") as f:
            flux = f["data/flux"][index]
            wavelengths = f["data/wavelengths"][:]

            return wavelengths, flux
    

    def get_header(self):
        """
        Gets the metadata from the file

        Returns:
            dict: metadata in the file
        """
        with h5py.File(self.file_name, "r") as f:
            meta = f["metadata"].attrs
            return {k: meta[k] for k in meta}


if __name__ == "__main__":
    gp_p = "/vera/u/jerbo/my_ptmp/L25n128_suite/gridpoint0/"
    w, y, p = load_spectra_from_boxes([gp_p], [1])

    print(w, y, p)