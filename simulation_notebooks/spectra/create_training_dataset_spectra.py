import numpy as np
import h5py
from patch_spectra_together_helpers import *


class SpectraHDF5:
    """ 
    Wrapper class for hdf5 files with the following structure:
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
        

def get_data_from_header(output_dir, snapN, param='BoxSize'):
    """ get data from the header of a specified simulation and snapshot
    
    Args:
        output_dir (string): base directory of the simulation
        snapN (int): number of the snapshot
        param (string): parameter of the hdf5 file (the parameter to read)

    Returns:
        np.array : Array containing the specific data
    """

    snapdir = glob.glob(output_dir+f"output/snapdir_*{snapN}")[0]
    snap_files = os.listdir(snapdir)
    file_name = snap_files[0]
    file_path = snapdir+f"/{file_name}"

    with h5py.File(file_path, "r") as f:
        header = f['Header']
        return_param = header.attrs[param]
        
    return return_param


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
    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gridpoint_path+"output/")  # TODO: change the get method to take in gp path

    boxsize = get_data_from_header(gridpoint_path, 0, "BoxSize")*1e-3  # convert to Mpc/h
    redshift = get_data_from_header(gridpoint_path, 0, "Redshift")

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
    spec_file = SpectraHDF5(outfile_path)
    spec_file.create_file(metadata_file, wavelengths, noisy_spectra)


def main():
    suite_to_use = "L25n128_suite_var"
    gps_to_use = [i for i in range(50)]

    for i in gps_to_use:
        print(f"starting gp {i}")
        gp_path = f"/vera/ptmp/gc/jerbo/{suite_to_use}/gridpoint{i}/"
        out_file_path = f"/vera/ptmp/gc/jerbo/training_data/{suite_to_use}/gp{i}_spectra.hdf5"

        n_spectra_to_make = 10000
        snr=0.1
        min_w = 3550
        max_w = 3950
        noise_random_distr = "normal"
        total_spectra_in_file = 10000

        make_training_spectra_one_box(gp_path, out_file_path, n_spectra_to_make, snr, 
                                  min_w, max_w, noise_random_distr=noise_random_distr,
                                  total_num_spectra_per_file=total_spectra_in_file)


if __name__ == "__main__":
    """
    gp_number = 0
    gp_path = f"/vera/ptmp/gc/jerbo/L25n256_suite/gridpoint{gp_number}/"
    out_path = "/vera/ptmp/gc/jerbo/training_data/L25n256_suite/"
    out_file_path = out_path + "spectra_test.hdf5"
    
    n_spectra_to_make = 10000
    total_spectra_in_file = 10000
    snr=0.1
    min_w = 3550
    max_w = 3950
    noise_random_distr = "normal"

    make_training_spectra_one_box(gp_path, out_file_path, n_spectra_to_make, snr, 
                                  min_w, max_w, noise_random_distr=noise_random_distr,
                                  total_num_spectra_per_file=total_spectra_in_file)
    """
    main()
