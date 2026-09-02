import h5py


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

    
    def create_file(self, meta_data_dict, wavelengths, spectra, mask=None):
        """
        Creates an hdf5 file storing the metadata and wavelengths and fluxes of spectra 

        Args:
            meta_data_dict (dict): dict containing the metadata
            wavelengths (np.array): array with the wavelengths of the spectrum
            spectra (np.array): the fluxes of spectra (2d np.array if multiple spectra)
            mask (np.array, optional): boolean mask indicating which pixels are valid
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
            if mask is not None:
                data.create_dataset("mask", data=mask)


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


    def get_all_spectra_with_mask(self):
        """
        Returns wavelengths, all spectra and the mask from file

        Returns:
            wavelengths (np.array): wavelengths of the spectrum
            fluxes (np.array): array containing the fluxes of the spectra
            mask (np.array): boolean mask indicating which pixels are valid
        """
        with h5py.File(self.file_name, "r") as f:
            fluxes = f["data/flux"][:]
            wavelengths = f["data/wavelengths"][:]

            try:
                mask = f["data/mask"][:]
            except KeyError:
                mask = None

            return wavelengths, fluxes, mask
        

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
    ...