import os
import numpy as np
import hashlib
import json
from spectra_stitching import load_forest_spectra, forest_spectra_path
from compressed_spectra import load_spectra_cache

# Version of the SDSS filtering in load_sdss_sigma_F. It is part of the hash of the filtered cache, so caches
# made by older code are never loaded. Increase it with every change that alters the filtered SDSS spectra.
SDSS_FILTER_VERSION = 1


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


def get_cosmo_parameters(basepath):
    """ Returns the cosmological parameters for one simulation

    Args:
        basepath (string): base directory of the simulation

    Returns:
        Omega0 (float): Energydensity of matter in the simulation
        OmegaBaryon (float): Energydensity of Baryons in the simulation
        OmegaLambda (float): Energydensity of Dark Energy in the simulation
        HubbleParam (float): h in the simulation
    """
    path = os.path.join(basepath, "output", "txt-files", "parameters-usedvalues")
    Omega0 = None
    OmegaLambda = None
    HubbleParam = None
    OmegaBaryon = None
    
    with open(path, "r") as f:
        for line in f:
            if "Omega0" in line:
                Omega0 = float(line.split()[-1])
            if "OmegaBaryon" in line:
                OmegaBaryon = float(line.split()[-1])
            if "OmegaLambda" in line:
                OmegaLambda = float(line.split()[-1])
            if "HubbleParam" in line:
                HubbleParam = float(line.split()[-1])
    
    return Omega0, OmegaBaryon, OmegaLambda, HubbleParam


def find_closest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def hash_args(unfiltered_cache_path, min_wavelength, max_wavelength, target_length, n_valid_pixels_min, max_sigma_f, n_chars=16):

    args = {
        "unfiltered_cache_path": str(unfiltered_cache_path), # TODO: not sure if i can put str into the sha256?
        "wavelength_range": [float(min_wavelength), float(max_wavelength)],
        "target_length": int(target_length),
        "n_valid_pixels_min": int(n_valid_pixels_min),
        "max_sigma_f": float(max_sigma_f),
        "sdss_filter_version": SDSS_FILTER_VERSION,
    }

    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:n_chars]


def make_outfile_path(box_path, outfile_path_base):

    hash_name = hashlib.sha256(box_path.encode()).hexdigest()[:6]

    path = os.path.join(outfile_path_base, hash_name)

    # the mapping lives next to the files it describes, in the dataset folder
    with open(os.path.join(outfile_path_base, "hash_mapping.txt"), "a") as f:
        f.write(f"{box_path} -> {hash_name}\n")

    return path


def load_sdss_sigma_F(unfiltered_cache_file, min_wavelength, max_wavelength, target_length, n_valid_pixels_min=100, max_sigma_f=0.5):

    # the filtered cache is written into the same folder as the unfiltered one
    cache_dir = os.path.dirname(unfiltered_cache_file)

    args_hash = hash_args(unfiltered_cache_file, min_wavelength, max_wavelength, target_length, n_valid_pixels_min, max_sigma_f)

    print(f"Loading SDSS spectra from {unfiltered_cache_file} and saving them under hash {args_hash}")

    filtered_cache_path = os.path.join(cache_dir, f"spectra_filtered_{args_hash}.npz")

    if os.path.isfile(filtered_cache_path):
        with np.load(filtered_cache_path) as npz:
            data_sdss = {
                "wavelength": npz["wavelength"],
                "mask": npz["mask"],
                "sigma_F": npz["sigma_F"]
                }

            return data_sdss["wavelength"], data_sdss["sigma_F"], data_sdss["mask"], args_hash

    data_sdss = load_spectra_cache(unfiltered_cache_file)

    ####################### Filter SDSS spectra for Quasar redshift #######################
    min_req_z = max_wavelength/1215.67 - 1

    mask = (data_sdss["redshift"] >= min_req_z) & (data_sdss["valid_spectrum"] == 1)

    wave_sdss = data_sdss["wavelength"]
    flux_sdss = data_sdss["flux"][mask, :]
    sigma_F_sdss = data_sdss["sigma_F"][mask, :]
    mask_sdss = data_sdss["mask"][mask, :]

    ####################### truncate SDSS spectra to correct wavelength interval #######################
    i0 = find_closest_index(wave_sdss, min_wavelength)
    i1 = find_closest_index(wave_sdss, max_wavelength)

    diff = abs(wave_sdss[i0:i1].shape[0] - target_length)

    # TODO: fix the root cause of this eventually instead of this ugly work around
    if 1 >= diff > 0:
        i1 += diff
    if diff > 1:
        raise ValueError(f"Length mismatch between SDSS and simulation spectra: {wave_sdss[i0:i1].shape[0]} vs {target_length}")

    wave_sdss = wave_sdss[i0:i1]
    flux_sdss = flux_sdss[:, i0:i1]
    sigma_F_sdss = sigma_F_sdss[:, i0:i1]
    mask_sdss = mask_sdss[:, i0:i1]

    ####################### Enforce minimum of 100 valid pixels #######################

    mask_valid_pixels = np.sum(mask_sdss, axis=1) >= n_valid_pixels_min

    wave_sdss = wave_sdss
    flux_sdss = flux_sdss[mask_valid_pixels, :]
    sigma_F_sdss = sigma_F_sdss[mask_valid_pixels, :]
    mask_sdss = mask_sdss[mask_valid_pixels, :]

    ####################### Sigma Flux filter #######################

    mask_valid_sigma_F_spectra = np.nanmean(sigma_F_sdss, axis=1) <= max_sigma_f

    print(np.nanmean(sigma_F_sdss, axis=1).shape, mask_valid_sigma_F_spectra.shape)

    flux_sdss = flux_sdss[mask_valid_sigma_F_spectra, :]
    sigma_F_sdss = sigma_F_sdss[mask_valid_sigma_F_spectra, :]
    mask_sdss = mask_sdss[mask_valid_sigma_F_spectra, :]

    np.savez_compressed(
        filtered_cache_path,
        wavelength=wave_sdss,
        flux=flux_sdss,
        mask=mask_sdss,
        sigma_F=sigma_F_sdss,
    )

    return wave_sdss, sigma_F_sdss, mask_sdss, args_hash


def add_noise_to_spectra(flux_sim, noise_type, snr, sigma_F_sdss, sample_idx, rng):

    if noise_type == "constant":
        sigma = 1/snr

        noisy_flux_sim = flux_sim + rng.normal(0.0, sigma, size=flux_sim.shape)

    elif noise_type == "realistic":

        sigma_F_appl = sigma_F_sdss[sample_idx]

        noisy_flux_sim = flux_sim + rng.normal(0.0, sigma_F_appl)

    else:
        raise ValueError(f"Unknown noise_type {noise_type!r}, expected 'constant' or 'realistic'")

    return noisy_flux_sim


def add_masking_to_spectra(flux_sim, mask_sdss, sample_idx):

    mask_appl = mask_sdss[sample_idx]

    # masked pixels hold NaN in every setting (with realistic noise they already do, because sigma_F is NaN there)
    return np.where(mask_appl, flux_sim, np.nan), mask_appl
    

def apply_noise_and_mask(gridpoint_path, outfile_path_base, splits, min_w, max_w, long_specs_hash, noise, masking, noise_type=None, snr=None, spectra_cache_file=None, min_valid_pixels=None, rng=None):

    outfile_path = make_outfile_path(gridpoint_path, outfile_path_base)

    if rng is None:
        rng = np.random.default_rng()

    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gridpoint_path)
    
    metadata_file = {
        "Omega0": Omega0,
        "OmegaLambda": OmegaLambda,
        "OmegaBaryon": OmegaBaryon,
        "HubbleParam": HubbleParam
    }

    for split in splits:
        # Load the spectra that need noise and/or masking
        sim_spec_path = forest_spectra_path(gridpoint_path, long_specs_hash, split)
        wave_sim, flux_sim, _ = load_forest_spectra(sim_spec_path)

        # same rule as in the stitching: from the pixel closest to min_w to the pixel closest to max_w
        i_min, i_max = find_closest_index(wave_sim, min_w), find_closest_index(wave_sim, max_w)
        wave_sim, flux_sim = wave_sim[i_min : i_max + 1], flux_sim[:, i_min : i_max + 1]

        if (noise and noise_type=="realistic") or masking: # load sdss spectra
            # set number of min valid pixels
            if masking:
                assert min_valid_pixels <= wave_sim.shape[0], f"The minimal number of valid pixels is set to {min_valid_pixels}, but the Spectra only have {wave_sim.shape[0]} Pixels"
                n_valid_pixels = min_valid_pixels
            else:
                n_valid_pixels = wave_sim.shape[0]

            # set snr filter
            if noise:
                max_sigma_f = 1/snr
            else:
                max_sigma_f = np.inf

            # get noise and masking from sdss spectra
            _, sigma_F_sdss, mask_sdss, sdss_specs_hash = load_sdss_sigma_F(spectra_cache_file, wave_sim[0], wave_sim[-1], wave_sim.shape[0], n_valid_pixels_min=n_valid_pixels, max_sigma_f=max_sigma_f)

            if mask_sdss.shape[0] == 0:
                raise ValueError(f"No SDSS spectrum passes the filters (min. {n_valid_pixels} valid pixels, "
                                 f"mean sigma_F <= {max_sigma_f}) in {wave_sim[0]:.1f}-{wave_sim[-1]:.1f} A.")

            # draw the SDSS spectra once, so noise and mask of each sim spectrum come from the same quasar
            n_sdss, n_sim = mask_sdss.shape[0], flux_sim.shape[0]
            replace = n_sdss < n_sim  # only reuse SDSS spectra if there are not enough of them
            sample_idx = rng.choice(n_sdss, size=n_sim, replace=replace)

        else:
            # no infromation from sdss data used
            sigma_F_sdss, mask_sdss, sdss_specs_hash, sample_idx = None, None, None, None

        if noise:
            flux_sim = add_noise_to_spectra(flux_sim, noise_type, snr, sigma_F_sdss, sample_idx, rng)

        if masking:
            flux_sim, mask_appl = add_masking_to_spectra(flux_sim, mask_sdss, sample_idx)
        else:
            mask_appl = np.full_like(flux_sim, True, dtype=bool)

        # check for unmasked nans (pixels flagged valid but with non-finite flux)
        unmasked_nans = ~np.isfinite(flux_sim) & mask_appl
        if np.any(unmasked_nans):
            n_bad_spectra = int(np.sum(np.any(unmasked_nans, axis=1)))
            raise ValueError(
                f"[{split}] {int(np.sum(unmasked_nans))} unmasked non-finite flux pixels "
                f"in {n_bad_spectra}/{flux_sim.shape[0]} spectra"
            )

        spec_file = SpectraCustomHDF5(outfile_path + f"_{split}.hdf5")
        spec_file.create_file(metadata_file, wave_sim, flux_sim, mask_appl)

    return sdss_specs_hash


def main():
    test_gp = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/gridpoint0"
    splits = {"train": ((0.02, 0.68), 7000),
        "eval": ((0.72, 0.83), 1500),
        "test": ((0.87, 0.98), 1500),}
    min_w = 3900
    max_w = 4300
    noise = True
    masking = True
    noise_type = "realistic"
    snr = 10
    spectra_cache_file = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/spectra_cache.npz"
    min_valid_pixels = 100
    long_specs_hash = "fa583d77a369688b"
    outfile_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/ML_training_datasets/gridpoint0"

    rng = np.random.default_rng(123)

    apply_noise_and_mask(test_gp, outfile_path, splits, min_w, max_w, long_specs_hash, noise, masking, noise_type, snr, spectra_cache_file, min_valid_pixels, rng=rng)


if __name__ == "__main__":
    main()