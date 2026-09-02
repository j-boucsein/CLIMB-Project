import os
import numpy as np
from spectra_stitching import load_forest_spectra, hash_spectra_args
from compressed_spectra import load_spectra_cache

# This is not super pretty, but I think this is the best way to import stuff from ..util?
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

from util.sim_data_helpers import get_cosmo_parameters
from util.spectra_helpers import SpectraCustomHDF5


def find_closest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def add_noise_to_spectrum(spec, snr, mask=None, fill_value=np.nan, rng=None):

    if rng is None:
        rng = np.random  # legacy global state, seeded with np.random.seed

    if mask is None:
        valid = np.ones(spec.shape, dtype=bool)
    else:
        valid = mask
        assert valid.shape == spec.shape, f"shape of mask with {valid.shape=} doesnt match shape of spectrum with {spec.shape=}"

    sigma = np.abs(spec[valid] / snr[valid])

    noisy_spec = np.full(spec.shape, fill_value, dtype=float)
    noisy_spec[valid] = spec[valid] + rng.normal(0.0, sigma)
    noisy_spec[valid & (noisy_spec < 0)] = 0

    return noisy_spec


def load_sdss_snr(cache_path, min_wavelength, max_wavelength, target_length, n_valid_pixels_min=100):

    data_sdss = load_spectra_cache(cache_path)

    ####################### Filter SDSS spectra for Quasar redshift #######################
    min_req_z = max_wavelength/1215.67 - 1

    mask = (data_sdss["redshift"] >= min_req_z) & (data_sdss["valid_spectrum"] == 1)

    wave_sdss = data_sdss["wavelength"]
    flux_sdss = data_sdss["flux"][mask, :]
    snr_sdss = data_sdss["snr"][mask, :]
    mask_sdss = data_sdss["mask"][mask, :]

    ####################### truncate SDSS spectra to correct wavelength interval #######################
    i0 = find_closest_index(wave_sdss, min_wavelength)
    i1 = find_closest_index(wave_sdss, max_wavelength)

    diff = abs(wave_sdss[i0:i1].shape[0] - target_length)

    if 1 >= diff > 0:
        i1 += diff
    else:
        raise ValueError(f"Length mismatch between SDSS and simulation spectra: {wave_sdss[i0:i1].shape[0]} vs {target_length}")

    wave_sdss = wave_sdss[i0:i1]
    snr_sdss = snr_sdss[:, i0:i1]
    mask_sdss = mask_sdss[:, i0:i1]

    ####################### Enforce minimum of 100 valid pixels #######################

    mask_valid_pixels = np.sum(mask_sdss, axis=1) >= n_valid_pixels_min

    wave_sdss = wave_sdss
    snr_sdss = snr_sdss[mask_valid_pixels, :]
    mask_sdss = mask_sdss[mask_valid_pixels, :]

    return wave_sdss, snr_sdss, mask_sdss



def make_training_spectra_one_box(gridpoint_path, outfile_path, min_wavelength, max_wavelength, rng=None, total_num_spectra_per_file=10000):

    n_train = int(total_num_spectra_per_file * 0.7)
    n_eval = int(total_num_spectra_per_file * 0.15)

    ####################### Load the saved Forest spectra #######################
    wavelength_range = (min_wavelength, max_wavelength)
    redshifts_to_use = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    args_hash = hash_spectra_args(wavelength_range, redshifts_to_use, None, total_num_spectra_per_file, 1215.67, 123)
    
    sim_spec_path = os.path.join(gridpoint_path, "lya_forest_spectra", f"forest_spectra_{args_hash}.hdf5")

    print(f"Loading Simulated Forest Spectra from {sim_spec_path}")
    wave_sim, flux_sim, _ = load_forest_spectra(sim_spec_path)

    ####################### Load the SDSS spectra #######################

    sdss_cache_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/spectra_cache.npz"

    _, snr_sdss, mask_sdss = load_sdss_snr(sdss_cache_path, wave_sim[0], wave_sim[-1], wave_sim.shape[0])

    ####################### Add noise to simulated Spectra #######################
    if rng is None:
        rng = np.random

    n_sdss, n_sim = snr_sdss.shape[0], flux_sim.shape[0]
    replace = n_sdss < n_sim  # only reuse SDSS spectra if there are not enough of them

    sample_idx = rng.choice(n_sdss, size=n_sim, replace=replace)

    snr_appl = snr_sdss[sample_idx]
    mask_appl = mask_sdss[sample_idx]

    assert snr_appl.shape == mask_appl.shape == flux_sim.shape, \
        f"shapes {snr_appl.shape=} / {mask_appl.shape=} dont match {flux_sim.shape=}"

    noisy_flux_sim = add_noise_to_spectrum(flux_sim, snr_appl, mask_appl, rng=rng)

    print(f"{flux_sim.shape} -> {noisy_flux_sim.shape}, drawn from {n_sdss} SDSS spectra with {replace=}")
    print(f"masked pixel fraction: {1 - mask_appl.mean():.3f}, nan fraction in output: {np.isnan(noisy_flux_sim).mean():.3f}")

    ####################### Split into train, eval and test sets #######################
    n_total = noisy_flux_sim.shape[0]
    perm = rng.permutation(n_total)

    noisy_flux_sim = noisy_flux_sim[perm]
    mask_appl = mask_appl[perm]

    waves_train = waves_eval = waves_test = wave_sim

    fluxes_train, masks_train = noisy_flux_sim[:n_train], mask_appl[:n_train]
    fluxes_eval, masks_eval = noisy_flux_sim[n_train:n_train + n_eval], mask_appl[n_train:n_train + n_eval]
    fluxes_test, masks_test = noisy_flux_sim[n_train + n_eval:], mask_appl[n_train + n_eval:]

    print(f"Training: {waves_train.shape}, {fluxes_train.shape}, {masks_train.shape}")
    print(f"Evaluation: {waves_eval.shape}, {fluxes_eval.shape}, {masks_eval.shape}")
    print(f"Test: {waves_test.shape}, {fluxes_test.shape}, {masks_test.shape}")

    ####################### collect the metadata for the file #######################
    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gridpoint_path)

    metadata_file = {
        "Omega0": Omega0,
        "OmegaLambda": OmegaLambda,
        "OmegaBaryon": OmegaBaryon,
        "HubbleParam": HubbleParam
    }

    outfile_path_train = outfile_path + "_train.hdf5"
    outfile_path_eval = outfile_path + "_eval.hdf5"
    outfile_path_test = outfile_path + "_test.hdf5"

    ####################### create the file and write the data to it #######################
    spec_file_train = SpectraCustomHDF5(outfile_path_train)
    spec_file_eval = SpectraCustomHDF5(outfile_path_eval)
    spec_file_test = SpectraCustomHDF5(outfile_path_test)
    spec_file_train.create_file(metadata_file, waves_train, fluxes_train, masks_train)
    spec_file_eval.create_file(metadata_file, waves_eval, fluxes_eval, masks_eval)
    spec_file_test.create_file(metadata_file, waves_test, fluxes_test, masks_test)


def main():
    base_path_test = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/gridpoint0"
    out_path_test = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/ML_training_data_test/gridpoint0"
    min_wavelength = 3800
    max_wavelength = 4500

    make_training_spectra_one_box(base_path_test, out_path_test, min_wavelength, max_wavelength)

    spec_file_test = SpectraCustomHDF5(out_path_test + "_train.hdf5")
    waves, fluxes, masks = spec_file_test.get_all_spectra_with_mask()

    print(f"Loaded {fluxes.shape[0]} spectra from {out_path_test + '_train.hdf5'} with {waves.shape=}, {fluxes.shape=}, {masks.shape=}")



if __name__ == "__main__":
    main()