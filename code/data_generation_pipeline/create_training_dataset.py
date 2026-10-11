import os
import numpy as np
import hashlib
import json
from spectra_stitching import load_forest_spectra, forest_spectra_path, SPLITS
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


def hash_args(unfiltered_cache_path, min_wavelength, max_wavelength, target_length, n_valid_pixels_min, max_sigma_f, n_chars=16):

    args = {
        "unfiltered_cache_path": str(unfiltered_cache_path), # TODO: not sure if i can put str into the sha256?
        "wavelength_range": [float(min_wavelength), float(max_wavelength)],
        "target_length": int(target_length),
        "n_valid_pixels_min": int(n_valid_pixels_min),
        "max_sigma_f": float(max_sigma_f)
    }

    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:n_chars]


def add_noise_to_spectrum(spec, sigma_F, mask=None, fill_value=np.nan, rng=None):

    if rng is None:
        rng = np.random  # legacy global state, seeded with np.random.seed

    if mask is None:
        valid = np.ones(spec.shape, dtype=bool)
    else:
        valid = mask
        assert valid.shape == spec.shape, f"shape of mask with {valid.shape=} doesnt match shape of spectrum with {spec.shape=}"

    noisy_spec = np.full(spec.shape, fill_value, dtype=float)
    noisy_spec[valid] = spec[valid] + rng.normal(0.0, sigma_F[valid])
    # noisy_spec[valid & (noisy_spec < 0)] = 0

    return noisy_spec


def load_sdss_sigma_F(cache_path, min_wavelength, max_wavelength, target_length, n_valid_pixels_min=100, max_sigma_f=0.5):

    unfiltered_cache_path = os.path.join(cache_path, "spectra_cache.npz")

    args_hash = hash_args(cache_path, min_wavelength, max_wavelength, target_length, n_valid_pixels_min, max_sigma_f)

    filtered_cache_path = os.path.join(cache_path, f"spectra_filtered_{args_hash}.npz")

    if os.path.isfile(filtered_cache_path):
        with np.load(filtered_cache_path) as npz:
            data_sdss = {
                "wavelength": npz["wavelength"],
                "mask": npz["mask"],
                "sigma_F": npz["sigma_F"]
                }

            return data_sdss["wavelength"], data_sdss["sigma_F"], data_sdss["mask"]

    data_sdss = load_spectra_cache(unfiltered_cache_path)

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

    if 1 >= diff > 0:
        i1 += diff
    else:
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

    return wave_sdss, sigma_F_sdss, mask_sdss


def make_training_spectra_one_box(gridpoint_path, outfile_path, min_wavelength, max_wavelength, args_hash, max_sigma_f, no_mask=False, no_noise=False, rng=None):

    if rng is None:
        rng = np.random

    ####################### collect the metadata for the file #######################
    Omega0, OmegaBaryon, OmegaLambda, HubbleParam = get_cosmo_parameters(gridpoint_path)

    metadata_file = {
        "Omega0": Omega0,
        "OmegaLambda": OmegaLambda,
        "OmegaBaryon": OmegaBaryon,
        "HubbleParam": HubbleParam
    }

    # train/eval/test are stitched from separate regions of the box (see spectra_stitching.SPLITS),
    # so every split has its own forest spectra file and nothing is re-split here
    for split in SPLITS:
        sim_spec_path = forest_spectra_path(gridpoint_path, args_hash, split)

        print(f"Loading Simulated Forest Spectra from {sim_spec_path}")
        wave_sim, flux_sim, _ = load_forest_spectra(sim_spec_path)

        wave_range_sel = (min_wavelength <= wave_sim) & (max_wavelength >= wave_sim)

        wave_sim, flux_sim = wave_sim[wave_range_sel], flux_sim[:, wave_range_sel]

        ####################### Load the SDSS spectra #######################

        sdss_cache_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/"

        if no_mask:
            n_valid_pixels = wave_sim.shape[0]
        else:
            n_valid_pixels = 100

        _, sigma_F_sdss, mask_sdss = load_sdss_sigma_F(sdss_cache_path, wave_sim[0], wave_sim[-1], wave_sim.shape[0], n_valid_pixels_min=n_valid_pixels, max_sigma_f=max_sigma_f)

        # if no_mask, force the entire mask to be True
        if no_mask:
            mask_sdss = np.full_like(mask_sdss, True, dtype=bool)

        ####################### Add noise to simulated Spectra #######################
        n_sdss, n_sim = sigma_F_sdss.shape[0], flux_sim.shape[0]
        replace = n_sdss < n_sim  # only reuse SDSS spectra if there are not enough of them

        if n_sdss > 0:
            sample_idx = rng.choice(n_sdss, size=n_sim, replace=replace)

            sigma_F_appl = sigma_F_sdss[sample_idx]
            mask_appl = mask_sdss[sample_idx]

            assert sigma_F_appl.shape == mask_appl.shape == flux_sim.shape, \
                    f"shapes {sigma_F_appl.shape=} / {mask_appl.shape=} dont match {flux_sim.shape=}"
        else:
            # sigma_F_appl = np.full(flux_sim.shape, 0.1, dtype=float)
            mask_appl = np.full(flux_sim.shape, True, dtype=bool)

        # if no_noise=False, add noise to the spectrum
        if no_noise:
            noisy_flux_sim = np.full(flux_sim.shape, np.nan, dtype=float)
            noisy_flux_sim[mask_appl] = flux_sim[mask_appl]
        else:
            noisy_flux_sim = add_noise_to_spectrum(flux_sim, sigma_F_appl, mask_appl, rng=rng)

        print(f"{split}: {flux_sim.shape} -> {noisy_flux_sim.shape}, drawn from {n_sdss} SDSS spectra with {replace=}")
        print(f"masked pixel fraction: {1 - mask_appl.mean():.3f}, nan fraction in output: {np.isnan(noisy_flux_sim).mean():.3f}")

        ####################### create the file and write the data to it #######################
        spec_file = SpectraCustomHDF5(outfile_path + f"_{split}.hdf5")
        spec_file.create_file(metadata_file, wave_sim, noisy_flux_sim, mask_appl)


def main():
    min_wavelength = 3800
    max_wavelength = 4500
    max_sigma_f = 0.2
    no_noise = True
    no_mask = True
    hash_spectra_files = "e9f3ace363dbe895"  # region-split version of "e16590fc78e11efb" (3800-4500 A, z = 2.0-3.0)

    traing_set_folder_name = "long_test_no_noise_no_mask_set"

    path_training_set_foler = f"ML_test_datasets/{traing_set_folder_name}"

    os.makedirs(f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/{path_training_set_foler}", exist_ok=True)

    gp_paths = [f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/gridpoint{i}" for i in range(50)]
    out_paths = [f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/{path_training_set_foler}/gridpoint{i}" for i in range(50)]

    out_paths.append(f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/{path_training_set_foler}/reference")
    gp_paths.append("/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/reference")

    for i in range(len(gp_paths)):
        base_path_it = gp_paths[i]
        out_path_it = out_paths[i]

        print(f"Making training dataset for {base_path_it}")

        make_training_spectra_one_box(base_path_it, out_path_it, min_wavelength, max_wavelength,  args_hash=hash_spectra_files, max_sigma_f=max_sigma_f, no_mask=no_mask, no_noise=no_noise)



if __name__ == "__main__":
    main()