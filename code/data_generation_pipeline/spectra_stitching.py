import temet
import numpy as np
from typing import Callable
import os
import h5py
from temet.spectra.util import create_wavelength_grid
import math


def round_half_up(x):
    return math.floor(x + 0.5)


def correction_function(x):
    return x


def find_closest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def get_spectra_info(path, redshifts_to_use, Line_wavelength_restframe):

    available_spectra_info = {}
    for z in redshifts_to_use:

        # get dz of simulated box
        sim = temet.sim(path, redshift=z)
        dz = sim.dz
        sim_z = sim.redshift

        # get spectra file paths
        gp_name = path.split("/")[-1]
        path_spec_here = os.path.join(path, "data.files", "spectra", f"spectra_{gp_name}_z{z:.1f}_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5")

        # get spectra width in pixels
        left_edge = (sim_z+1)*Line_wavelength_restframe
        right_edge = (sim_z+dz+1)*Line_wavelength_restframe

        with h5py.File(path_spec_here, "r") as f:     
            wavelengths = f["wave"][:]
            n_spectra_available = f["tau_HI_1215"].shape[0]

        assert wavelengths.min() <= left_edge and wavelengths.max() >= right_edge

        idx_left = find_closest_index(wavelengths, left_edge)
        idx_right = find_closest_index(wavelengths, right_edge)

        n_pixels = idx_right - idx_left + 1 # +1 for inclusive right edge

        available_spectra_info[z] = {
            "sim_z": sim_z,
            "dz": dz,
            "path": path_spec_here,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "idx_left": idx_left,
            "n_pixels": n_pixels,
            "n_spectra_available": n_spectra_available,
        }

    return available_spectra_info


def get_interval_boundaries(spectra_info, wavelength_range):

    wl_min, wl_max = wavelength_range
    redshifts = sorted(spectra_info.keys())

    # keep only redshifts whose interval actually overlaps wavelength_range
    used_redshifts = [
        z for z in redshifts
        if spectra_info[z]["right_edge"] > wl_min
        and spectra_info[z]["left_edge"] < wl_max
    ]

    if not used_redshifts:
        raise ValueError("No available spectra overlap the given wavelength_range.")

    boundaries = [wl_min]

    for z_prev, z_next in zip(used_redshifts[:-1], used_redshifts[1:]):
        left_edge_prev = spectra_info[z_prev]["left_edge"]
        left_edge_next = spectra_info[z_next]["left_edge"]
        midpoint = 0.5 * (left_edge_prev + left_edge_next)
        # clip in case something unexpected pushes the midpoint outside range
        midpoint = min(max(midpoint, wl_min), wl_max)
        boundaries.append(midpoint)

    boundaries.append(wl_max)

    return used_redshifts, boundaries


def compute_repeat_counts(spectra_info, used_redshifts, boundary_indices):

    current_pos = boundary_indices[0]
    assert current_pos == 0, "First boundary index must be 0."

    n_redshifts = len(used_redshifts)

    for i, z in enumerate(used_redshifts):
        interval_width = boundary_indices[i + 1] - current_pos
        L = spectra_info[z]["n_pixels"]

        n_fit = interval_width / L

        is_last = (i == n_redshifts - 1)
        n_repeats = math.ceil(n_fit) if is_last else round_half_up(n_fit)
        n_repeats = max(n_repeats, 1)  # guard against degenerate zero-width intervals

        spectra_info[z]["start_idx"] = current_pos
        spectra_info[z]["n_repeats"] = n_repeats

        current_pos += n_repeats * L

    final_end_idx = current_pos
    return spectra_info, final_end_idx


def fill_master_tau(spectra_info, used_redshifts, tau_master_1d, n_spectra_to_generate,
                     spectra_correction_function, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_pixels_master = tau_master_1d.shape[0]
    tau_master = np.tile(tau_master_1d, (n_spectra_to_generate, 1)).astype(float)

    # record some values for plotting
    usage_records = [[] for _ in range(n_spectra_to_generate)]

    for z in used_redshifts:
        info = spectra_info[z]
        idx_left = info["idx_left"]
        start_idx = info["start_idx"]
        n_repeats = info["n_repeats"]
        n_available = info["n_spectra_available"]
        L = info["n_pixels"]  # still used to advance pos between repeats

        with h5py.File(info["path"], "r") as f:
            tau_dset = f["tau_HI_1215"]
            n_source_pixels = tau_dset.shape[1]

            for out_i in range(n_spectra_to_generate):
                chosen_indices = rng.choice(n_available, size=n_repeats, replace=False)

                pos = start_idx
                for spec_idx in chosen_indices:
                    full_segment = tau_dset[int(spec_idx), :]
                    full_segment = spectra_correction_function(full_segment)

                    # shift so that idx_left in the source aligns with pos in the master grid
                    shift = pos - idx_left

                    # clip to valid overlap between shifted source and master grid
                    src_start = max(0, -shift)
                    src_end = min(n_source_pixels, n_pixels_master - shift)

                    if src_end > src_start:
                        dst_start = src_start + shift
                        dst_end = src_end + shift
                        tau_master[out_i, dst_start:dst_end] += full_segment[src_start:src_end]

                    usage_records[out_i].append({
                        "redshift": z,
                        "sim_redshift": info["sim_z"],
                        "redshift_width": info["dz"],
                        "spec_idx": int(spec_idx),
                        "left_pos": int(pos),
                        "spec_path": info["path"],
                    })

                    pos += L

    return tau_master, usage_records


def create_forest_spectra(
        path: str,
        wavelength_range: tuple[float, float],
        redshifts_to_use: list[float],
        spectra_correction_function: Callable,
        n_spectra_to_generate: int,
        Line_wavelength_restframe=1215.67
    ):

    # get the necessary iwnfo about the spectra (dz, path, number of pixels and number of available spectra)
    spectra_info = get_spectra_info(path, redshifts_to_use, Line_wavelength_restframe)

    # make the master grid
    wave_master, _, tau_master = create_wavelength_grid(instrument="SDSS-BOSS")

    # truncate master grid for left wavelength range
    wl_min, wl_max = wavelength_range
    idx_start = find_closest_index(wave_master, wl_min)
    wave_master, tau_master  = wave_master[idx_start:], tau_master[idx_start:]

    #print(wave_master)
    #print(tau_master)

    # get the boundaries in wavelength
    used_redshifts, boundaries = get_interval_boundaries(spectra_info, wavelength_range)

    #print(used_redshifts)
    #print(boundaries)

    # get the boundary indices
    boundary_indices = [find_closest_index(wave_master, b) for b in boundaries]
    assert boundary_indices[0] == 0, (f"Expected first boundary index to be 0 after left truncation, got {boundary_indices[0]} instead.")

    #print(boundary_indices)

    spectra_info, final_end_idx = compute_repeat_counts(spectra_info, used_redshifts, boundary_indices)
    
    #print(tau_master)

    tau_master, usage_records = fill_master_tau(spectra_info, used_redshifts, tau_master, n_spectra_to_generate, spectra_correction_function, rng=None)

    for j in range(len(usage_records)):
        for i, x in enumerate(usage_records[j]):
            #print(x)
            usage_records[j][i]["left_wavelength"] = wave_master[x["left_pos"]]

    #print(tau_master)

    idx_end = find_closest_index(wave_master, wl_max)
    wave_master, tau_master = wave_master[: idx_end + 1], tau_master[:, : idx_end + 1]

    flux_master = np.exp(-tau_master)

    #print(spectra_info)

    return wave_master, flux_master, usage_records
        

if __name__ == "__main__":

    test_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/reference"
    test_range = (3800, 4500)
    n_specs = 1
    redshifts_to_use = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

    wave_master, flux_master, usage_records = create_forest_spectra(test_path, test_range, redshifts_to_use, correction_function, n_specs)

    # TODO: write function that saves the long spectra to file so I dont have to recompute them