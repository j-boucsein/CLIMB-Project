import temet
import numpy as np
import os
import json
import h5py
import hashlib
from temet.spectra.util import create_wavelength_grid
import math


def round_half_up(x):
    return math.floor(x + 0.5)


def find_closest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def get_spectra_info(path, redshifts_to_use, Line_wavelength_restframe):
    """ Get the necessary info about the spectra (dz, path, number of pixels and number of available spectra) for each redshift in redshifts_to_use.
    
    Args:
        path (str): Path to the simulated box.
        redshifts_to_use (list): List of redshifts to use.
        Line_wavelength_restframe (float): Rest-frame wavelength of the line to use
    
    Returns:
        dict: Dictionary with redshift as key and a dictionary with the following keys:
            - sim_z: redshift of the simulated box
            - dz: redshift width of the simulated box
            - path: path to the spectra file
            - left_edge: left edge of the spectra in wavelength
            - right_edge: right edge of the spectra in wavelength
            - idx_left: index of the left edge in the spectra
            - n_pixels: number of pixels in the spectrum
            - n_spectra_available: number of available spectra in the file
    """

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

    n_available = {info["n_spectra_available"] for info in available_spectra_info.values()}
    assert len(n_available) == 1, (
        f"All redshifts need the same sightline grid so that a region is the same slab at every z, got {n_available}.")

    return available_spectra_info


def get_interval_boundaries(spectra_info, wavelength_range):
    """ Get the boundaries of the intervals in wavelength that correspond to each redshift, given a wavelength_range. Each wavelength is assigned to the snapshot whose own redshift is closest. 
    
    Args:
        spectra_info (dict): Dictionary with redshift as key and a dictionary with the following keys:
            - left_edge: left edge of the spectra in wavelength
            - right_edge: right edge of the spectra in wavelength
        wavelength_range (tuple): Tuple of (wl_min, wl_max) in Angstroms
        
    Returns:
        used_redshifts (list): List of redshifts that are used in the given wavelength_range.
        boundaries (list): List of boundaries in wavelength that correspond to each redshift,
            including the left and right edges of the wavelength_range.
    """

    wl_min, wl_max = wavelength_range
    redshifts = sorted(spectra_info.keys(), key=lambda z: spectra_info[z]["left_edge"])

    first_left = spectra_info[redshifts[0]]["left_edge"]
    last_right = spectra_info[redshifts[-1]]["right_edge"]
    if wl_max <= first_left or wl_min >= last_right:
        raise ValueError("No available spectra overlap the given wavelength_range.")

    # nearest-snapshot cells in wavelength, from all available redshifts
    anchors = [spectra_info[z]["left_edge"] for z in redshifts]
    midpoints = [0.5 * (a + b) for a, b in zip(anchors[:-1], anchors[1:])]
    cell_lo = [-np.inf] + midpoints
    cell_hi = midpoints + [np.inf]

    # keep the snapshots whose cell overlaps wavelength_range (always a contiguous run)
    used = [i for i in range(len(redshifts)) if cell_hi[i] > wl_min and cell_lo[i] < wl_max]
    used_redshifts = [redshifts[i] for i in used]

    # inner boundaries lie strictly inside (wl_min, wl_max) by construction, so no clipping is needed
    boundaries = [wl_min] + [cell_hi[i] for i in used[:-1]] + [wl_max]

    return used_redshifts, boundaries


def compute_repeat_counts(spectra_info, used_redshifts, boundary_indices):
    """ Compute how many times each redshift's spectra need to be repeated to fill the master grid, and the starting index for each redshift in the master grid. The last redshift is allowed to overshoot the right edge of the master grid.
    
    Args:
        spectra_info (dict): Dictionary with redshift as key and a dictionary with the following keys:
            - n_pixels: number of pixels in the spectrum
        used_redshifts (list): List of redshifts that are used in the given wavelength_range.
        boundary_indices (list): List of boundaries in wavelength that correspond to each redshift, including the left and right edges of the wavelength_range, in terms of indices in the master grid
    
    Returns:
        spectra_info (dict): Updated dictionary with redshift as key and a dictionary with the following keys:
            - n_repeats: number of times the spectra need to be repeated to fill the master grid
            - start_idx: starting index for each redshift in the master grid
        final_end_idx (int): The final end index of the master grid after filling with all redshifts' spectra. This may overshoot the right edge of the master grid for the last redshift.
    """

    current_pos = boundary_indices[0]

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


def fill_master_tau(spectra_info, used_redshifts, tau_master_1d, n_spectra_to_generate, region=(0.0, 1.0), rng=None):
    """ Fill the master tau grid with spectra from the given redshifts, repeating each redshift's spectra as needed to fill the master grid. The region parameter specifies the fraction of the available spectra to use for each redshift, where 0.0 is the left edge and 1.0 is the right edge of the available spectra. The same region is used for all redshifts, so that a given region corresponds to a slab of the box at every redshift.
    
    Args:
        spectra_info (dict): Dictionary with redshift as key and a dictionary with the following keys:
            - idx_left: index of the left edge in the spectra
            - n_repeats: number of times the spectra need to be repeated to fill the master grid
            - start_idx: starting index for each redshift in the master grid
            - n_pixels: number of pixels in the spectrum
            - n_spectra_available: number of available spectra
            - path: path to the spectra file
            - sim_z: redshift of the simulated box
            - dz: redshift width of the simulated box
        used_redshifts (list): List of redshifts that are used in the given wavelength_range.
        tau_master_1d (np.ndarray): 1D array of the master tau grid to fill with spectra.
        n_spectra_to_generate (int): Number of spectra to generate.
        region (tuple): Tuple of (left, right) specifying the fraction of the available spectra to use for each redshift, where 0.0 is the left edge and 1.0 is the right edge of the available spectra.
        rng (np.random.Generator): Random number generator to use for selecting spectra. If None, a new generator is created.
        
    Returns:
        tau_master (np.ndarray): 2D array of the master tau grid filled with spectra, with shape (n_spectra_to_generate, n_pixels_master).
        usage_records (list): List of dictionaries recording which spectra were used for each generated spectrum, with keys:
            - redshift: ideal redshift of the used spectrum
            - sim_redshift: actual redshift of the simulated box
            - redshift_width: redshift width of the simulated box
            - spec_idx: index of the used spectrum in the available spectra
            - left_pos: index of the left edge of the used spectrum in the master grid
            - spec_path: path to the used spectrum file
    """
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
        L = info["n_pixels"]

        # only draw from the sightlines inside region (a slab of the box)
        region_indices = np.arange(round(region[0] * n_available), round(region[1] * n_available))

        with h5py.File(info["path"], "r") as f:
            tau_dset = f["tau_HI_1215"]
            n_source_pixels = tau_dset.shape[1]

            for out_i in range(n_spectra_to_generate):
                chosen_indices = rng.choice(region_indices, size=n_repeats, replace=False)

                pos = start_idx
                for spec_idx in chosen_indices:
                    full_segment = tau_dset[int(spec_idx), :]

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


def save_forest_spectra(save_path, wave_master, flux_master, usage_records=None, attrs=None):
    """ Save the master wavelength and flux arrays to an HDF5 file, along with optional usage records and attributes.
    
    Args:
        save_path (str): Path to save the HDF5 file.
        wave_master (np.ndarray): 1D array of the master wavelength grid.
        flux_master (np.ndarray): 2D array of the master flux grid, with shape (n_spectra, n_pixels).
        usage_records (list): Optional list of dictionaries recording which spectra were used for each generated spectrum.
        attrs (dict): Optional dictionary of attributes to save in the HDF5 file.
    
    Returns:
        save_path (str): Path the file was saved to
    """

    flux_master = np.atleast_2d(flux_master)
    assert flux_master.shape[1] == wave_master.shape[0], (
        f"flux has {flux_master.shape[1]} pixels but wavelength grid has {wave_master.shape[0]}.")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("wave", data=np.asarray(wave_master, dtype=np.float64))
        f.create_dataset("flux", data=flux_master.astype(np.float32),
                         chunks=(min(64, flux_master.shape[0]), flux_master.shape[1]),
                         compression="gzip", compression_opts=4)

        if usage_records is not None:
            records_json = json.dumps(usage_records, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            f.create_dataset("usage_records", data=np.bytes_(records_json))

        for key, value in (attrs or {}).items():
            f.attrs[key] = value

    return save_path


def load_forest_spectra(save_path):
    """ Load forest spectra written by save_forest_spectra. File attributes are not loaded.

    Args:
        save_path (str): Path to the HDF5 file.

    Returns:
        wave_master (np.ndarray): 1D array of the wavelength grid.
        flux_master (np.ndarray): 2D array of the flux, with shape (n_spectra, n_pixels).
        usage_records (list or None): The usage records decoded from JSON, or None if the file has none.
    """

    with h5py.File(save_path, "r") as f:
        wave_master = f["wave"][:]
        flux_master = f["flux"][:]
        usage_records = json.loads(f["usage_records"][()]) if "usage_records" in f else None

    return wave_master, flux_master, usage_records


def forest_spectra_path(path, args_hash, split):
    return os.path.join(path, "lya_forest_spectra", f"forest_spectra_{args_hash}_{split}.hdf5")


def check_splits_disjoint(results):
    """Assert that no sightline is used in more than one split. The same spec_idx at a different redshift
    is the same position in the box, so the redshift is deliberately ignored here."""

    used = {split: {r["spec_idx"] for records in usage_records for r in records}
            for split, (_, _, usage_records) in results.items()}

    names = list(used)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = used[a] & used[b]
            assert not shared, f"Splits {a} and {b} share {len(shared)} sightlines."


def hash_spectra_args(wavelength_range, redshifts_to_use,
                      splits, Line_wavelength_restframe, seed=None, n_chars=16):
    """Hash of everything that influences the generated spectra, used as the file name so that
    cached spectra are only reused when all arguments are unchanged."""

    args = {
        "wavelength_range": [float(w) for w in wavelength_range],
        "redshifts_to_use": sorted(float(z) for z in redshifts_to_use),
        "splits": {split: [[float(f) for f in region], int(n)] for split, (region, n) in splits.items()},
        "Line_wavelength_restframe": float(Line_wavelength_restframe),
        "seed": None if seed is None else int(seed),
    }

    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:n_chars]


def write_log(log_path, message):
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


def create_forest_spectra(
        path: str,
        wavelength_range: tuple[float, float],
        redshifts_to_use: list[float],
        splits: dict,
        Line_wavelength_restframe=1215.67,
        save_spectra = False,
        seed = None,
        log_path = "logs/spectra_stitching.log",
    ):
    """ Create mock Lyman-alpha forest spectra on the SDSS-BOSS wavelength grid by stitching together
    simulated sightlines from the given redshifts, separately for each split.

    The wavelength range is divided between the snapshots (get_interval_boundaries). Each interval is
    filled with boxes laid end to end (compute_repeat_counts), using a randomly drawn sightline per box
    from the split's region of the sightline grid (fill_master_tau). The stitched optical depth is turned
    into flux as exp(-tau) and cut to wavelength_range. If files for exactly these arguments already exist
    for every split, they are loaded instead of being recomputed.

    Args:
        path (str): Path to the simulated box.
        wavelength_range (tuple[float, float]): Tuple of (wl_min, wl_max) in Angstroms.
        redshifts_to_use (list[float]): Redshifts of the snapshots to stitch spectra from.
        splits (dict): Split name as key and a tuple (region, n_spectra) as value, where region is a tuple
            (left, right) of fractions of the sightline grid to draw from, and n_spectra is the number of
            spectra to generate for that split. The regions should not overlap.
        Line_wavelength_restframe (float): Rest-frame wavelength of the line, in Angstroms. Defaults to Lyman-alpha.
        save_spectra (bool): If True, save each split to forest_spectra_path(path, args_hash, split).
        seed (int): Seed for the random number generator.
        log_path (str): Path to the log file. Its directory must exist.

    Returns:
        dict: Split name as key and a tuple (wave, flux, usage_records) as value, where wave is the 1D
            wavelength grid, flux is a 2D array of shape (n_spectra, n_pixels), and usage_records is the
            list from fill_master_tau, with each record also holding the "left_wavelength" of its box.

    Raises:
        AssertionError: If the stitched boxes do not cover the full wavelength range, or if two splits share a sightline.
    """

    # hash the arguments to create a unique file name for the generated spectra
    args_hash = hash_spectra_args(wavelength_range, redshifts_to_use, splits, Line_wavelength_restframe, seed)

    save_paths = {split: forest_spectra_path(path, args_hash, split) for split in splits}

    # check if the spectra already exist for these arguments, and load them if so
    if all(os.path.exists(p) for p in save_paths.values()):
        write_log(log_path, f"Forest spectra for these arguments already exist. Loading them instead of recomputing.")
        return {split: load_forest_spectra(p) for split, p in save_paths.items()}

    # get the necessary info about the spectra (dz, path, number of pixels and number of available spectra)
    spectra_info = get_spectra_info(path, redshifts_to_use, Line_wavelength_restframe)

    # make the master grid
    wave_master, _, tau_master = create_wavelength_grid(instrument="SDSS-BOSS")

    # truncate master grid for left wavelength range
    wl_min, wl_max = wavelength_range
    idx_start = find_closest_index(wave_master, wl_min)
    wave_master, tau_master  = wave_master[idx_start:], tau_master[idx_start:]

    # get the boundaries in wavelength
    used_redshifts, boundaries = get_interval_boundaries(spectra_info, wavelength_range)
    # get the boundary indices
    boundary_indices = [find_closest_index(wave_master, b) for b in boundaries]
    assert boundary_indices[0] == 0, (f"Expected first boundary index to be 0 after left truncation, got {boundary_indices[0]} instead.")

    spectra_info, final_end_idx = compute_repeat_counts(spectra_info, used_redshifts, boundary_indices)

    idx_end = find_closest_index(wave_master, wl_max)

    assert final_end_idx >= idx_end, (f"End index of computed coverage of spectra {final_end_idx} is less than the end index of the given wavelength interval {idx_end}. This indicates that the spectra stitching did not cover the entire wavelength range.")

    rng = np.random.default_rng(seed)
    results = {}
    for split, (region, n_spectra) in splits.items():
        print(f"Stitching {n_spectra} {split} spectra from region {region} of the sightline grid")
        tau_split, usage_records = fill_master_tau(spectra_info, used_redshifts, tau_master, n_spectra, region=region, rng=rng)

        for j in range(len(usage_records)):
            for i, x in enumerate(usage_records[j]):
                #print(x)
                usage_records[j][i]["left_wavelength"] = wave_master[x["left_pos"]]

        flux_split = np.exp(-tau_split[:, : idx_end + 1]) # +1 for inclusive right edge
        results[split] = (wave_master[: idx_end + 1], flux_split, usage_records)

    check_splits_disjoint(results)

    if save_spectra:
        for split, (wave_split, flux_split, usage_records) in results.items():
            region, n_spectra = splits[split]
            write_log(log_path, f"Done! Saving {n_spectra} {split} forest spectra to {save_paths[split]}")
            save_forest_spectra(save_paths[split], wave_split, flux_split, usage_records,
                                attrs={"sim_path": path, "wl_min": wl_min, "wl_max": wl_max,
                                       "args_hash": args_hash,
                                       "split": split,
                                       "region": np.asarray(region, dtype=float),
                                       "n_spectra": n_spectra,
                                       "seed": -1 if seed is None else int(seed),
                                       "redshifts_to_use": np.asarray(redshifts_to_use, dtype=float)})

    return results
        

if __name__ == "__main__":
    test_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/gridpoint3"
    test_range = (3600, 3950)
    test_splits = {'training': ((0.02, 0.68), 100), 'evaluation': ((0.72, 0.83), 100), 'testing': ((0.87, 0.98), 100)}
    redshifts_to_use = [2.0]

    results = create_forest_spectra(test_path, test_range, redshifts_to_use, test_splits, save_spectra=False, seed=123)
