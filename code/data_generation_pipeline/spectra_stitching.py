import temet
import numpy as np
from typing import Callable
import os
import json
import h5py
import hashlib
import inspect
from temet.spectra.util import create_wavelength_grid
import math


# The short spectra lie on an nRaysPerDim x nRaysPerDim grid perpendicular to the line of sight, with x
# as the slow index (temet: meshgrid indexing="ij" + ravel). A contiguous index block is therefore a slab
# in x, and the same block is the same physical region at every redshift. Each split only stitches short
# spectra from its own slab, so no sightline ends up in more than one split. The 0.04 gaps (4 rows,
# 2 cMpc/h in L50) separate neighbouring slabs, including across the periodic boundary (0.98 -> 0.02).
SPLITS = {
    # name: ((start_frac, end_frac) of the sightline grid, number of long spectra to generate)
    "train": ((0.02, 0.68), 7000),
    "eval": ((0.72, 0.83), 1500),
    "test": ((0.87, 0.98), 1500),
}


def round_half_up(x):
    return math.floor(x + 0.5)


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

    n_available = {info["n_spectra_available"] for info in available_spectra_info.values()}
    assert len(n_available) == 1, (
        f"All redshifts need the same sightline grid so that a region is the same slab at every z, got {n_available}.")

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
                     spectra_correction_function, region=(0.0, 1.0), rng=None):
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

        # only draw from the sightlines inside region (a slab of the box, see SPLITS)
        region_indices = np.arange(round(region[0] * n_available), round(region[1] * n_available))

        with h5py.File(info["path"], "r") as f:
            tau_dset = f["tau_HI_1215"]
            n_source_pixels = tau_dset.shape[1]

            for out_i in range(n_spectra_to_generate):
                chosen_indices = rng.choice(region_indices, size=n_repeats, replace=False)

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


def save_forest_spectra(save_path, wave_master, flux_master, usage_records=None, attrs=None):

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


def hash_spectra_args(wavelength_range, redshifts_to_use, spectra_correction_function,
                      splits, Line_wavelength_restframe, seed=None, n_chars=16):
    """Hash of everything that influences the generated spectra, used as the file name so that
    cached spectra are only reused when all arguments are unchanged."""

    if spectra_correction_function is None:
        correction_repr = "None"
    else:
        try:
            correction_repr = inspect.getsource(spectra_correction_function)
        except (TypeError, OSError):
            correction_repr = getattr(spectra_correction_function, "__name__", repr(spectra_correction_function))

    args = {
        "wavelength_range": [float(w) for w in wavelength_range],
        "redshifts_to_use": sorted(float(z) for z in redshifts_to_use),
        "spectra_correction_function": correction_repr,
        "splits": {split: [[float(f) for f in region], int(n)] for split, (region, n) in splits.items()},
        "Line_wavelength_restframe": float(Line_wavelength_restframe),
        "seed": None if seed is None else int(seed),
    }

    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:n_chars]


def create_forest_spectra(
        path: str,
        wavelength_range: tuple[float, float],
        redshifts_to_use: list[float],
        spectra_correction_function: Callable,
        splits: dict,
        Line_wavelength_restframe=1215.67,
        save_spectra = False,
        seed = None,
    ):
    """Stitch long spectra separately for every split in splits (see SPLITS), each one only using the short
    spectra from its own region of the box. Returns {split: (wave, flux, usage_records)}."""

    print("Creating forest spectra for path:", path)

    # hash the arguments before defaulting, so that the identity case is stable
    args_hash = hash_spectra_args(wavelength_range, redshifts_to_use, spectra_correction_function, splits, Line_wavelength_restframe, seed)

    print("Args hash:", args_hash)

    if spectra_correction_function is None:
        spectra_correction_function = lambda x: x
    save_paths = {split: forest_spectra_path(path, args_hash, split) for split in splits}

    if all(os.path.exists(p) for p in save_paths.values()):
        print(f"Forest spectra for these arguments already exist at {list(save_paths.values())}. Loading them instead of recomputing.")
        return {split: load_forest_spectra(p) for split, p in save_paths.items()}

    # get the necessary iwnfo about the spectra (dz, path, number of pixels and number of available spectra)
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

    rng = np.random.default_rng(seed)
    results = {}
    for split, (region, n_spectra) in splits.items():
        print(f"Stitching {n_spectra} {split} spectra from region {region} of the sightline grid")
        tau_split, usage_records = fill_master_tau(spectra_info, used_redshifts, tau_master, n_spectra, spectra_correction_function, region=region, rng=rng)

        for j in range(len(usage_records)):
            for i, x in enumerate(usage_records[j]):
                #print(x)
                usage_records[j][i]["left_wavelength"] = wave_master[x["left_pos"]]

        flux_split = np.exp(-tau_split[:, : idx_end + 1])
        results[split] = (wave_master[: idx_end + 1], flux_split, usage_records)

    check_splits_disjoint(results)

    if save_spectra:
        for split, (wave_split, flux_split, usage_records) in results.items():
            region, n_spectra = splits[split]
            print(f"Saving {n_spectra} {split} forest spectra to {save_paths[split]}")
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
    test_splits = {split: (region, 100) for split, (region, _) in SPLITS.items()}
    redshifts_to_use = [2.0]

    results = create_forest_spectra(test_path, test_range, redshifts_to_use, None, test_splits, save_spectra=True, seed=123)
