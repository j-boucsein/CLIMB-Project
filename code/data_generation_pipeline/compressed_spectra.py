import numpy as np
import csv
from astropy.table import Table
from astropy.io import fits
import tqdm
from concurrent.futures import ProcessPoolExecutor


def get_sdss_file_ids(path, snr_lowe_edge, snr_upper_edge=np.inf):
    """ Get a list of (PLATE, MJD, FIBERID) from the SDSS BOSS Lyman alpha forest catalogue
      of spectra with S/N in given range.

    Args:
        path (string): path to the catalogue file
        snr_lowe_edge (float): lower filter for the signa to noise ratio of the spectra
        snr_upper_edge (float, optional): upper filter for the signa to noise ratio of the spectra

    Returns:
        list: list of (PLATE, MJD, FIBERID) tuples for spectra in the SDSS BOSS Lyman alpha forest catalogue
    """
    lya_cat = Table.read(path)
    pmf_list = []

    for row in lya_cat:
        if row["SNR"] > snr_lowe_edge and row["SNR"] < snr_upper_edge:
            pmf_list.append((row["PLATE"], row["MJD"], row["FIBERID"]))

    return pmf_list


def build_common_grid(resid_lam_file, z_min=1.95, z_max=3.05):
    """Build a fixed wavelength grid (aligned to the resid grid) spanning the given
    Lyman-alpha absorption redshift range."""
    LYA_REST = 1215.67  # Angstroms
    wave_min = LYA_REST * (1 + z_min)
    wave_max = LYA_REST * (1 + z_max)
    log_min = np.log10(wave_min)
    log_max = np.log10(wave_max)

    grid_mask = (resid_lam_file >= log_min) & (resid_lam_file <= log_max)
    common_loglam = resid_lam_file[grid_mask]
    common_wavelength = 10**common_loglam
    return common_loglam, common_wavelength


_resid_lam_file = None
_resid_file = None
_common_loglam = None
_speclya_basepath = None


def _init_worker(resid_lam_file, resid_file, common_loglam, speclya_basepath):
    global _resid_lam_file, _resid_file, _common_loglam, _speclya_basepath
    _resid_lam_file = resid_lam_file
    _resid_file = resid_file
    _common_loglam = common_loglam
    _speclya_basepath = speclya_basepath


def _process_one_spectrum(task):
    idx, plate, mjd, fiber, z_qso = task
    n_pix = len(_common_loglam)

    F_full = np.full(n_pix, np.nan, dtype=np.float32)
    SIGMA_F_full = np.full(n_pix, np.nan, dtype=np.float32)
    mask_full = np.zeros(n_pix, dtype=bool)  # True = good/valid pixel

    fiber_str = f"{fiber:04d}"
    filename = f"speclya-{plate}-{mjd}-{fiber_str}.fits"
    file_path = f"{_speclya_basepath}/{plate}/{filename}"

    try:
        with fits.open(file_path, memmap=True) as hdul:
            data = hdul[1].data
            loglam     = data["LOGLAM"]
            flux       = data["FLUX"]
            ivar       = data["IVAR"]
            cont       = data["CONT"]
            dla_corr   = data["DLA_CORR"]
            mask_comb  = data["MASK_COMB"]
            noise_corr = data["NOISE_CORR"]
    except (FileNotFoundError, OSError):
        return (idx, (plate, mjd, fiber, z_qso), F_full, SIGMA_F_full, mask_full)

    STEP = 1e-4         # BOSS/SDSS log-wavelength pixel size

    # -------- Resid index lookup (arithmetic, with safety fallback) --------
    i0 = int(round((loglam[0] - _resid_lam_file[0]) / STEP))
    i1 = i0 + len(loglam)

    if not (np.isclose(_resid_lam_file[i0], loglam[0], atol=1e-6) and
            np.isclose(_resid_lam_file[i1 - 1], loglam[-1], atol=1e-6)):
        i0 = np.searchsorted(_resid_lam_file, loglam[0] - 1e-6)
        i1 = i0 + len(loglam)
        if not np.isclose(_resid_lam_file[i0], loglam[0], atol=1e-6):
            return (idx, (plate, mjd, fiber, z_qso), F_full, SIGMA_F_full, mask_full)

    resid = _resid_file[i0:i1]
    # ------------------------------------------------------------------

    wavelength = 10**loglam

    # Basic SDSS quality mask + guard against division blowups
    good_pixel = (
        (mask_comb == 0) &
        (ivar != 0) &
        (cont != 0) &
        (resid != 0)
    )

    F = np.full_like(flux, np.nan)
    SIGMA_F = np.full_like(flux, np.nan)

    F[good_pixel] = (flux[good_pixel] * dla_corr[good_pixel]
                     / (cont[good_pixel] * resid[good_pixel]))
    SIGMA_F[good_pixel] = np.sqrt(
        ivar[good_pixel] * resid[good_pixel]**2 * noise_corr[good_pixel]**2
        * cont[good_pixel]**2 / dla_corr[good_pixel]**2
    )

    # Extra safety: drop any pixel that still isn't finite despite passing masks
    good_pixel = good_pixel & np.isfinite(F) & np.isfinite(SIGMA_F)

    # -------- Map onto the common grid (same absolute grid => same offset trick) --------
    common_idx = np.round((loglam - _common_loglam[0]) / STEP).astype(int)
    in_range = (common_idx >= 0) & (common_idx < n_pix)

    idxs = common_idx[in_range]
    F_full[idxs] = F[in_range]
    SIGMA_F_full[idxs] = SIGMA_F[in_range]
    mask_full[idxs] = good_pixel[in_range]

    # NaN out anything that landed in-range but isn't actually good
    F_full[~mask_full] = np.nan
    SIGMA_F_full[~mask_full] = np.nan
    # ------------------------------------------------------------------

    return (idx, (plate, mjd, fiber, z_qso), F_full, SIGMA_F_full, mask_full)


def build_and_save_spectra(resid_file_path, pmf_list, redshifts, output_path,
                            speclya_basepath="/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/BOSSLyaDR9_spectra",
                            z_min=1.95, z_max=3.05,
                            n_workers=8, chunksize=20):
    """
    Loads, corrects, and masks all spectra in pmf_list, projects them onto a common
    wavelength grid spanning [z_min, z_max] (Lyman-alpha absorption redshift), and
    saves the result to a single .npz file for fast repeated access.

    Args:
        resid_file_path: path to the resid file
        pmf_list: list of (PLATE, MJD, FIBERID) tuples
        redshifts: list/array of quasar redshifts, same length and order as pmf_list
        output_path: path to save the .npz file to (e.g. "spectra_cache.npz")
        z_min, z_max: Lyman-alpha absorption redshift range to keep
    """
    assert len(pmf_list) == len(redshifts), "pmf_list and redshifts must be the same length"

    # -------- Read RESID File --------
    resid, resid_lam = [], []
    with open(resid_file_path) as rfile:
        reader = csv.reader(rfile, delimiter=" ")
        for i, row in enumerate(reader):
            if i != 0:
                resid.append(float(row[-1]))
                resid_lam.append(float(row[1]))
    resid_lam_file = np.array(resid_lam)
    resid_file = np.array(resid)
    # --------------------------------

    common_loglam, common_wavelength = build_common_grid(resid_lam_file, z_min, z_max)
    n_pix = len(common_loglam)
    n = len(pmf_list)

    tasks = [
        (idx, plate, mjd, fiber, z)
        for idx, ((plate, mjd, fiber), z) in enumerate(zip(pmf_list, redshifts))
    ]

    flux_matrix = np.full((n, n_pix), np.nan, dtype=np.float32)
    sigma_matrix = np.full((n, n_pix), np.nan, dtype=np.float32)
    mask_matrix = np.zeros((n, n_pix), dtype=bool)
    pmf_array = np.zeros((n, 3), dtype=np.int64)
    z_array = np.zeros(n, dtype=np.float64)
    valid_spectrum = np.zeros(n, dtype=bool)  # False = file missing / totally unusable

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(resid_lam_file, resid_file, common_loglam, speclya_basepath),
    ) as executor:
        for idx, pmf_z, F_full, SIGMA_F_full, mask_full in tqdm.tqdm(
            executor.map(_process_one_spectrum, tasks, chunksize=chunksize), total=n
        ):
            plate, mjd, fiber, z_qso = pmf_z
            flux_matrix[idx] = F_full
            sigma_matrix[idx] = SIGMA_F_full
            mask_matrix[idx] = mask_full
            pmf_array[idx] = (plate, mjd, fiber)
            z_array[idx] = z_qso
            valid_spectrum[idx] = np.any(mask_full)

    np.savez_compressed(
        output_path,
        wavelength=common_wavelength,
        flux=flux_matrix,
        mask=mask_matrix,
        snr=sigma_matrix,          # keeping your original naming (this is sigma_F / per-pixel error)
        redshift=z_array,
        pmf=pmf_array,
        valid_spectrum=valid_spectrum,
        z_min=z_min,
        z_max=z_max,
    )

    print(f"Saved {n} spectra ({np.sum(valid_spectrum)} with at least one good pixel) "
          f"to {output_path}, grid size = {n_pix} pixels.")


def load_spectra_cache(path):
    """Load a spectra cache saved by build_and_save_spectra. Returns a dict of arrays."""
    with np.load(path) as npz:
        return {
            "wavelength": npz["wavelength"],
            "flux": npz["flux"],
            "mask": npz["mask"],
            "snr": npz["snr"],
            "redshift": npz["redshift"],
            "pmf": npz["pmf"],
            "valid_spectrum": npz["valid_spectrum"],
            "z_min": float(npz["z_min"]),
            "z_max": float(npz["z_max"]),
        }


if __name__ == "__main__":
    cat_file = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/BOSSLyaDR9_cat.fits"
    resid_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/residcorr_v5_4_45.dat"
    out_file_path = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/SDSS_spectra/SDSS_support_files/spectra_cache.npz"

    lya_cat = Table.read(cat_file)
    redshifts_list = []
    pmf_list = []

    for row in lya_cat:
        redshifts_list.append(row['Z_VI'])
        pmf_list.append((row["PLATE"], row["MJD"], row["FIBERID"]))

    build_and_save_spectra(resid_path, pmf_list, redshifts_list, out_file_path, z_min=1.95, z_max=3.05)