#!/usr/bin/env python3
"""Full DESI DR1 Lyα: coverage map + SNR histogram.

Streams every ``delta-*.fits.gz`` file in DATA_DIR once and produces two figures
(saved into ./plots/):

  1. Coverage map -- one row per spectrum, white where the pixel is usable
     (non-NaN ``DELTA_BLIND``) and black where masked / missing, sorted by the
     first then last usable pixel.
  2. SNR histogram of the "training-usable" spectra -- those that span the
     WAVE_RANGE window with at most MAX_INTERP_FRAC interior NaNs (interior gaps
     are linearly interpolated so the window is fully valued).

This mirrors the two final cells of DESI_spectra_testing.ipynb but runs over the
full catalogue instead of the 5-file local sample. Single pass, ~a few minutes.

Run:  python desi_full_coverage_and_snr.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt
from astropy.io import fits

DATA_DIR = "/virgotng/mpia/obs/DESI/DR1/vac/lya-deltas/delta-lya-0-0"
WAVE_RANGE = (3600.0, 4000.0)    # training window [Å]
MAX_INTERP_FRAC = 0.20           # drop a LOS if more than this fraction of the window is masked
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def fill_interior_nans(row, x):
    """Linearly interpolate NaNs in ``row`` (assumes both ends are finite)."""
    good = np.isfinite(row)
    out = row.copy()
    out[~good] = np.interp(x[~good], x[good], row[good])
    return out


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "delta-*.fits.gz")))
    if not files:
        raise SystemExit(f"no delta-*.fits.gz files found in {DATA_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"found {len(files)} delta files in {DATA_DIR}")

    finite_rows = []          # per-file (n_LOS, n_pix) boolean usable masks
    snr_usable = []           # MEANSNR of training-usable spectra
    usable_spectra = []       # interpolated NaN-free window deltas (the NN inputs)
    n_total = n_no_coverage = n_too_masked = 0
    lam_grid = win = lam_w = None

    for k, fn in enumerate(files):
        with fits.open(fn) as f:
            lam = f["LAMBDA"].data.astype(float)
            snr = np.asarray(f["METADATA"].data["MEANSNR"], dtype=float)
            delta = f["DELTA_BLIND"].data.astype(np.float32)

        if lam_grid is None:
            lam_grid = lam
            win = (lam >= WAVE_RANGE[0]) & (lam <= WAVE_RANGE[1])
            lam_w = lam[win]
        elif lam.shape != lam_grid.shape or not np.allclose(lam, lam_grid):
            raise SystemExit(f"{os.path.basename(fn)} has a different wavelength grid")

        finite = np.isfinite(delta)
        finite_rows.append(finite)

        # training-usable criterion on the window
        finw = finite[:, win]
        spans = finw[:, 0] & finw[:, -1]                    # finite at both window edges
        interp_frac = (~finw).sum(axis=1) / finw.shape[1]
        usable = spans & (interp_frac <= MAX_INTERP_FRAC)

        n_total += len(snr)
        n_no_coverage += int((~spans).sum())
        n_too_masked += int((spans & (interp_frac > MAX_INTERP_FRAC)).sum())

        dw = delta[:, win]
        for i in np.where(usable)[0]:
            usable_spectra.append(fill_interior_nans(dw[i].astype(float), lam_w))
        snr_usable.append(snr[usable])

        if (k + 1) % 100 == 0 or k + 1 == len(files):
            print(f"  processed {k + 1}/{len(files)} files  (LOS so far: {n_total})")

    snr_usable = np.concatenate(snr_usable)
    usable_spectra = np.array(usable_spectra)
    n_usable = len(snr_usable)

    print(f"\ntotal spectra (LOS):                 {n_total}")
    print(f"training-usable (interpolated):      {n_usable}")
    if n_usable:
        print(f"  -> array shape {usable_spectra.shape}, "
              f"contains NaNs: {np.isnan(usable_spectra).any()}")
    print(f"dropped:                             {n_total - n_usable}")
    print(f"  - forest does not span window:     {n_no_coverage}")
    print(f"  - >{MAX_INTERP_FRAC:.0%} of window masked:        {n_too_masked}")

    # -------- SNR histogram --------
    snr_view = (0.0, 20.0)        # display range
    n_bins = 100                  # fine bins to resolve the curve shape
    counts, edges = np.histogram(snr_usable, bins=n_bins, range=snr_view)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(centres, counts, ls="--", color="steelblue", lw=1.2)

    thresholds = (2.0, 5.0, 10.0)
    print("\nspectra above SNR thresholds:")
    for thr in thresholds:
        n_above = int((snr_usable > thr).sum())
        print(f"  SNR > {thr:>4.0f} : {n_above:>7d}  ({n_above / n_usable:.1%})")
        ax.axvline(thr, color="0.4", ls=":", lw=1.0)
        ax.text(thr + 0.15, counts.max() * 0.95,
                f"> {thr:.0f}: {n_above:,}\n({n_above / n_usable:.0%})",
                rotation=0, va="top", ha="left", fontsize=8, color="0.25")

    ax.set_xlim(*snr_view)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("MEANSNR")
    ax.set_ylabel("number of spectra")
    ax.set_title(f"SNR of training-usable spectra "
                 f"({n_usable} of {n_total}, {WAVE_RANGE[0]:.0f}-{WAVE_RANGE[1]:.0f} Å)")
    snr_path = os.path.join(OUT_DIR, "desi_full_snr_hist.pdf")
    fig.savefig(snr_path, format="pdf", dpi=120)
    print(f"saved {snr_path}")

    # -------- coverage map --------
    finite = np.concatenate(finite_rows, axis=0)            # (n_total, n_pix) bool
    del finite_rows
    n_pix = finite.shape[1]
    fv = np.where(finite.any(axis=1), finite.argmax(axis=1), n_pix)
    lv = np.where(finite.any(axis=1), n_pix - 1 - finite[:, ::-1].argmax(axis=1), -1)
    order = np.lexsort((lv, fv))
    img = finite[order].astype(np.uint8)                   # white = usable, black = not
    del finite

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1,
              extent=[lam_grid.min(), lam_grid.max(), 0, img.shape[0]],
              interpolation="nearest")
    for edge in WAVE_RANGE:
        ax.axvline(edge, color="lime", ls="--", lw=1.0)
    ax.set_xlabel(r"observed wavelength $\lambda$ [Å]")
    ax.set_ylabel("spectrum (sorted by first, then last usable pixel)")
    ax.set_title(f"DESI DR1: per-pixel coverage of {img.shape[0]} spectra "
                 "(white = usable, black = masked / no data)")
    cov_path = os.path.join(OUT_DIR, "desi_full_coverage_map.pdf")
    fig.savefig(cov_path, format="pdf", dpi=120)
    print(f"saved {cov_path}")


if __name__ == "__main__":
    main()
