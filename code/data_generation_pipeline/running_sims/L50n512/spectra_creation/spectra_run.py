import argparse
import os
import time

import temet

BASE_PATH = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite"
N_REDSHIFTS = 11  # z = 2.0, 2.1, ..., 3.0


def redshift_at(i):
    return round(2 + i / 10, 1)


def spectrum_filename(gridpoint, z):
    return f"spectra_gridpoint{gridpoint}_z{z:.1f}_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"


def main():
    parser = argparse.ArgumentParser(description="Generate Ly-alpha spectra for one gridpoint.")
    parser.add_argument("--gridpoint", type=int, required=True, help="Gridpoint index, e.g. 3 for gridpoint3")
    args = parser.parse_args()

    gp = args.gridpoint
    path = os.path.join(BASE_PATH, f"gridpoint{gp}")
    spectra_dir = os.path.join(path, "data.files", "spectra")

    for i in range(N_REDSHIFTS):
        z = redshift_at(i)
        fname = spectrum_filename(gp, z)
        fpath = os.path.join(spectra_dir, fname)

        if os.path.isfile(fpath):
            print(f"[gridpoint{gp}] z={z:.1f}: already exists, skipping.", flush=True)
            continue

        t0 = time.time()
        sim = temet.sim(path, redshift=z)
        temet.spectra.spectrum.generate_rays_voronoi_fullbox(sim, nRaysPerDim=100)
        temet.spectra.spectrum.generate_spectra_from_saved_rays(
            sim, ion="H I", instrument="SDSS-BOSS", nRaysPerDim=100
        )
        t1 = time.time()
        print(f"[gridpoint{gp}] z={z:.1f}: done in {t1 - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
