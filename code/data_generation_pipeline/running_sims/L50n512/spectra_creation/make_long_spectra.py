import numpy as np

# This is not super pretty, but I think this is the best way to import stuff from ../../../?
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
from spectra_stitching import create_forest_spectra


def main():
    wavelength_range = (3800, 4500)
    redshifts_to_use = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    n_specs = 10000

    gp_paths = [f"/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/gridpoint{i}" for i in range(50)]
    gp_paths.append("/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/reference")

    for gp_path in gp_paths:
        create_forest_spectra(gp_path, wavelength_range, redshifts_to_use, None, n_specs, save_spectra=True, seed=123)


if __name__ == "__main__":
    main()