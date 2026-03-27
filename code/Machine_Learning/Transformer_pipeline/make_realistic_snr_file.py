import numpy as np
from dataset_functions import get_sdss_spectra

def main():
    resid_file_path = "SDSS_support_files/residcorr_v5_4_45.dat"

    data = np.load("SDSS_support_files/Custom_cat.npz")
    all_snrs = data["SNR"]
    all_pmfs = data["PMF"]

    filter_value = 2

    mask = np.where(all_snrs > filter_value)
    pmfs_filtered = all_pmfs[mask]

    snrs, _, fluxes_boss_specs = get_sdss_spectra(resid_file_path, pmfs_filtered)

    np_specs = []
    ivar_snrs = []
    for i, spec in enumerate(fluxes_boss_specs):
        if len(spec) == 402:
            np_specs.append(np.append(spec, spec[-1]))
            ivar_snrs.append(np.append(snrs[i], snrs[i][-1]))

    specs = np.array(np_specs)
    snrs_pp = np.array(ivar_snrs)

    print(specs.shape)
    print(snrs_pp.shape)

    snrs_median = np.median(snrs_pp, axis=0)

    np.save(f"snr_SDSS_spectra_mean_v2_snr{filter_value}.npy", snrs_median)




if __name__ == "__main__":
    main()