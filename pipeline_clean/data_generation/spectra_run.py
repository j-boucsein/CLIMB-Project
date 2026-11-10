import os
import time
import temet


def write_log(log_path, message):
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


def spectrum_filename(name, z):
    return f"spectra_{name}_z{z:.1f}_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"


def generate_short_spectra(path, redshifts, log_path):
    """
    Generate spectra for a given simulation path and a list of redshifts.
    
    Args:
        path (str): The path to the simulation data.
        redshifts (list): A list of redshift values for which simulation snapshots exist and to generate spectra for.  
        log_path (str): The path to the log file.
    
    """
    spectra_dir = os.path.join(path, "data.files", "spectra")
    box_name = os.path.basename(os.path.normpath(path))

    for z in redshifts:
        write_log(log_path, f"Generating spectra for redshift: {z:.1f} , box: {box_name}")
        fname = spectrum_filename(box_name, z)
        fpath = os.path.join(spectra_dir, fname)

        if os.path.isfile(fpath):
            write_log(log_path, f"[{box_name}] z={z:.1f}: already exists, skipping.")
            continue

        t0 = time.time()
        sim = temet.sim(path, redshift=z)
        temet.spectra.spectrum.generate_rays_voronoi_fullbox(sim, nRaysPerDim=100)
        temet.spectra.spectrum.generate_spectra_from_saved_rays(
            sim, ion="H I", instrument="SDSS-BOSS", nRaysPerDim=100
        )
        t1 = time.time()
        write_log(log_path, f"Done! Generation took {(t1 - t0)/3600:.3f} h")


def main():
    # Example of usage:
    basepath = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite/"
    gridpoint_to_run = "gridpoint0"
    path = os.path.join(basepath, gridpoint_to_run)

    log_path = "/logs/datageneration.log"

    redshifts = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

    generate_short_spectra(path, redshifts, log_path)


if __name__ == "__main__":
    main()
