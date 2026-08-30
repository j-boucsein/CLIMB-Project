"""
Scan all gridpoints for missing spectra (z = 2.0 ... 3.0), then submit a
single Slurm array job that runs only the gridpoints that still need work.
The array's %20 suffix caps concurrent running tasks at 20, whatever the
cluster's per-user job limit is - so you can safely leave this to submit
all missing gridpoints at once.

Usage:
    python check_and_submit.py            # check + submit
    python check_and_submit.py --dry-run  # check only, don't submit
"""

import argparse
import os
import subprocess

BASE_PATH = "/pfs/10/project/bw21g005/ly_alpha_sbi_paper/L50n512_suite"
N_GRIDPOINTS = 50
N_REDSHIFTS = 11  # z = 2.0, 2.1, ..., 3.0
MAX_CONCURRENT = 20
MISSING_LIST_FILE = "missing_gridpoints.txt"
SBATCH_SCRIPT = "spectra_job.sbatch"


def redshift_at(i):
    return round(2 + i / 10, 1)


def spectrum_filename(gridpoint, z):
    return f"spectra_gridpoint{gridpoint}_z{z:.1f}_n100d2-fullbox_SDSS-BOSS_HI_combined.hdf5"


def missing_redshifts(gridpoint):
    """Return list of redshifts for which the spectrum file is missing."""
    spectra_dir = os.path.join(BASE_PATH, f"gridpoint{gridpoint}", "data.files", "spectra")
    missing = []
    for i in range(N_REDSHIFTS):
        z = redshift_at(i)
        fpath = os.path.join(spectra_dir, spectrum_filename(gridpoint, z))
        if not os.path.isfile(fpath):
            missing.append(z)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only report status, don't submit jobs.")
    args = parser.parse_args()

    incomplete = {}
    for gp in range(N_GRIDPOINTS):
        missing = missing_redshifts(gp)
        if missing:
            incomplete[gp] = missing

    complete_count = N_GRIDPOINTS - len(incomplete)
    print(f"{complete_count}/{N_GRIDPOINTS} gridpoints already fully complete.")

    if not incomplete:
        print("Nothing to do - all spectra already exist.")
        return

    print(f"{len(incomplete)} gridpoints need work:")
    for gp, missing in sorted(incomplete.items()):
        z_str = ", ".join(f"{z:.1f}" for z in missing)
        print(f"  gridpoint{gp}: missing z = [{z_str}]")

    gridpoints_to_run = sorted(incomplete.keys())
    with open(MISSING_LIST_FILE, "w") as f:
        for gp in gridpoints_to_run:
            f.write(f"{gp}\n")
    print(f"\nWrote {len(gridpoints_to_run)} gridpoint(s) to {MISSING_LIST_FILE}")

    n_jobs = len(gridpoints_to_run)
    array_spec = f"0-{n_jobs - 1}%{MAX_CONCURRENT}"
    cmd = ["sbatch", f"--array={array_spec}", SBATCH_SCRIPT]

    if args.dry_run:
        print(f"[dry run] Would submit: {' '.join(cmd)}")
        return

    os.makedirs("job_messages", exist_ok=True)
    print(f"Submitting: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
