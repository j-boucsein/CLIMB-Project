#!/bin/bash -l
# Standard output and error (%A = job id, %a = array task id):
#SBATCH -o ./job_messages/job.out.%A_%a
#SBATCH -e ./job_messages/job.err.%A_%a
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J Spectra
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=2
#SBATCH --mem=460000
#
# Wall clock limit:
#SBATCH --time=60:00:00
#
#SBATCH --constraint=ib
#
# NOTE: --array is set at submission time by check_and_submit.py,
# e.g. --array=0-12%20 to run 13 jobs with at most 20 concurrent.

module purge
module load compiler/gnu/14.2
module load lib/hdf5/1.12-gnu-14.2
module load mpi/openmpi/4.1-gnu-14.2
module load numlib/gsl/2.8-gnu-14.2
module load numlib/fftw/3.3.10-gnu-14.2

module load devel/python/3.12_gnu_11.4

source /home/hd/hd_hd/hd_pv276/.local/envs/myenv/bin/activate

# Each array task looks up which gridpoint it should process.
# missing_gridpoints.txt has one gridpoint number per line;
# SLURM_ARRAY_TASK_ID is 0-indexed, sed lines are 1-indexed.
GRIDPOINT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" missing_gridpoints.txt)

echo "Array task ${SLURM_ARRAY_TASK_ID} -> gridpoint ${GRIDPOINT}"

python spectra_run.py --gridpoint "${GRIDPOINT}"
