#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_messages/job.out.%j
#SBATCH -e ./job_messages/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J Spectra_processing
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=2
#SBATCH --mem=250000 # 500000
#
##SBATCH --dependency=singleton
#
#SBATCH --mail-type=none
#SBATCH --mail-user=pv276@uni-heidelberg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=50:00:00

##SBATCH --constraint=ib

# Load compiler and MPI modules (must be the same as used for compiling the code)

module purge
module load compiler/gnu/14.2
module load lib/hdf5/1.12-gnu-14.2
module load mpi/openmpi/4.1-gnu-14.2
module load numlib/gsl/2.8-gnu-14.2
module load numlib/fftw/3.3.10-gnu-14.2

module load devel/python/3.12_gnu_11.4

source /home/hd/hd_hd/hd_pv276/.local/envs/myenv/bin/activate

# Run the program:
python make_long_spectra.py
