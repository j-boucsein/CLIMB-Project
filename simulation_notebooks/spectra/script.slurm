#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J CLIMB-spectra
#
# Number of nodes and MPI tasks per node:
#SBATCH -p p.vera # p.large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --mem=250000 # 500000
#
#SBATCH --dependency=singleton
#
#SBATCH --mail-type=none
#SBATCH --mail-user=pv276@uni-heidelberg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
module purge
module load gcc/12
module load hdf5-serial/1.12.2
module load openmpi/4.1
module load gsl/2.4
module load fftw-serial/3.3.10
# Run the program:
./create_spectra.py
