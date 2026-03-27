#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_messages/job.out.%j
#SBATCH -e ./job_messages/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J optuna-CLIMB-Project
#SBATCH -p p.gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00 # one day
#SBATCH --mem=50000 # p.gpu nodes have 500000 total
##SBATCH --exclusive # no
##SBATCH --dependency=singleton # no
#SBATCH --ntasks 1

module purge
module load gcc/12
module load hdf5-serial/1.12.2
module load openmpi/4.1
module load gsl/2.4
module load fftw-serial/3.3.10

/vera/u/jerbo/CLIMB-Project/env/bin/python optimization.py