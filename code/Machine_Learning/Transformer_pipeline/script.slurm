#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./run.out
#SBATCH -e ./run.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J CLIMB-Transformer-training
#SBATCH -p p.gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH -t 6:00:00 
#SBATCH --mem=50000 # p.gpu nodes have 500000 total
##SBATCH --exclusive # no
##SBATCH --dependency=singleton
#SBATCH --ntasks 1

module purge
module load gcc/12
module load hdf5-serial/1.12.2
module load openmpi/4.1
module load gsl/2.4
module load fftw-serial/3.3.10

/vera/u/jerbo/CLIMB-Project/env/bin/python train_model.py