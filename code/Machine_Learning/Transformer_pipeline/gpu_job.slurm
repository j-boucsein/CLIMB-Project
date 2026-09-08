#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job_messages/job.out.%j
#SBATCH -e ./job_messages/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J Transformer
#
# GPU partition on binAC2:
#SBATCH --partition=gpu
#
# One node, one process, one GPU.
# a30  : 32 nodes x 2 GPUs (24 GB)  -> shortest queue, plenty for this model
# a100 : 8 nodes x 4 GPUs (40/80 GB)
# h200 : 4 nodes x 4 GPUs (141 GB)
#SBATCH --gres=gpu:a30:1
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=pv276@uni-heidelberg.de
#
# Wall clock limit (max. on the gpu partition is 14 days):
#SBATCH --time=24:00:00

# Python module. NOTE: do NOT activate ~/.local/envs/myenv here -- it has no
# torch and it hides ~/.local/lib/python3.12/site-packages, which is where the
# CUDA build of torch (2.14.0+cu130) lives.
module purge
module load devel/python/3.12_gnu_11.4

# The pip torch wheels ship their own CUDA 13 runtime, so no devel/cuda module
# is needed. If you ever hit a libcuda/driver mismatch, try uncommenting:
# module load devel/cuda/13.0

# Threading: keep the CPU-side ops inside the allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Headless plotting for the figures written at the end of the run.
export MPLBACKEND=Agg
# Unbuffered stdout so job.out.%j updates live.
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Allocated GPU(s): ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "----------------------------------------"

# Run the program:
python train_model.py
