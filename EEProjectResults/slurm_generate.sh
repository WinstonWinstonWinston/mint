#!/bin/bash
#SBATCH -J mint-test
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p interactive-gpu
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1           # ONE task (one Python process)
#SBATCH --cpus-per-task=8             # CPUs for DataLoader workers
#SBATCH --gres=gpu:a40:1

# ---- Env setup -----------------------------------------
module purge
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2  

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
# initialize your bash shell for micromamba
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"
micromamba activate e3ti

# load again ??? somehow activating makes it break??
module purge
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

# Launch exactly one Python process bound to this GPU
srun -n 1 python generate.py
