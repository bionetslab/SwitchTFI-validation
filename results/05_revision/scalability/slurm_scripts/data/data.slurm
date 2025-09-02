#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --error=%x_%j.error
#SBATCH --output=%x_%j.log
#SBATCH --job-name=data
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_val

cd $HOME/SwitchTFI-validation

srun python scalability.py -m data
