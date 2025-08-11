#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres:gpu:rtx2080ti:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --error=%x_%j.error
#SBATCH --output=%x_%j.log
#SBATCH --job-name=drivaer
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_drivaer

cd $HOME/SwitchTFI-validation/validation

srun python scalability.py -m drivaer

