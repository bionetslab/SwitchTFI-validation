#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --array=0-6
#SBATCH --error=%x_%A_%a.error
#SBATCH --output=%x_%A_%a.log
#SBATCH --job-name=drivaer
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_drivaer

cd $HOME/SwitchTFI-validation/validation

# Define NUM_CELLS options
NUM_CELLS=(100 500 1000 5000 10000 50000 100000)

# Pick entry for this task
NC=${NUM_CELLS[$SLURM_ARRAY_TASK_ID]}

echo "Running DrivAER with $NC cells on CPU"
srun python scalability.py -m drivaer -sg -nc $NC
