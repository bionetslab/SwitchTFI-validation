#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --array=0-5
#SBATCH --error=%x_%A_%a.error
#SBATCH --output=%x_%A_%a.log
#SBATCH --job-name=switchtfi_grn
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_val   # <-- env for SwitchTFI

cd $HOME/SwitchTFI-validation/validation

# Define NUM_EDGES options
NUM_EDGES=(100 500 1000 5000 10000 50000)

# Pick entry for this task
NE=${NUM_EDGES[$SLURM_ARRAY_TASK_ID]}

echo "Running SwitchTFI with GRN size $NE edges"
srun python scalability.py -m switchtfi_grn -ne $NE
