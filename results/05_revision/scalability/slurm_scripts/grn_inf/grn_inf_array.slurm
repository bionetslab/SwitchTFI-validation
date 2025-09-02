#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=0-6
#SBATCH --error=%x_%A_%a.error
#SBATCH --output=%x_%A_%a.log
#SBATCH --job-name=grn_inf
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_grn   # <-- env for GRN inference

cd $HOME/SwitchTFI-validation/validation

# Source common parameters file (NUM_CELLS defined here)
source $HOME/SwitchTFI-validation/results/05_revision/scalability/slurm_scripts/params.sh

# Pick the entry for this task
NC=${VARY_NUM_CELLS_NUM_CELLS[$SLURM_ARRAY_TASK_ID]}

echo "Running GRN inference with $NC cells"
srun python scalability.py -m grn_inf -vncnc $NC
