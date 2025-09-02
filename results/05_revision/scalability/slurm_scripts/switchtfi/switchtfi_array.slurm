#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=0-20
#SBATCH --error=%x_%A_%a.error
#SBATCH --output=%x_%A_%a.log
#SBATCH --job-name=switchtfi
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_val

cd $HOME/SwitchTFI-validation

# Source common parameters file (NUM_CELLS defined here)
source $HOME/SwitchTFI-validation/results/05_revision/scalability/slurm_scripts/params.sh

NUM_EDGES_LEN=${#VARY_NUM_CELLS_NUM_EDGES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

CELL_IDX=$(( TASK_ID / NUM_EDGES_LEN))
EDGE_IDX=$(( TASK_ID % NUM_EDGES_LEN))

NC=${VARY_NUM_CELLS_NUM_CELLS[$CELL_IDX]}
NE=${VARY_NUM_CELLS_NUM_EDGES[$EDGE_IDX]}

echo "Running SwitchTFI with $NC cells and $NE edge fraction"
srun python scalability.py -m switchtfi -vncnc $NC -vncne $NE
