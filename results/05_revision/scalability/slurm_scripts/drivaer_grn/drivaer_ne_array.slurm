#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --array=0-17
#SBATCH --error=%x_%A_%a.error
#SBATCH --output=%x_%A_%a.log
#SBATCH --job-name=drivaer_grn
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate switchtfi_drivaer

cd $HOME/SwitchTFI-validation/validation

# Source common parameters file (NUM_EDGES defined here)
source $HOME/SwitchTFI-validation/results/05_revision/scalability/slurm_scripts/params.sh

NUM_CELLS_LEN=${#VARY_NUM_EDGES_NUM_CELLS[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

EDGE_IDX=$(( TASK_ID / NUM_CELLS_LEN))
CELL_IDX=$(( TASK_ID % NUM_CELLS_LEN))

NE=${VARY_NUM_EDGES_NUM_EDGES[$EDGE_IDX]}
NC=${VARY_NUM_EDGES_NUM_CELLS[$CELL_IDX]}

echo "Running DrivAER with GRN size $NE edges and $NC cells"
srun python scalability.py -m drivaer_grn -vnene $NE -vnenc $NC
