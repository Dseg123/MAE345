#!/bin/bash
#SBATCH --job-name=test_minimal
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Define scratch directory
SCRATCH_DIR=/scratch/gpfs/TSILVER/de7281/MAE345
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR=${SCRATCH_DIR}/test_minimal_${TIMESTAMP}

mkdir -p ${EXPERIMENT_DIR}

# Redirect output
exec > ${EXPERIMENT_DIR}/test.out 2> ${EXPERIMENT_DIR}/test.err

echo "Job ID: $SLURM_JOB_ID"
echo "Experiment directory: ${EXPERIMENT_DIR}"

# Initialize conda
source /home/de7281/miniconda3/etc/profile.d/conda.sh
conda activate mae345

# Navigate to script directory
cd /home/de7281/MAE345/final_project/drone/scripts

echo "Running minimal test..."
python training_minimal.py \
    experiment_dir=${EXPERIMENT_DIR} \
    hydra.job.chdir=false \
    hydra.output_subdir=null

echo "End time: $(date)"
echo "Output at: ${EXPERIMENT_DIR}/test.out"
