#!/bin/bash
#SBATCH --job-name=test_hydra
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Define scratch directory
SCRATCH_DIR=/scratch/gpfs/TSILVER/de7281/MAE345

# Create timestamped experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR=${SCRATCH_DIR}/test_${TIMESTAMP}

# Create directory structure
mkdir -p ${EXPERIMENT_DIR}

# Redirect SLURM output to experiment directory
exec > ${EXPERIMENT_DIR}/test_${SLURM_JOB_ID}.out 2> ${EXPERIMENT_DIR}/test_${SLURM_JOB_ID}.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Output directory: ${EXPERIMENT_DIR}"

# Initialize conda for bash shell
source /home/de7281/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate mae345

echo "Conda environment activated"

# Navigate to script directory
cd /home/de7281/MAE345/final_project/drone/scripts

echo "Changed to scripts directory: $(pwd)"

# Run test script
python test_hydra.py experiment_dir=${EXPERIMENT_DIR}

echo "End time: $(date)"
echo "Check output at: ${EXPERIMENT_DIR}/test_${SLURM_JOB_ID}.out"
