#!/bin/bash
#SBATCH --job-name=drone_train       # Job name
#SBATCH --time=04:00:00              # Time limit (4 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-ee


# Define scratch directory
SCRATCH_DIR=/scratch/gpfs/TSILVER/de7281/MAE345

# Create experiment directory using SLURM job ID
EXPERIMENT_DIR=${SCRATCH_DIR}/job_${SLURM_JOB_ID}

# Create directory structure
mkdir -p ${EXPERIMENT_DIR}/logs
mkdir -p ${EXPERIMENT_DIR}/configs
mkdir -p ${EXPERIMENT_DIR}/models
mkdir -p ${EXPERIMENT_DIR}/results

# Redirect SLURM output to experiment directory
exec > ${EXPERIMENT_DIR}/logs/train_${SLURM_JOB_ID}.out 2> ${EXPERIMENT_DIR}/logs/train_${SLURM_JOB_ID}.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Experiment directory: ${EXPERIMENT_DIR}"

# Load CUDA module (required for GPU access)
module load cudatoolkit/11.8
module load cuda/11.8

# Initialize conda for bash shell
source /home/de7281/miniconda3/etc/profile.d/conda.sh

echo "hello2"


# Activate conda environment
conda activate mae345

echo "hello1"

# Verify GPU is accessible
echo "Checking GPU..."
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check PyTorch CUDA availability
echo "Checking PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Set threading environment variables to prevent deadlock
# MUST use 1 thread to avoid deadlock with PyTorch backward pass
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Navigate to script directory
cd /home/de7281/MAE345/final_project

echo "hello"
# Run training with Hydra
# Pass the experiment directory to the script
#
# Override examples:
#   Switch model:        models=continuous_action_model
#   Change num_bins:     models.num_bins=21
#   Use pretrained:      models.pretrained=true
#   Change action range: models.action_low=-0.5 models.action_high=0.5
#
python drone/scripts/training.py \
    --config-name=train_config_test \
    training.lr=1e-3 \
    training.batch_size=32 \
    training.num_epochs=20 \
    training.weight_decay=1e-5 \
    training.early_stopping_patience=3 \
    models.v3=false \
    experiment_dir=${EXPERIMENT_DIR} \


echo "End time: $(date)"
echo "Results saved to: ${EXPERIMENT_DIR}"
