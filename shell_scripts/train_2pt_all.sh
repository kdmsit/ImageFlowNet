#!/bin/bash

#SBATCH --job-name=train_simulation
#SBATCH --partition=pi_krishnaswamy
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --output=../sbatch_logs/train_simulation_%j.out
#SBATCH --error=../sbatch_logs/train_simulation_%j.err

set -euo pipefail
mkdir -p ../sbatch_logs

module --force purge
module load StdEnv
module load miniconda

# Activate conda environment
# Note: conda activation scripts may reference unset vars; temporarily disable 'nounset'.
set +u
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate imageflownet_gpu
set -u


cd /home/sa2556/imageflownet/ImageFlowNet

# Diagnostics
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__, "built cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

python src/train_2pt_all.py \
  --mode train \
  --max-epochs 120 \
  --dataset-name synthetic \
  --dataset-path /home/sa2556/imageflownet/ImageFlowNet/data/synthesized \
  --image-folder base \
  --segmentor-ckpt /home/sa2556/imageflownet/ImageFlowNet/checkpoints/segmentor.pth \
  --gpu-id 0
