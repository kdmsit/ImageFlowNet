#!/bin/bash

#SBATCH --job-name=imageflownet        # Job name
#SBATCH --output=logs/imageflownet_%j.out
#SBATCH --error=logs/imageflownet_%j.out
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --partition=gpu_l40
#SBATCH --gpus=1
#SBATCH --nodelist=gnode4

#export CUDA_VISIBLE_DEVICES='0'
#export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
#export NUMEXPR_MAX_THREADS=64

source /home/rs1/21CS92R01/miniconda3/bin/activate imageflownet

python synthetic.py
python train_2pt_all.py --model ImageFlowNetODE --random-seed 1 --dataset-name synthetic --max-epochs 100
