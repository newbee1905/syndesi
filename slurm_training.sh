#!/bin/bash
#SBATCH --qos=xbatch
#SBATCH --gpus=l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/pretrain_deepspeed_%j.out
#SBATCH --error=logs/pretrain_deepspeed_%j.err
#SBATCH --job-name=deepspeed_pretrain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s221056384@deakin.edu.au

module load NVHPC/24.9-CUDA-12.6.0
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
set -euo pipefail

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

echo "Starting Fine-tuning: Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

srun torchrun \
	--nproc_per_node=4 \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	main_deepspeed.py \
	experiment=pretrain

echo "Pre-training complete: Job $SLURM_JOB_ID"
