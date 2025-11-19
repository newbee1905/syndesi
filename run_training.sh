#!/bin/bash
# run_training.sh
# Script to launch multi-GPU training with DeepSpeed

# Number of GPUs to use
NUM_GPUS=1

# Configuration file (defaults to config.yaml in conf/)
CONFIG_NAME=${1:-config}

# Task type: pretrain or finetune
TASK=${2:-pretrain}

echo "================================="
echo "DeepSpeed Multi-GPU Training"
echo "================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "================================="

# Launch with torchrun (recommended for PyTorch 1.10+)
torchrun \
	--nproc_per_node=$NUM_GPUS \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr=localhost \
	--master_port=29500 \
	main_deepspeed.py \
	experiment=$TASK

# Alternative: Launch with DeepSpeed launcher
# deepspeed --num_gpus=$NUM_GPUS main_deepspeed.py experiment=$TASK

echo ""
echo "Training complete!"
echo "Check logs/ for TensorBoard logs"
echo "Check checkpoints/ for model checkpoints"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=logs/"
