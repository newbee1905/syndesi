from tqdm import tqdm
import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer

import deepspeed

from data import (
	BytePatchSMILESProcessor, 
	MLMMapDataset, mlm_collate_fn,
	FinetuneMapDataset, finetune_collate_fn,
	ChemBERTaFinetuneDataset, chemberta_collate_fn
)
from modules_deepspeed import MLMTrainingModule, QM9TrainingModule

from typing import Dict

import torch
import torch.distributed as dist
import time
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter

def get_dataloaders(cfg: DictConfig, world_size: int, rank: int):
	"""Initialize and return train and validation dataloaders."""
	
	if rank == 0:
		print(f"Setting up dataloaders for task: {cfg.task_name}")
	
	if cfg.task_name == "pretrain":
		dataset = load_dataset(cfg.data.dataset_name)
		dataset_split = dataset['train'].train_test_split(test_size=0.01, seed=42)
		train_smiles = dataset_split['train']['smiles']
		val_smiles = dataset_split['test']['smiles']
		
		processor = BytePatchSMILESProcessor(
			vocab_size=cfg.data.vocab_size, 
			max_bytes_per_atom=cfg.data.max_bytes_per_atom
		)
		
		train_dataset = MLMMapDataset(
			train_smiles, processor, 
			mlm_probability=cfg.data.mlm_probability, 
			augment_prob=cfg.data.augment_prob
		)
		val_dataset = MLMMapDataset(
			val_smiles, processor, 
			mlm_probability=cfg.data.mlm_probability, 
			augment_prob=0.0
		)
		collate_fn = mlm_collate_fn
		
	elif cfg.task_name == "finetune":
		dataset = load_dataset(cfg.data.dataset_name, split='train')
		
		# Filter out rows with missing data
		dataset = dataset.filter(
			lambda x: x.get('SMILES') is not None and 
					 x.get(cfg.data.target_column) is not None
		)
		
		# Split dataset
		train_size = int((1.0 - cfg.data.val_split - cfg.data.test_split) * len(dataset))
		val_size = int(cfg.data.val_split * len(dataset))
		test_size = len(dataset) - train_size - val_size
		
		generator = torch.Generator().manual_seed(42)
		train_data, val_data, _ = random_split(
			dataset, [train_size, val_size, test_size], generator=generator
		)

		# Extract SMILES and targets
		train_smiles = [x['SMILES'] for x in train_data]
		train_targets = [float(x[cfg.data.target_column]) for x in train_data]
		val_smiles = [x['SMILES'] for x in val_data]
		val_targets = [float(x[cfg.data.target_column]) for x in val_data]

		if cfg.model.architecture == "mamba_bimamba":
			if rank == 0:
				print("Using Mamba-BiMamba Finetune dataset")
			processor = BytePatchSMILESProcessor(
				cfg.module.model_config.vocab_size, 
				cfg.module.model_config.max_bytes_per_atom
			)
			train_dataset = FinetuneMapDataset(train_smiles, train_targets, processor)
			val_dataset = FinetuneMapDataset(val_smiles, val_targets, processor)
			collate_fn = finetune_collate_fn
			
		elif cfg.model.architecture == "chemberta":
			if rank == 0:
				print("Using ChemBERTa Finetune dataset")
			tokenizer = AutoTokenizer.from_pretrained(cfg.module.chemberta_model)
			train_dataset = ChemBERTaFinetuneDataset(
				train_smiles, train_targets, tokenizer, 
				max_length=cfg.model.chemberta.max_length
			)
			val_dataset = ChemBERTaFinetuneDataset(
				val_smiles, val_targets, tokenizer, 
				max_length=cfg.model.chemberta.max_length
			)
			collate_fn = chemberta_collate_fn
		else:
			raise ValueError(f"Unknown architecture: {cfg.model.architecture}")
	
	else:
		raise ValueError(f"Unknown task_name: {cfg.task_name}")

	# Create distributed samplers
	train_sampler = DistributedSampler(
		train_dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True,
		seed=42
	)
	val_sampler = DistributedSampler(
		val_dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=False
	)
	
	# Create dataloaders
	train_loader = DataLoader(
		train_dataset, 
		batch_size=cfg.data.batch_size, 
		collate_fn=collate_fn,
		sampler=train_sampler,
		num_workers=cfg.data.num_workers,
		pin_memory=True,
		persistent_workers=True if cfg.data.num_workers > 0 else False
	)
	val_loader = DataLoader(
		val_dataset, 
		batch_size=cfg.data.batch_size, 
		collate_fn=collate_fn,
		sampler=val_sampler,
		num_workers=cfg.data.num_workers,
		pin_memory=True,
		persistent_workers=True if cfg.data.num_workers > 0 else False
	)
	
	return train_loader, val_loader, train_sampler, val_sampler


def get_deepspeed_config(cfg: DictConfig, world_size: int) -> Dict:
	"""Generate DeepSpeed configuration."""
	
	# Base configuration
	ds_config = {
		"train_batch_size": cfg.data.batch_size * cfg.trainer.gradient_accumulation_steps * world_size,
		"train_micro_batch_size_per_gpu": cfg.data.batch_size,
		"gradient_accumulation_steps": cfg.trainer.gradient_accumulation_steps,
		"steps_per_print": cfg.trainer.log_every_n_steps,
		"wall_clock_breakdown": False,
	}

	ds_config["optimizer"] = {
		"type": "AdamW",
		"params": {
			"lr": cfg.module.lr,
			"weight_decay": cfg.module.weight_decay,
			"betas": [0.9, 0.95],
			"eps": 1e-8
		}
	}
	
	# FP16 configuration
	if cfg.trainer.get("precision", "32") == "16-mixed":
		ds_config["fp16"] = {
			"enabled": True,
			"loss_scale": 0,
			"loss_scale_window": 1000,
			"hysteresis": 2,
			"min_loss_scale": 1
		}
	
	# BF16 configuration
	elif cfg.trainer.get("precision", "32") == "bf16":
		ds_config["bf16"] = {
			"enabled": True
		}
	
	# ZeRO optimization
	zero_stage = cfg.trainer.get("zero_stage", 2)
	ds_config["zero_optimization"] = {
		"stage": zero_stage,
		"overlap_comm": True,
		"contiguous_gradients": True,
		"reduce_bucket_size": 5e7,
	}
	
	if zero_stage == 3:
		ds_config["zero_optimization"].update({
			"stage3_prefetch_bucket_size": 5e7,
			"stage3_param_persistence_threshold": 1e5,
			"stage3_max_live_parameters": 1e9,
			"stage3_max_reuse_distance": 1e9,
		})
	
	return ds_config


def train_epoch(model_engine, training_module, train_loader, epoch, 
				writer, global_step, rank, cfg):
	"""Train for one epoch."""
	model_engine.train()
	
	total_loss = 0
	num_batches = 0
	epoch_metrics = {}
	
	# Progress bar (only on rank 0)
	if rank == 0:
		pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
	else:
		pbar = train_loader
	
	for batch_idx, batch in enumerate(pbar):
		# Compute loss and metrics
		loss, metrics = training_module.compute_loss_and_metrics(
			batch, 
			device=torch.cuda.current_device(),
			is_training=True
		)
		
		# Backward pass
		model_engine.backward(loss)
		model_engine.step()
		
		# Accumulate metrics
		total_loss += loss.item()
		num_batches += 1
		
		for key, value in metrics.items():
			if key not in epoch_metrics:
				epoch_metrics[key] = 0
			epoch_metrics[key] += value
		
		global_step += 1
		
		# Log to TensorBoard
		if global_step % cfg.trainer.log_every_n_steps == 0 and rank == 0:
			for key, value in metrics.items():
				writer.add_scalar(f'train/{key}', value, global_step)
			
			# Log learning rate
			lr = model_engine.optimizer.param_groups[0]['lr']
			writer.add_scalar('train/lr', lr, global_step)
		
		# Update progress bar
		if rank == 0:
			pbar.set_postfix({
				'loss': f"{loss.item():.4f}",
				'lr': f"{model_engine.optimizer.param_groups[0]['lr']:.2e}"
			})
	
	# Average metrics
	avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
	
	return global_step, avg_metrics


def validate(model_engine, training_module, val_loader, epoch, 
			writer, global_step, rank, cfg):
	"""Validate the model."""
	model_engine.eval()
	
	total_loss = 0
	num_batches = 0
	epoch_metrics = {}
	
	# Progress bar (only on rank 0)
	if rank == 0:
		pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
	else:
		pbar = val_loader
	
	with torch.no_grad():
		for batch in pbar:
			# Compute loss and metrics
			loss, metrics = training_module.compute_loss_and_metrics(
				batch,
				device=torch.cuda.current_device(),
				is_training=False
			)
			
			# Accumulate metrics
			total_loss += loss.item()
			num_batches += 1
			
			for key, value in metrics.items():
				if key not in epoch_metrics:
					epoch_metrics[key] = 0
				epoch_metrics[key] += value
			
			# Update progress bar
			if rank == 0:
				pbar.set_postfix({'loss': f"{loss.item():.4f}"})
	
	# Average metrics across all batches
	avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
	
	# Gather metrics from all ranks
	if dist.is_initialized():
		for key in avg_metrics:
			metric_tensor = torch.tensor(avg_metrics[key]).cuda()
			dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
			avg_metrics[key] = (metric_tensor / dist.get_world_size()).item()
	
	# Log to TensorBoard (rank 0 only)
	if rank == 0:
		for key, value in avg_metrics.items():
			writer.add_scalar(f'val/{key}', value, global_step)
	
	return avg_metrics


def main_worker(local_rank: int, cfg: DictConfig):
	"""Main training function for each process."""
	
	# Initialize distributed training
	deepspeed.init_distributed()
	
	# Get distributed info
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	
	# Set device
	torch.cuda.set_device(local_rank)
	device = torch.device(f'cuda:{local_rank}')
	
	# Print configuration (rank 0 only)
	if rank == 0:
		print("=" * 80)
		print("Training Configuration:")
		print(OmegaConf.to_yaml(cfg))
		print("=" * 80)
		print(f"World Size: {world_size}")
		print(f"Task: {cfg.task_name}")
		print("=" * 80)
	
	# Set random seed
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	
	# Get dataloaders
	train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(
		cfg, world_size, rank
	)
	
	# Calculate total training steps
	steps_per_epoch = len(train_loader)
	total_training_steps = steps_per_epoch * cfg.trainer.max_epochs

	module_args = OmegaConf.to_container(cfg.module, resolve=True)

	module_args["total_training_steps"] = total_training_steps
	if "_target_" in module_args:
		del module_args["_target_"]
	
	# Initialize training module
	if cfg.task_name == "pretrain":
		training_module = MLMTrainingModule(**module_args)
		model = training_module.model
	elif cfg.task_name == "finetune":
		training_module = QM9TrainingModule(**module_args)
		model = training_module.get_full_model()
	else:
		raise ValueError(f"Unknown task: {cfg.task_name}")
	
	# Get optimizer parameter groups
	param_groups = training_module.get_optimizer_groups(model)
	
	# DeepSpeed configuration
	ds_config = get_deepspeed_config(cfg, world_size)

	optimizer = torch.optim.AdamW(
		param_groups,
		lr=training_module.lr,
		weight_decay=training_module.weight_decay,
		betas=(0.9, 0.95),
		eps=1e-8
	)

	lr_scheduler = training_module.get_scheduler(optimizer)
	
	# Initialize DeepSpeed
	model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
		model=model,
		model_parameters=param_groups,
		optimizer=optimizer,
		lr_scheduler=lr_scheduler,
		config=ds_config
	)
	
	# Setup TensorBoard (rank 0 only)
	writer = None
	if rank == 0:
		log_dir = os.path.join("logs", cfg.trainer.logger_name)
		os.makedirs(log_dir, exist_ok=True)
		writer = SummaryWriter(log_dir)
		print(f"TensorBoard logs: {log_dir}")
	
	# Setup checkpointing directory
	checkpoint_dir = os.path.join("checkpoints", cfg.trainer.logger_name)
	if rank == 0:
		os.makedirs(checkpoint_dir, exist_ok=True)
		print(f"Checkpoints: {checkpoint_dir}")
	
	# Training loop
	global_step = 0
	best_val_loss = float('inf')
	
	if rank == 0:
		print("\n" + "=" * 80)
		print("Starting Training")
		print("=" * 80 + "\n")
	
	for epoch in range(cfg.trainer.max_epochs):
		epoch_start = time.time()
		
		# Set epoch for sampler (important for proper shuffling)
		train_sampler.set_epoch(epoch)
		
		# Train
		global_step, train_metrics = train_epoch(
			model_engine, training_module, train_loader, epoch,
			writer, global_step, rank, cfg
		)
		
		# Validate
		val_metrics = validate(
			model_engine, training_module, val_loader, epoch,
			writer, global_step, rank, cfg
		)
		
		epoch_time = time.time() - epoch_start
		
		# Print epoch summary (rank 0 only)
		if rank == 0:
			print(f"\n{'=' * 80}")
			print(f"Epoch {epoch+1}/{cfg.trainer.max_epochs} Complete")
			print(f"  Time: {epoch_time:.2f}s")
			print(f"  Train Loss: {train_metrics['loss']:.4f}")
			print(f"  Val Loss: {val_metrics['loss']:.4f}")
			
			# Print additional metrics
			for key, value in val_metrics.items():
				if key != 'loss':
					print(f"  Val {key}: {value:.4f}")
			
			print(f"{'=' * 80}\n")
		
		# Save checkpoint
		val_loss = val_metrics['loss']
		
		# Save last checkpoint
		if rank == 0:
			model_engine.save_checkpoint(
				checkpoint_dir,
				tag=f"epoch_{epoch+1}",
				client_state={'epoch': epoch+1, 'global_step': global_step}
			)
		
		# Save best checkpoint
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			if rank == 0:
				model_engine.save_checkpoint(
					checkpoint_dir,
					tag="best",
					client_state={
						'epoch': epoch+1, 
						'global_step': global_step,
						'best_val_loss': best_val_loss
					}
				)
				print(f"Saved best checkpoint (val_loss: {val_loss:.4f})\n")
		
		# Step scheduler
		if lr_scheduler is not None:
			lr_scheduler.step()
	
	# Cleanup
	if rank == 0:
		print("\n" + "=" * 80)
		print("Training Complete!")
		print(f"Best validation loss: {best_val_loss:.4f}")
		print("=" * 80)
		writer.close()
	
	dist.barrier()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
	"""Main entry point."""
	
	# Get local rank from environment (set by torchrun or deepspeed launcher)
	local_rank = int(os.environ.get('LOCAL_RANK', 0))
	
	# Add DeepSpeed-specific arguments
	import sys
	sys.argv.extend([
		'--local_rank', str(local_rank)
	])
	
	# Run training
	main_worker(local_rank, cfg)


if __name__ == "__main__":
	main()
