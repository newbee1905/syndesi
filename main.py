import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, Optional, List
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer

from data import (
	BytePatchSMILESProcessor, 
	MLMMapDataset, mlm_collate_fn,
	FinetuneMapDataset, finetune_collate_fn,
	ChemBERTaFinetuneDataset, chemberta_collate_fn
)
from modules import MLMLightningModule, QM9LightningModule

from mamba_ssm import Mamba


def get_dataloaders(cfg: DictConfig) -> (DataLoader, DataLoader):
	"""Initializes and returns train and validation dataloaders based on config."""
	
	print(f"Setting up dataloaders for task: {cfg.task_name}")
	
	if cfg.task_name == "pretrain":
		dataset = load_dataset(cfg.data.dataset_name)
		dataset_split = dataset['train'].train_test_split(test_size=0.01)
		train_smiles, val_smiles = dataset_split['train']['smiles'], dataset_split['test']['smiles']
		
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
		train_size = int((1.0 - cfg.data.val_split - cfg.data.test_split) * len(dataset))
		val_size = int(cfg.data.val_split * len(dataset))
		test_size = len(dataset) - train_size - val_size
		train_data, val_data, _ = random_split(dataset, [train_size, val_size, test_size])

		train_smiles = [x['smiles'] for x in train_data]
		train_targets = [x[cfg.data.target_column] for x in train_data]
		val_smiles = [x['smiles'] for x in val_data]
		val_targets = [x[cfg.data.target_column] for x in val_data]

		if cfg.model.architecture == "mamba_bimamba":
			print("Using Mamba-BiMamba Finetune dataset")
			processor = BytePatchSMILESProcessor(
				cfg.module.model_config.vocab_size, 
				cfg.module.model_config.max_bytes_per_atom
			)
			train_dataset = FinetuneMapDataset(train_smiles, train_targets, processor)
			val_dataset = FinetuneMapDataset(val_smiles, val_targets, processor)
			collate_fn = finetune_collate_fn
			
		elif cfg.model.architecture == "chemberta":
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
			raise ValueError(f"Unknown model architecture for fine-tuning: {cfg.model.architecture}")
	
	elif cfg.task_name == "finetune":
		dataset = load_dataset(cfg.data.dataset_name, split='train')
		
		# Filter out rows with missing SMILES or targets
		dataset = dataset.filter(lambda x: x.get('SMILES') is not None and x.get(cfg.data.target_column) is not None)
		
		# Split
		train_size = int((1.0 - cfg.data.val_split - cfg.data.test_split) * len(dataset))
		val_size = int(cfg.data.val_split * len(dataset))
		test_size = len(dataset) - train_size - val_size
		train_data, val_data, _ = random_split(dataset, [train_size, val_size, test_size])

		# Extract SMILES and targets (note: column is uppercase 'SMILES')
		train_smiles = [x['SMILES'] for x in train_data]
		train_targets = [float(x[cfg.data.target_column]) for x in train_data]
		val_smiles = [x['SMILES'] for x in val_data]
		val_targets = [float(x[cfg.data.target_column]) for x in val_data]

		if cfg.model.architecture == "mamba_bimamba":
			print("Using Mamba-BiMamba Finetune dataset")
			processor = BytePatchSMILESProcessor(
				cfg.module.model_config.vocab_size, 
				cfg.module.model_config.max_bytes_per_atom
			)
			train_dataset = FinetuneMapDataset(train_smiles, train_targets, processor)
			val_dataset = FinetuneMapDataset(val_smiles, val_targets, processor)
			collate_fn = finetune_collate_fn
			
		elif cfg.model.architecture == "chemberta":
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
			raise ValueError(f"Unknown model architecture for fine-tuning: {cfg.model.architecture}")
	
	else:
		raise ValueError(f"Unknown task_name: {cfg.task_name}")

	train_loader = DataLoader(
		train_dataset, batch_size=cfg.data.batch_size, collate_fn=collate_fn,
		num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True,
		persistent_workers=True if cfg.data.num_workers > 0 else False
	)
	val_loader = DataLoader(
		val_dataset, batch_size=cfg.data.batch_size, collate_fn=collate_fn,
		num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True,
		persistent_workers=True if cfg.data.num_workers > 0 else False
	)
	
	return train_loader, val_loader

def auto_detect_accelerator():
	"""Auto-detects accelerator and precision."""
	if torch.cuda.is_available():
		return 'gpu', '16-mixed'
	elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
		return 'mps', 'bf16'
	else:
		return 'cpu', '32'

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
	print("="*80)
	print("Starting new run with configuration:")
	print(OmegaConf.to_yaml(cfg))
	print("="*80)
	
	pl.seed_everything(42)

	train_loader, val_loader = get_dataloaders(cfg)
	default_accelerator, default_precision = auto_detect_accelerator()
	
	accelerator = cfg.trainer.get("accelerator", default_accelerator)
	precision = cfg.trainer.get("precision", default_precision)
	devices = cfg.trainer.get("devices", 1)
	
	print(f"Using accelerator: {accelerator}, precision: {precision}, devices: {devices}")
	
	# Logger
	logger = pl.loggers.TensorBoardLogger(
		"logs/", 
		name=cfg.trainer.logger_name,
	)
	
	# Checkpointing
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		monitor="val_loss", 
		mode="min", 
		save_last=True
	)
	
	trainer = pl.Trainer(
		max_epochs=cfg.trainer.max_epochs,
		accelerator=accelerator,
		devices=devices,
		precision=precision,
		log_every_n_steps=cfg.trainer.log_every_n_steps,
		logger=logger,
		callbacks=[checkpoint_callback]
	)

	print("Initializing Lightning Module...")
	
	# Calculate total steps and inject into config
	steps_per_epoch = len(train_loader)
	total_training_steps = steps_per_epoch * cfg.trainer.max_epochs
	# cfg.module.total_training_steps = total_training_steps
	cfg.module = OmegaConf.merge(
		cfg.module, 
		{"total_training_steps": total_training_steps}
	)

	# Instantiate the module specified in the config
	# This automatically calls either MLMLightningModule or QM9LightningModule
	lightning_model = hydra.utils.instantiate(cfg.module)

	print(f"Starting task: {cfg.task_name}")
	
	# `cfg.checkpoint_path` is used for resuming pre-training
	# `cfg.module.checkpoint_path` is used by QM9 module to load pre-trained weights
	ckpt_path = cfg.get("checkpoint_path", None)
	if ckpt_path:
		print(f"Resuming training from checkpoint: {ckpt_path}")

	trainer.fit(
		model=lightning_model,
		train_dataloaders=train_loader,
		val_dataloaders=val_loader,
		ckpt_path=ckpt_path 
	)
	
	print("Run complete.")
	print(f"Logs saved to: {logger.log_dir}")


if __name__ == "__main__":
	main()
