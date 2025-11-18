import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset

from data import BytePatchSMILESProcessor, MLMMapDataset, mlm_collate_fn
from lightning_module import MLMLightningModule

if __name__ == "__main__":
	
	# --- 1. Configuration ---
	BATCH_SIZE = 64
	NUM_WORKERS = 4
	MAX_EPOCHS = 10
	AUGMENT_PROB = 0.5
	MLM_PROB = 0.15
	MAX_BYTES_PER_ATOM = 8
	VOCAB_SIZE = 257 # 256 bytes + 1 padding
	
	# New Hyperparameters from 
	LEARNING_RATE = 1e-4
	WEIGHT_DECAY = 0.1
	WARMUP_RATIO = 0.1
	LABEL_SMOOTHING = 0.1
	
	# Model Config (V2 - MambaBiMamba)
	model_config = {
		# General 
		"vocab_size": VOCAB_SIZE,
		"max_bytes_per_atom": MAX_BYTES_PER_ATOM,
		"dropout": 0.1,
		
		# Local Mamba Config
		"patch_dim": 128,			# Width of the local encoder
		"local_mamba_layers": 4,	 # Depth of the local encoder
		"local_mamba_d_state": 16,   # State dim (d_state) for local Mamba
		"local_mamba_conv_kernel": 4,  # Conv kernel (d_conv) for local Mamba
		"local_mamba_expand": 2,	 # Expansion factor for local Mamba
		
		# Global BiMamba Config
		"d_model": 512,			  # Width of the global encoder
		"n_layers": 12,			  # Depth of the global encoder
		"global_mamba_d_state": 32,  # State dim (d_state) for global BiMamba
		"global_mamba_conv_kernel": 4, # Conv kernel (d_conv) for global BiMamba
		"global_mamba_expand": 2,	  # Expansion factor for global BiMamba
	}

	# --- 2. Load ZINC250k Dataset ---
	print("Loading ZINC250k dataset from Hugging Face...")
	try:
		dataset = load_dataset("yairschiff/zinc250k")
	except Exception as e:
		print(f"Failed to load dataset. Do you have internet access and 'datasets' installed?")
		print(f"Error: {e}")
		exit()

	# Split dataset (ZINC250k only has 'train')
	dataset_split = dataset['train'].train_test_split(test_size=0.01) # 1% for validation
	train_smiles = dataset_split['train']['smiles']
	val_smiles = dataset_split['test']['smiles']
	
	print(f"Loaded {len(train_smiles)} training SMILES and {len(val_smiles)} validation SMILES.")

	processor = BytePatchSMILESProcessor(
		vocab_size=VOCAB_SIZE,
		max_bytes_per_atom=MAX_BYTES_PER_ATOM,
	)
	
	train_dataset = MLMMapDataset(
		list_of_smiles=train_smiles,
		processor=processor,
		mlm_probability=MLM_PROB,
		augment_prob=AUGMENT_PROB,
		mask_token_prob=0.8,
		random_token_prob=0.1,
		keep_token_prob=0.1
	)
	
	val_dataset = MLMMapDataset(
		list_of_smiles=val_smiles,
		processor=processor,
		mlm_probability=MLM_PROB,
		augment_prob=0.0, # No augmentation for validation
		mask_token_prob=0.8,
		random_token_prob=0.1,
		keep_token_prob=0.1
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		collate_fn=mlm_collate_fn, # Use improved collate
		num_workers=NUM_WORKERS,
		shuffle=True,
		pin_memory=True,
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=BATCH_SIZE,
		collate_fn=mlm_collate_fn, # Use improved collate
		num_workers=NUM_WORKERS,
		shuffle=False,
		pin_memory=True,
	)
	
	steps_per_epoch = len(train_loader)
	total_training_steps = steps_per_epoch * MAX_EPOCHS
	print(f"Trainer: {steps_per_epoch} steps per epoch, {total_training_steps} total steps.")

	lightning_model = MLMLightningModule(
		model_config=model_config,
		architecture='mamba_bimamba',
		lr=LEARNING_RATE,
		weight_decay=WEIGHT_DECAY,
		total_training_steps=total_training_steps,
		warmup_ratio=WARMUP_RATIO,
		label_smoothing=LABEL_SMOOTHING
	)

	if torch.cuda.is_available():
		accelerator = 'gpu'
		devices = [0] 
		precision = '16-mixed'
	elif torch.backends.mps.is_available():
		accelerator = 'mps'
		devices = 1
		precision = 'bf16'
	else:
		accelerator = 'cpu'
		devices = 1
		precision = '32'

	print(f"Using accelerator: {accelerator}, precision: {precision}")

	trainer = pl.Trainer(
		max_epochs=MAX_EPOCHS,
		accelerator=accelerator,
		devices=devices,
		precision=precision,
		log_every_n_steps=50,
		enable_checkpointing=True,
		logger=pl.loggers.TensorBoardLogger("logs/", name="MambaBiMamba"),
	)

	print("Starting training with improved model and dataset...")
	trainer.fit(
		model=lightning_model,
		train_dataloaders=train_loader,
		val_dataloaders=val_loader
	)

	print("Training complete.")
