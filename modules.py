import torch
import torch.nn as nn
from typing import Dict, Optional

import pytorch_lightning as pl
from adam_atan2_pytorch import AdamAtan2
from transformers import get_linear_schedule_with_warmup

from model import MolecularMambaBiMamba, MeanPooling, CLSPooling

import abc

class AbsLightningModule(pl.LightningModule, abc.ABC):
	"""Abstract base class for Lightning Modules."""
	
	def __init__(self, lr, weight_decay, total_training_steps, warmup_ratio):
		super().__init__()
		self.save_hyperparameters(
			"lr", "weight_decay", "total_training_steps", "warmup_ratio"
		)
	
	@abc.abstractmethod
	def _common_step(self, batch: Dict, batch_idx: int, step_type: str) -> torch.Tensor:
		"""Subclasses must implement this."""
		raise NotImplementedError

	def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
		return self._common_step(batch, batch_idx, 'train')

	def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
		return self._common_step(batch, batch_idx, 'val')
	
	def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
		return self._common_step(batch, batch_idx, 'test')

	def configure_optimizers(self):
		"""Shared optimizer configuration."""
		decay_params = []
		no_decay_params = []
		
		for name, param in self.model.named_parameters():
			if not param.requires_grad:
				continue
			if (param.dim() == 1 or "bias" in name or "norm" in name or 
				"embedding" in name or "layer_scale" in name):
				no_decay_params.append(param)
			else:
				decay_params.append(param)
				
		optimizer_grouped_parameters = [
			{'params': decay_params, 'weight_decay': self.hparams.weight_decay},
			{'params': no_decay_params, 'weight_decay': 0.0}
		]
		
		optimizer = AdamAtan2(
			optimizer_grouped_parameters, 
			lr=self.hparams.lr,
			betas=(0.9, 0.98), 
		)
		
		num_warmup_steps = int(
			self.hparams.total_training_steps * self.hparams.warmup_ratio
		)

		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=num_warmup_steps,
			num_training_steps=self.hparams.total_training_steps
		)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler, 
				"interval": "step", 
				"frequency": 1,
			},
		}

class MLMLightningModule(AbsLightningModule):
	"""Lightning Module for MLM Pre-training."""

	def __init__(
		self,
		model_config: Dict,
		architecture: str = 'mamba_bimamba',
		lr: float = 1e-4,
		weight_decay: float = 0.01,
		total_training_steps: int = 100000,
		warmup_ratio: float = 0.05,
		label_smoothing: float = 0.1,
	):
		super().__init__(lr, weight_decay, total_training_steps, warmup_ratio)
		self.save_hyperparameters("model_config", "architecture", "label_smoothing")

		if architecture == 'mamba_bimamba':
			self.model = MolecularMambaBiMamba(**model_config)
		else:
			raise ValueError(f"Unknown architecture: {architecture}")
			
		self.architecture = architecture
		self.criterion = nn.CrossEntropyLoss(
			ignore_index=-100,
			label_smoothing=label_smoothing
		)
	
	def forward(self, **batch) -> Dict[str, torch.Tensor]:
		return self.model(**batch)

	def _common_step(self, batch: Dict, batch_idx: int, step_type: str) -> torch.Tensor:
		outputs = self(**batch)
		
		logits = outputs['logits']
		logits_flat = logits.view(-1, logits.size(-1))
		labels_flat = batch['atom_labels'].view(-1)
		
		loss = self.criterion(logits_flat, labels_flat)
		
		with torch.no_grad():
			mask_positions = labels_flat != -100
			if mask_positions.sum() > 0:
				predictions = logits_flat.argmax(dim=-1)
				correct = (predictions == labels_flat) & mask_positions
				accuracy = correct.sum().float() / mask_positions.sum()
				
				self.log(f'{step_type}_byte_accuracy', accuracy, 
						 on_step=False, on_epoch=True, 
						 prog_bar=True, logger=True, sync_dist=True)
				
				perplexity = torch.exp(loss)
				self.log(f'{step_type}_perplexity', perplexity,
						 on_step=False, on_epoch=True,
						 prog_bar=False, logger=True, sync_dist=True)
		
		self.log(f'{step_type}_loss', loss, 
				 on_step=(step_type == 'train'), on_epoch=True, 
				 prog_bar=True, logger=True, sync_dist=True)
		
		return loss

class QM9LightningModule(AbsLightningModule):
	"""
	Lightning Module for fine-tuning on QM9.
	Supports loading our pre-trained model or a ChemBERTa baseline.
	"""
	def __init__(
		self,
		architecture: str,
		d_model: int,
		lr: float = 1e-5,
		weight_decay: float = 0.01,
		total_training_steps: int = 10000,
		warmup_ratio: float = 0.1,
		model_config: Optional[Dict] = None,
		checkpoint_path: Optional[str] = None,
		chemberta_model: str = "DeepChem/ChemBERTa-77M-MTR"
	):
		super().__init__(lr, weight_decay, total_training_steps, warmup_ratio)
		self.save_hyperparameters("architecture", "d_model")

		if architecture == 'mamba_bimamba':
			if model_config is None or checkpoint_path is None:
				raise ValueError("model_config and checkpoint_path must be provided for mamba_bimamba")
			
			# Load our pre-trained Mamba model
			# We load the weights from the pre-trained module
			pretrain_module = MLMLightningModule.load_from_checkpoint(
				checkpoint_path, model_config=model_config
			)
			self.model = pretrain_module.model

			# We only need the backbone, not the MLM head
			del self.model.mlm_head
			
			self.pooler = MeanPooling()
			self.regressor = nn.Linear(d_model, 1)

		elif architecture == 'chemberta':
			# Load a pre-trained ChemBERTa model
			self.model = AutoModel.from_pretrained(chemberta_model)
			# Update d_model to match ChemBERTa
			self.hparams.d_model = self.model.config.hidden_size
			
			self.pooler = CLSPooling()
			self.regressor = nn.Linear(self.hparams.d_model, 1)
		
		else:
			raise ValueError(f"Unknown architecture: {architecture}")
			
		self.criterion = nn.MSELoss()
		self.mae = nn.L1Loss()

	def forward(self, batch: Dict) -> torch.Tensor:
		if self.hparams.architecture == 'mamba_bimamba':
			# Our model expects 'byte_ids'
			outputs = self.model.bimamba_backbone(
				self.model.get_patch_embeddings(batch['byte_ids']),
				attention_mask=batch['bert_attention_mask']
			)
			# Pool using our masked mean pooling
			pooled_output = self.pooler(outputs, batch['atom_attention_mask'])
		
		else: # chemberta
			# ChemBERTa expects 'input_ids', 'attention_mask'
			outputs = self.model(
				input_ids=batch['input_ids'],
				attention_mask=batch['attention_mask']
			)
			# Pool using CLS token
			pooled_output = self.pooler(outputs.last_hidden_state)
			
		return self.regressor(pooled_output).squeeze(-1) # (B,)

	def _common_step(self, batch: Dict, batch_idx: int, step_type: str) -> torch.Tensor:
		targets = batch['targets']
		predictions = self(batch)
		
		loss = self.criterion(predictions, targets) # MSE Loss
		
		self.log(
			f'{step_type}_loss', loss, 
			on_step=False, on_epoch=True, 
			prog_bar=True, logger=True, sync_dist=True,
		)
		
		if step_type in ('val', 'test'):
			mae = self.mae(predictions, targets)
			self.log(
				f'{step_type}_mae', mae, 
				on_step=False, on_epoch=True, 
				prog_bar=True, logger=True, sync_dist=True,
			)
			
		return loss
