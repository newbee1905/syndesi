import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import get_linear_schedule_with_warmup, AutoModel

from model import MolecularMambaBiMamba, MeanPooling, CLSPooling


class BaseTrainingModule:
	"""Base class for training modules."""
	
	def __init__(self, lr, weight_decay, total_training_steps, warmup_ratio):
		self.lr = lr
		self.weight_decay = weight_decay
		self.total_training_steps = total_training_steps
		self.warmup_ratio = warmup_ratio
		self.global_step = 0
	
	def get_optimizer_groups(self, model):
		"""Get parameter groups with and without weight decay."""
		decay_params = []
		no_decay_params = []
		
		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			if (param.dim() == 1 or "bias" in name or "norm" in name or 
				"embedding" in name or "layer_scale" in name):
				no_decay_params.append(param)
			else:
				decay_params.append(param)
				
		return [
			{'params': decay_params, 'weight_decay': self.weight_decay},
			{'params': no_decay_params, 'weight_decay': 0.0}
		]
	
	def get_scheduler(self, optimizer):
		"""Get learning rate scheduler with warmup."""
		num_warmup_steps = int(self.total_training_steps * self.warmup_ratio)
		
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=num_warmup_steps,
			num_training_steps=self.total_training_steps
		)
		
		return scheduler
	
	def save_checkpoint(self, model_engine, optimizer, scheduler, epoch, 
					   global_step, checkpoint_path):
		"""Save checkpoint including model, optimizer, and scheduler state."""
		# DeepSpeed handles distributed checkpoint saving
		model_engine.save_checkpoint(checkpoint_path, tag=f"epoch_{epoch}")
		
		# Save additional metadata
		metadata = {
			'epoch': epoch,
			'global_step': global_step,
			'lr': self.lr,
			'weight_decay': self.weight_decay,
			'total_training_steps': self.total_training_steps,
			'warmup_ratio': self.warmup_ratio,
		}
		
		if torch.distributed.get_rank() == 0:
			metadata_path = f"{checkpoint_path}/epoch_{epoch}/metadata.pt"
			torch.save(metadata, metadata_path)
	
	def load_checkpoint(self, model_engine, checkpoint_path, tag):
		"""Load checkpoint."""
		_, client_state = model_engine.load_checkpoint(checkpoint_path, tag=tag)
		return client_state


class MLMTrainingModule(BaseTrainingModule):
	"""Module for MLM Pre-training."""

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
		
		self.model_config = model_config
		self.architecture = architecture
		self.label_smoothing = label_smoothing

		if architecture == 'mamba_bimamba':
			self.model = MolecularMambaBiMamba(**model_config)
		else:
			raise ValueError(f"Unknown architecture: {architecture}")
		
		self.criterion = nn.CrossEntropyLoss(
			ignore_index=-100,
			label_smoothing=label_smoothing
		)
	
	def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
		"""Forward pass through model."""
		return self.model(**batch)

	def compute_loss_and_metrics(self, batch: Dict, device: torch.device, is_training: bool = True) -> Dict:
		"""Compute loss and metrics for a batch."""
		# Move batch to device
		batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
				for k, v in batch.items()}
		
		atom_labels = batch['atom_labels']
		
		# Forward pass
		outputs = self.forward(batch)
		
		logits = outputs['logits']  # (B, N, P, V)
		logits_flat = logits.view(-1, logits.size(-1))
		labels_flat = atom_labels.view(-1)

		# Calculate loss
		loss = self.criterion(logits_flat, labels_flat)
		
		# Calculate metrics
		metrics = {'loss': loss.item()}
		
		with torch.no_grad():
			non_padding_mask = (labels_flat != -100) & (labels_flat != 0)
			if non_padding_mask.sum() > 0:
				predictions = logits_flat.argmax(dim=-1)
				padding_correct = (predictions == labels_flat) & non_padding_mask
				accuracy = padding_correct.sum().float() / non_padding_mask.sum()
				
				metrics['byte_accuracy'] = accuracy.item()
				metrics['perplexity'] = torch.exp(loss).item()
		
		return loss, metrics


class QM9TrainingModule(BaseTrainingModule):
	"""Module for fine-tuning on QM9."""
	
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
		chemberta_model: str = "DeepChem/ChemBERTa-77M-MTR",
		normalize_targets: bool = True,
		target_mean: Optional[float] = None,
		target_std: Optional[float] = None,
	):
		super().__init__(lr, weight_decay, total_training_steps, warmup_ratio)
		
		self.architecture = architecture
		self.d_model = d_model
		self.normalize_targets = normalize_targets
		self.checkpoint_path = checkpoint_path
		self.chemberta_model = chemberta_model
		
		# Initialize model
		if architecture == 'mamba_bimamba':
			if model_config is None or checkpoint_path is None:
				raise ValueError("model_config and checkpoint_path required for mamba_bimamba")
			
			# Load pre-trained model
			print(f"Loading pre-trained Mamba model from {checkpoint_path}")
			pretrain_module = MLMTrainingModule(
				model_config=model_config,
				architecture=architecture
			)
			
			# Load checkpoint weights
			checkpoint = torch.load(checkpoint_path, map_location='cpu')
			if 'state_dict' in checkpoint:
				pretrain_module.model.load_state_dict(checkpoint['state_dict'])
			else:
				pretrain_module.model.load_state_dict(checkpoint)
			
			self.model = pretrain_module.model
			
			# Remove MLM head
			if hasattr(self.model, 'mlm_head'):
				del self.model.mlm_head
			
			self.pooler = MeanPooling()
			self.regressor = nn.Sequential(
				nn.Linear(d_model, d_model // 2),
				nn.SiLU(),
				nn.Dropout(0.1),
				nn.Linear(d_model // 2, 1),
			)

		elif architecture == 'chemberta':
			self.model = AutoModel.from_pretrained(chemberta_model)
			self.d_model = self.model.config.hidden_size
			
			self.pooler = CLSPooling()
			self.regressor = nn.Sequential(
				nn.Linear(self.d_model, self.d_model // 2),
				nn.SiLU(),
				nn.Dropout(0.1),
				nn.Linear(self.d_model // 2, 1),
			)
		else:
			raise ValueError(f"Unknown architecture: {architecture}")

		# Initialize regressor weights
		for module in self.regressor.modules():
			if isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
		
		self.criterion = nn.MSELoss()
		self.mae = nn.L1Loss()
		
		# Target normalization
		self._target_mean = target_mean
		self._target_std = target_std
		self._stats_computed = (target_mean is not None and target_std is not None)

	def _compute_statistics(self, targets: torch.Tensor):
		"""Compute mean and std from first training batch."""
		if not self._stats_computed:
			self._target_mean = targets.mean().item()
			self._target_std = targets.std().item()
			self._stats_computed = True

			if torch.distributed.get_rank() == 0:
				print(f"\n{'='*60}")
				print(f"Target Statistics Computed:")
				print(f"Mean: {self._target_mean:.4f}")
				print(f"Std:  {self._target_std:.4f}")
				print(f"{'='*60}\n")

	def normalize_targets_fn(self, targets: torch.Tensor) -> torch.Tensor:
		"""Normalize targets to zero mean and unit variance."""
		if not self.normalize_targets or not self._stats_computed:
			return targets
		return (targets - self._target_mean) / (self._target_std + 1e-8)
	
	def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
		"""Convert normalized predictions back to original scale."""
		if not self.normalize_targets or not self._stats_computed:
			return predictions
		return predictions * (self._target_std + 1e-8) + self._target_mean

	def forward(self, batch: Dict, device: torch.device) -> torch.Tensor:
		"""Forward pass."""
		if self.architecture == 'mamba_bimamba':
			byte_ids = batch['byte_ids'].to(device)
			bert_attention_mask = batch['bert_attention_mask'].to(device)
			atom_attention_mask = batch['atom_attention_mask'].to(device)
			
			outputs = self.model.bimamba_backbone(
				self.model.get_patch_embeddings(byte_ids),
				attention_mask=bert_attention_mask
			)
			pooled_output = self.pooler(outputs, atom_attention_mask)
		
		else:  # chemberta
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			
			outputs = self.model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)
			pooled_output = self.pooler(outputs.last_hidden_state)
		
		return self.regressor(pooled_output).squeeze(-1)

	def compute_loss_and_metrics(self, batch: Dict, device: torch.device, 
								 is_training: bool = True) -> tuple:
		"""Compute loss and metrics for a batch."""
		targets = batch['targets'].to(device)
		
		# Compute statistics on first batch
		if is_training and not self._stats_computed:
			self._compute_statistics(targets)
		
		# Normalize targets if needed
		if self.normalize_targets and self._stats_computed:
			targets_normalized = self.normalize_targets_fn(targets)
		else:
			targets_normalized = targets
		
		# Forward pass
		predictions = self.forward(batch, device)
		
		# Loss (on normalized targets)
		loss = self.criterion(predictions, targets_normalized)
		
		# Metrics
		metrics = {'loss': loss.item()}
		
		# Additional metrics for validation
		if not is_training:
			with torch.no_grad():
				if self.normalize_targets and self._stats_computed:
					predictions_original = self.denormalize_predictions(predictions)
					targets_original = targets
				else:
					predictions_original = predictions
					targets_original = targets
				
				mae = self.mae(predictions_original, targets_original)
				mse = self.criterion(predictions_original, targets_original)
				rmse = torch.sqrt(mse)
				relative_error = mae / (targets_original.abs().mean() + 1e-8)
				
				metrics['mae'] = mae.item()
				metrics['rmse'] = rmse.item()
				metrics['relative_error'] = relative_error.item()
		
		return loss, metrics
	
	def get_full_model(self):
		"""Return full model including backbone and regressor."""
		class FullModel(nn.Module):
			def __init__(self, backbone, pooler, regressor):
				super().__init__()
				self.backbone = backbone
				self.pooler = pooler
				self.regressor = regressor
		
		return FullModel(self.model, self.pooler, self.regressor)
