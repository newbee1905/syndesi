import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from mamba_ssm import Mamba2

# MODEL (Mamba-Local + BiMamba-Global)

# RMSNorm from gemma
# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class RMSNorm(nn.Module):
	def __init__(
		self,
		dim: int,
		eps: float = 1e-6,
		add_unit_offset: bool = True,
	):
		super().__init__()

		self.register_buffer("eps", torch.tensor(float(eps)))
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(dim))

	def forward(self, x):
		# Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
		# See https://github.com/huggingface/transformers/pull/29402
		# This use x as float32 instead

		x_fp32 = x.float()

		variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
		out = x_fp32 * torch.rsqrt(variance + self.eps)
		out = out.to(self.weight.dtype)

		if self.add_unit_offset:
			out = out * (1 + self.weight)
		else:
			out = out * self.weight

		return out


class BiMambaLayer(nn.Module):
	"""Bidirectional Mamba Layer"""
	def __init__(
		self, 
		d_model: int, 
		d_state: int = 16, 
		d_conv: int = 4, 
		expand: int = 2, 
		dropout: float = 0.1
	):
		super().__init__()
		
		self.mamba_fwd = Mamba2(
			d_model=d_model, d_state=d_state, d_conv=d_conv,
			expand=expand, 
			rmsnorm=True,
		)
		self.mamba_bwd = Mamba2(
			d_model=d_model, d_state=d_state, d_conv=d_conv,
			expand=expand,
			rmsnorm=True,
		)
		
		self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

	def forward(
		self, 
		x: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None # Not used
	) -> torch.Tensor:
		
		h_fwd = self.mamba_fwd(x)
		
		x_norm_rev = torch.flip(x, dims=[1])
		h_bwd_rev = self.mamba_bwd(x_norm_rev)
		h_bwd = torch.flip(h_bwd_rev, dims=[1])
		
		h_bi = torch.cat([h_fwd, h_bwd], dim=-1)
		mamba_output = self.out_proj(h_bi)
		
		x = x + self.dropout(mamba_output) * self.layer_scale
		return x

class LocalMambaLayer(nn.Module):
	"""
	Unidirectional Mamba Layer
	Used to stack the local Mamba encoder.
	"""
	def __init__(
		self, 
		d_model: int, 
		d_state: int = 16, 
		d_conv: int = 4, 
		expand: int = 2, 
		dropout: float = 0.1
	):
		super().__init__()
		self.mamba = Mamba2(
			d_model=d_model,
			d_state=d_state,
			d_conv=d_conv,
			expand=expand,
			rmsnorm=True,
		)
		self.dropout = nn.Dropout(dropout)
		self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		mamba_output = self.mamba(x)
		x = x + self.dropout(mamba_output) * self.layer_scale

		return x


class BiMambaEncoder(nn.Module):
	"""Stack of BiMamba layers"""
	def __init__(
		self, d_model: int, n_layers: int, d_state: int = 16, 
		d_conv: int = 4, expand: int = 2, dropout=0.1,
	):
		super().__init__()
		self.layer = nn.ModuleList([
			BiMambaLayer(
				d_model=d_model, d_state=d_state, d_conv=d_conv,
				expand=expand, dropout=dropout
			) for _ in range(n_layers)
		])

	def forward(
		self, hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states)
		return hidden_states

class BiMambaBackbone(nn.Module):
	"""Bi-Mamba model on top of Mamba patches"""
	def __init__(
		self, patch_dim: int, d_model: int, n_layers: int,
		d_state: int = 16, d_conv: int = 4, expand: int = 2, 
		dropout: float = 0.1,
	):
		super().__init__()
		self.patch_projection = nn.Linear(patch_dim, d_model)
		self.encoder = BiMambaEncoder(
			d_model=d_model, n_layers=n_layers, d_state=d_state,
			d_conv=d_conv, expand=expand, dropout=dropout,
		)
		self.final_norm = RMSNorm(d_model)

	def forward(
		self, patch_embeddings: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		x = self.patch_projection(patch_embeddings)
		x = self.encoder(x, attention_mask)

		x = self.final_norm(x)

		return x

class MLMHead(nn.Module):
	"""Enhanced MLM prediction head with intermediate projection"""
	def __init__(self, d_model: int, vocab_size: int, max_bytes_per_atom: int):
		super().__init__()
		# Add intermediate layer for better capacity
		self.dense1 = nn.Linear(d_model, d_model * 2)
		self.dense2 = nn.Linear(d_model, d_model)
		self.norm = RMSNorm(d_model)
		self.dropout = nn.Dropout(0.1)
		self.decoder = nn.Linear(d_model, vocab_size * max_bytes_per_atom)
		self.vocab_size = vocab_size
		self.max_bytes_per_atom = max_bytes_per_atom

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# GLU variant
		gate, x = self.dense1(x).chunk(2, dim=-1)
		x = F.silu(gate) * x

		x = self.dense2(x)
		x = self.dropout(x)
		x = self.norm(x)

		logits = self.decoder(x)
		
		bsz, n_atoms, _ = logits.shape
		logits = logits.view(
			bsz, n_atoms, self.max_bytes_per_atom, self.vocab_size
		)
		return logits

class MolecularMambaBiMamba(nn.Module):
	"""
	Enhanced V2 model with BERT-style masking strategies
	"""
	def __init__(
		self, vocab_size: int = 257, patch_dim: int = 128,
		local_mamba_d_state: int = 16, local_mamba_conv_kernel: int = 4,
		local_mamba_expand: int = 2, local_mamba_layers: int = 4,
		d_model: int = 256, global_mamba_d_state: int = 16,
		global_mamba_conv_kernel: int = 4, global_mamba_expand: int = 2,
		n_layers: int = 6, dropout: float = 0.1,
		max_bytes_per_atom: int = 8,
		**kwargs
	):
		super().__init__()
		self.vocab_size = vocab_size
		self.patch_dim = patch_dim
		self.max_bytes_per_atom = max_bytes_per_atom
		self.d_model = d_model

		self.byte_embedding = nn.Embedding(vocab_size, patch_dim)

		# Local Mamba encoder
		local_mamba_stack = []
		for _ in range(local_mamba_layers):
			local_mamba_stack.append(
				LocalMambaLayer(
					d_model=patch_dim,
					d_state=local_mamba_d_state,
					d_conv=local_mamba_conv_kernel,
					expand=local_mamba_expand,
					dropout=dropout
				)
			)
		self.local_mamba = nn.Sequential(*local_mamba_stack)

		self.bimamba_backbone = BiMambaBackbone(
			patch_dim=patch_dim, d_model=d_model, n_layers=n_layers,
			d_state=global_mamba_d_state, d_conv=global_mamba_conv_kernel,
			expand=global_mamba_expand, dropout=dropout,
		)

		self.mlm_head = MLMHead(d_model, vocab_size, max_bytes_per_atom)
		
		self.mask_patch_embedding = nn.Parameter(torch.randn(patch_dim))
		# Random patch embeddings will be sampled from normal distribution

	def get_patch_embeddings(self, byte_ids: torch.Tensor) -> torch.Tensor:
		B, L = byte_ids.shape
		P = self.max_bytes_per_atom
		N = L // P 
		
		x = self.byte_embedding(byte_ids)
		x = x.view(B, N, P, -1)
		x_patched = x.reshape(B * N, P, -1)
		mamba_out = self.local_mamba(x_patched)
		patch_vecs = mamba_out[:, -1, :]
		patch_seq = patch_vecs.view(B, N, -1)
		
		return patch_seq

	def apply_masking_with_strategy(
		self, 
		patch_embeddings: torch.Tensor, 
		atom_mask: torch.Tensor,
		mask_strategy: torch.Tensor
	) -> torch.Tensor:
		"""
		Apply BERT-style masking with three strategies:
		0 = use [MASK] token
		1 = use random patch
		2 = keep original patch
		"""
		patch_seq_masked = patch_embeddings.clone()
		
		# Get positions where we need to mask
		mask_positions = atom_mask.bool()
		
		# Apply [MASK] token where strategy == 0
		use_mask_token = mask_positions & (mask_strategy == 0)
		patch_seq_masked[use_mask_token] = self.mask_patch_embedding.to(patch_seq_masked.dtype)
		
		# Apply random patch where strategy == 1
		use_random = mask_positions & (mask_strategy == 1)
		if use_random.any():
			random_patches = torch.randn_like(patch_seq_masked[use_random])
			patch_seq_masked[use_random] = random_patches
		
		# Keep original where strategy == 2 (do nothing)
		
		return patch_seq_masked

	def forward(
		self,
		byte_ids: torch.Tensor,
		bert_attention_mask: Optional[torch.Tensor] = None,
		atom_mask: Optional[torch.Tensor] = None,
		mask_strategy: Optional[torch.Tensor] = None,
		is_atom_patch: Optional[torch.Tensor] = None,
		**kwargs
	) -> Dict[str, torch.Tensor]:
		
		patch_embeddings = self.get_patch_embeddings(byte_ids)
		original_patch_embeddings = patch_embeddings.clone()
		
		if atom_mask is not None and mask_strategy is not None:
			patch_embeddings_input = self.apply_masking_with_strategy(
				patch_embeddings, atom_mask, mask_strategy
			)
		elif atom_mask is not None:
			# Fallback to old masking (always use [MASK] token)
			patch_embeddings_input = patch_embeddings.clone()
			patch_embeddings_input[atom_mask] = self.mask_patch_embedding.to(patch_embeddings_input.dtype)
		else:
			patch_embeddings_input = patch_embeddings

		hidden_states = self.bimamba_backbone(
			patch_embeddings_input, attention_mask=bert_attention_mask
		)

		logits = self.mlm_head(hidden_states)

		return {
			'logits': logits,
			'patch_embeddings': original_patch_embeddings,
			'hidden_states': hidden_states,
			'is_atom_patch': is_atom_patch,
		}

class MeanPooling(nn.Module):
	"""Applies masked mean pooling to patch embeddings."""
	def __init__(self):
		super().__init__()
	
	def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			hidden_states (B, N, D): Patch embeddings.
			attention_mask (B, N): Boolean or binary mask (1 for real, 0 for pad).
		"""
		mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
		masked_sum = (hidden_states * mask_expanded).sum(dim=1)
		num_real_tokens = mask_expanded.sum(dim=1)
		# Avoid division by zero for empty sequences
		num_real_tokens = torch.clamp(num_real_tokens, min=1e-9)
		
		mean_pooled = masked_sum / num_real_tokens
		return mean_pooled

class CLSPooling(nn.Module):
	"""Selects the [CLS] token embedding."""
	def __init__(self):
		super().__init__()

	def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
		return hidden_states[:, 0] # (B, L, D) -> (B, D)
