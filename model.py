import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

import math
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

class Dense(nn.Module):
	"""
	Linear layer with optional activation and GLU support.
	Used to replicate PaiNN logic without external dependencies.
	"""
	def __init__(self, in_features, out_features, activation=None, bias=True, glu_variant=False):
		super().__init__()
		self.activation = activation
		self.glu_variant = glu_variant
		self.linear = nn.Linear(in_features, out_features * 2 if glu_variant else out_features, bias=bias)

	def forward(self, x):
		out = self.linear(x)
		if self.glu_variant:
			x, gate = torch.chunk(out, 2, dim=-1)
			if self.activation is not None:
				x = self.activation(x)
			out = x * torch.sigmoid(gate)
		else:
			if self.activation is not None:
				out = self.activation(out)
		return out

class RadialBasis(nn.Module):
	def __init__(self, n_rbf: int, cutoff: float):
		super().__init__()
		self.n_rbf = n_rbf
		self.cutoff = cutoff
		self.frequencies = nn.Parameter(torch.Tensor(n_rbf))
		self.reset_parameters()

	def reset_parameters(self):
		with torch.no_grad():
			self.frequencies.data = math.pi * torch.arange(1, self.n_rbf + 1).float()

	def forward(self, d: torch.Tensor) -> torch.Tensor:
		# d: (B, N, N)
		d_scaled = d / self.cutoff
		d_scaled = torch.clamp(d_scaled, max=1.0)

		# Cosine cutoff
		cutoff_fn = 0.5 * (torch.cos(math.pi * d_scaled) + 1.0) * (d_scaled < 1.0).float()

		# RBF expansion
		val = torch.sin(self.frequencies.view(1,1,1,-1) * d_scaled.unsqueeze(-1)) 
		return cutoff_fn.unsqueeze(-1) * val

class PaiNNInteraction(nn.Module):
	"""
	PaiNN Interaction Block (Dense implementation for B,N,D tensors).
	Updates scalar (s) and vector (v) features based on neighbors.
	"""
	def __init__(self, n_atom_basis: int, activation: callable, glu_variant: bool):
		super().__init__()
		self.n_atom_basis = n_atom_basis
		self.norm = RMSNorm(n_atom_basis)
		self.interatomic_context_net = nn.Sequential(
			Dense(n_atom_basis, n_atom_basis, activation=activation, glu_variant=glu_variant),
			Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
		)

	def forward(self, q, mu, Wij, dir_ij, mask_ij):
		# q: (B, N, D)
		# mu: (B, N, 3, D)
		# Wij: (B, N, N, 3*D)
		# dir_ij: (B, N, N, 3)
		
		# Scalar to messages
		x = self.interatomic_context_net(self.norm(q))
		xj = x.unsqueeze(1) # Broadcast as neighbors (B, 1, N, 3D)
		
		# Filter
		x_inter = Wij * xj * mask_ij.unsqueeze(-1) # (B, N, N, 3D)

		dq, dmuR, dmumu = torch.split(x_inter, self.n_atom_basis, dim=-1)
		
		# Aggregate Scalar (Sum over j)
		dq = dq.sum(dim=2) # (B, N, D)
		
		# Aggregate Vector
		muj = mu.unsqueeze(1) # (B, 1, N, 3, D)
		
		# dmuR * dir_ij -> (B, N, N, D, 3)
		term1 = dmuR.unsqueeze(-1) * dir_ij.unsqueeze(3)
		
		# dmumu * muj -> (B, N, N, D, 3) (after permute)
		# muj needs permute to (B, 1, N, D, 3)
		term2 = dmumu.unsqueeze(-1) * muj.permute(0, 1, 2, 4, 3)
		
		dmu_combined = term1 + term2
		dmu = dmu_combined.sum(dim=2) # Sum over j -> (B, N, D, 3)
		dmu = dmu.permute(0, 1, 3, 2) # -> (B, N, 3, D)

		return q + dq, mu + dmu

class PaiNNMixing(nn.Module):
	"""
	PaiNN Mixing Block.
	Mixes vector features into scalar features locally (intra-atomic).
	This is CRITICAL for passing geometric info (v) into Mamba (which sees only s).
	"""
	def __init__(self, n_atom_basis: int, activation: callable, epsilon: float = 1e-6, glu_variant: bool = False):
		super().__init__()
		self.n_atom_basis = n_atom_basis
		self.norm = RMSNorm(n_atom_basis)
		self.intraatomic_context_net = nn.Sequential(
			Dense(2 * n_atom_basis, n_atom_basis, activation=activation, glu_variant=glu_variant),
			Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
		)
		self.mu_channel_mix = Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)
		self.epsilon = epsilon

	def forward(self, q, mu):
		# q: (B, N, D)
		# mu: (B, N, 3, D)
		
		mu_mix = self.mu_channel_mix(mu)
		mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
		
		# Vector norm
		mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=2, keepdim=True) + self.epsilon).squeeze(2)

		# Context for scalar update
		ctx = torch.cat([self.norm(q), mu_Vn], dim=-1)
		x = self.intraatomic_context_net(ctx)
		
		dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)

		# Update vectors
		dmu_intra = dmu_intra.unsqueeze(2) * mu_W
		
		# Update scalars (via dot product of vector channels)
		dot_prod = torch.sum(mu_V * mu_W, dim=2)
		dqmu_intra = dqmu_intra * dot_prod

		return q + dq_intra + dqmu_intra, mu + dmu_intra

class PaiNNBlock(nn.Module):
	"""
	PaiNN Block with Hydrogen Reinforcement.
	Mixes Mamba scalar features (heavy atoms) with learnable H-embeddings,
	runs PaiNN, and returns updated heavy atom features and persistent vectors.
	"""
	def __init__(self, d_model: int, n_rbf: int, activation=F.silu):
		super().__init__()
		self.filter_net = Dense(n_rbf, 3 * d_model, activation=None)
		self.interaction = PaiNNInteraction(d_model, activation=activation, glu_variant=True)
		self.mixing = PaiNNMixing(d_model, activation=activation, glu_variant=True)
		
		# Learnable embedding for Hydrogen atoms
		self.h_atom_embedding = nn.Parameter(torch.randn(d_model))

	def forward(
		self, 
		s_heavy: torch.Tensor, 
		v_all: torch.Tensor, 
		rbf: torch.Tensor, 
		dir_ij: torch.Tensor, 
		mask_ij: torch.Tensor,
		is_hydrogen: torch.Tensor
	):
		"""
		s_heavy: (B, N_heavy_padded, D) - Scalar features from Mamba
		v_all:   (B, N_all_padded, 3, D) - Persistent vector features (Heavy + H)
		is_hydrogen: (B, N_all_padded)   - Mask indicating H atoms
		"""
		B, N_all, _, D = v_all.shape
		
		# Construct Full Scalar Features (s_all)
		s_all = torch.zeros(B, N_all, s_heavy.shape[-1], device=s_heavy.device, dtype=s_heavy.dtype)
		h_emb = self.h_atom_embedding.view(1, 1, -1).expand(B, N_all, -1)
		
		# Pad s_heavy to N_all length
		s_heavy_padded = F.pad(s_heavy, (0, 0, 0, N_all - s_heavy.shape[1]))
		
		# Fill H slots
		s_all = torch.where(is_hydrogen.unsqueeze(-1), h_emb, s_heavy_padded)

		# Run PaiNN
		Wij = self.filter_net(rbf) 
		s_out, v_out = self.interaction(s_all, v_all, Wij, dir_ij, mask_ij)
		s_out, v_out = self.mixing(s_out, v_out)
		
		# Extract Heavy Atoms to return to Mamba
		s_heavy_out = s_out[:, :s_heavy.shape[1], :]
		
		return s_heavy_out, v_out

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

class BiMambaPaiNNLayer(nn.Module):
	def __init__(
		self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, 
		dropout: float = 0.1, n_rbf: int = 20, use_painn: bool = True
	):
		super().__init__()
		self.use_painn = use_painn
		
		if self.use_painn:
			self.painn = PaiNNBlock(d_model, n_rbf)
			self.painn_norm = RMSNorm(d_model)
		else:
			self.painn = None
			self.painn_norm = None
			
		self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm=True)
		self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm=True)
		self.out_proj = nn.Linear(d_model * 2, d_model, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

	def forward(
		self, 
		s: torch.Tensor, 
		v: Optional[torch.Tensor], 
		rbf: Optional[torch.Tensor],
		dir_ij: Optional[torch.Tensor],
		mask_ij: Optional[torch.Tensor],
		is_atom_patch: Optional[torch.Tensor], 
		is_hydrogen: Optional[torch.Tensor] 
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""
		s: (Batch, Seq_Len, Dim)
		v: (Batch, N_Heavy+H, 3, Dim)
		is_atom_patch: (Batch, Seq_Len) - Bool mask
		"""
		
		# Geometric Update (PaiNN)
		v_new = v
		
		if self.use_painn and self.painn is not None and v is not None and is_atom_patch is not None:
			B, L, D = s.shape
			num_heavy = v.shape[1] 
			
			# Generate sort indices.
			sort_idx = torch.argsort(is_atom_patch.int(), dim=1, descending=True, stable=True)
			gather_idx = sort_idx.unsqueeze(-1).expand(-1, -1, D)
			
			# Gather the full sequence. 
			# Result: [Atom1, Atom2, ..., AtomN, Syntax1, Syntax2, ...]
			s_sorted = torch.gather(s, dim=1, index=gather_idx)
			
			# Slice the dense atom part
			# (If v includes Hydrogens, PaiNNBlock handles the padding/expansion internally)
			s_heavy_dense = s_sorted[:, :num_heavy, :]

			s_geo_out, v_new = self.painn(s_heavy_dense, v, rbf, dir_ij, mask_ij, is_hydrogen)
			
			# Update the sorted tensor with the new atom embeddings
			s_sorted_updated = s_sorted.clone()
			s_sorted_updated[:, :num_heavy, :] = s_geo_out
			
			# Scatter back to original positions.
			s = torch.zeros_like(s).scatter_(1, gather_idx, s_sorted_updated)
			s = self.painn_norm(s)

		h_fwd = self.mamba_fwd(s)
		s_rev = torch.flip(s, dims=[1])
		h_bwd = torch.flip(self.mamba_bwd(s_rev), dims=[1])
		h_bi = torch.cat([h_fwd, h_bwd], dim=-1)

		mamba_out = self.out_proj(h_bi)

		s = s + self.dropout(mamba_out) * self.layer_scale
		
		return s, v_new

class BiMambaBackbone(nn.Module):
	def __init__(
		self, patch_dim: int, d_model: int, n_layers: int,
		d_state: int = 16, d_conv: int = 4, expand: int = 2, 
		dropout: float = 0.1, painn_cutoff: float = 5.0, painn_n_rbf: int = 20,
		use_painn: bool = True
	):
		super().__init__()
		self.patch_projection = nn.Linear(patch_dim, d_model)
		self.final_norm = RMSNorm(d_model)
		self.use_painn = use_painn
		
		if self.use_painn:
			self.radial_basis = RadialBasis(painn_n_rbf, painn_cutoff)
			self.cutoff = painn_cutoff
		else:
			self.radial_basis = None
			self.cutoff = None

		self.layers = nn.ModuleList([
			BiMambaPaiNNLayer(
				d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, 
				dropout=dropout, n_rbf=painn_n_rbf, use_painn=use_painn
			) for _ in range(n_layers)
		])

	def forward(
		self, 
		patch_embeddings: torch.Tensor,
		pos: Optional[torch.Tensor], 
		atom_mask: Optional[torch.Tensor], 
		attention_mask: Optional[torch.Tensor] = None, 
		is_atom_patch: Optional[torch.Tensor] = None, 
		is_hydrogen: Optional[torch.Tensor] = None
	) -> torch.Tensor:
		
		s = self.patch_projection(patch_embeddings)
		B, L, D = s.shape
		
		# Graph State
		has_geo = (pos is not None and is_hydrogen is not None)
		rbf, dir_ij, mask_ij = None, None, None
		v = None
		
		if self.use_painn and has_geo:
			N_all = pos.shape[1]
			v = torch.zeros(B, N_all, 3, D, device=s.device, dtype=s.dtype)
			
			diff = pos.unsqueeze(2) - pos.unsqueeze(1)
			dist = torch.norm(diff, dim=-1)
			
			mask_ij = (dist < self.cutoff) & (dist > 1e-5)
			
			dir_ij = diff / torch.clamp(dist, min=1e-6).unsqueeze(-1)
			rbf_val = self.radial_basis(dist)
			d_scaled = dist / self.cutoff
			fcut = 0.5 * (torch.cos(math.pi * d_scaled) + 1.0) * mask_ij.float()
			rbf = rbf_val * fcut.unsqueeze(-1)

		for layer in self.layers:
			# Pass data only if initialized; layer handles None internally
			s, v = layer(s, v, rbf, dir_ij, mask_ij, is_atom_patch, is_hydrogen)

		s = self.final_norm(s)
		return s

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
		pos: Optional[torch.Tensor] = None,
		is_atom_patch: Optional[torch.Tensor] = None,
		is_hydrogen: Optional[torch.Tensor] = None,
		**kwargs
	) -> Dict[str, torch.Tensor]:
		
		patch_embeddings = self.get_patch_embeddings(byte_ids)
		original_patch_embeddings = patch_embeddings.clone()
		
		if atom_mask is not None and mask_strategy is not None:
			patch_embeddings_input = self.apply_masking_with_strategy(patch_embeddings, atom_mask, mask_strategy)
		elif atom_mask is not None:
			patch_embeddings_input = patch_embeddings.clone()
			patch_embeddings_input[atom_mask] = self.mask_patch_embedding.to(patch_embeddings_input.dtype)
		else:
			patch_embeddings_input = patch_embeddings

		hidden_states = self.bimamba_backbone(
			patch_embeddings_input, 
			pos=pos,
			atom_mask=atom_mask,
			attention_mask=bert_attention_mask,
			is_atom_patch=is_atom_patch,
			is_hydrogen=is_hydrogen
		)

		logits = self.mlm_head(hidden_states)

		return {
			'logits': logits,
			'patch_embeddings': original_patch_embeddings,
			'hidden_states': hidden_states,
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
