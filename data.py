import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict

from rdkit import Chem
from rdkit import RDLogger
# Disable verbose RDKit logging
RDLogger.DisableLog('rdApp.*')

from tokenizer import AtomLevelSMILESTokenizer, SMILESToken


class BytePatchSMILESProcessor:
	"""
	Converts SMILES to torch tensors.
	Processes a SINGLE SMILES string. Collate_fn handles batching.
	"""

	def __init__(
		self,
		vocab_size: int = 257,
		max_bytes_per_atom: int = 8,
	):
		"""
		Args:
			vocab_size: Byte vocabulary (0-255 + 1 pad)
			max_bytes_per_atom: Max bytes per atom/token (padding)
		"""
		self.tokenizer = AtomLevelSMILESTokenizer(vocab_size, max_bytes_per_atom)
		self.vocab_size = vocab_size
		self.max_bytes_per_atom = max_bytes_per_atom

	def process_smiles(self, smiles: str) -> Dict[str, torch.Tensor]:
		"""
		Processes a single SMILES string into byte IDs and token metadata.
		Returns tensors.
		"""
		if not smiles or not isinstance(smiles, str):
			raise ValueError(f"Invalid SMILES string: {smiles}")

		tokens = self.tokenizer.tokenize_smiles_atoms(smiles)
		if not tokens:
			raise ValueError(f"SMILES string '{smiles}' yielded no tokens.")

		byte_seq = self.tokenizer.tokens_to_bytes(tokens)
		
		is_atom_mask = [t.is_atom for t in tokens]
		
		num_tokens = len(tokens)
		num_bytes = len(byte_seq)

		if num_bytes != num_tokens * self.max_bytes_per_atom:
			# This should not happen if logic is correct
			raise ValueError(f"Logic error: num_bytes ({num_bytes}) != num_tokens ({num_tokens}) * max_bytes ({self.max_bytes_per_atom})")

		return {
			'byte_ids': torch.tensor(byte_seq, dtype=torch.long),
			'is_atom_patch': torch.tensor(is_atom_mask, dtype=torch.bool),
			'num_tokens': num_tokens
		}

class MLMMapDataset(Dataset):
	"""Dataset with masking strategy selection"""
	def __init__(
		self,
		list_of_smiles: List[str],
		processor: BytePatchSMILESProcessor,
		mlm_probability: float = 0.15,
		augment_prob: float = 0.0,

		# BERT-style masking ratios (applied at patch level in model)
		mask_token_prob: float = 0.8,   # Use [MASK] token
		random_token_prob: float = 0.1,  # Use random patch
		keep_token_prob: float = 0.1,	# Keep original
	):
		self.processor = processor
		self.mlm_probability = mlm_probability
		self.augment_prob = augment_prob
		self.mask_token_prob = mask_token_prob
		self.random_token_prob = random_token_prob
		self.keep_token_prob = keep_token_prob
		self.smiles_list = list_of_smiles
		
		assert abs(mask_token_prob + random_token_prob + keep_token_prob - 1.0) < 1e-6, \
			"Masking probabilities must sum to 1.0"

		if self.augment_prob > 0 and Chem is not None:
			print(f"ImprovedMLMMapDataset initialized with {len(self.smiles_list)} SMILES and augmentation (prob={self.augment_prob}).")
		else:
			print(f"ImprovedMLMMapDataset initialized with {len(self.smiles_list)} SMILES.")

	def __len__(self) -> int:
		return len(self.smiles_list)

	def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
		smiles = self.smiles_list[idx]
		
		if not smiles or not isinstance(smiles, str):
			return None
		
		# SMILES augmentation
		if Chem is not None and torch.rand(1).item() < self.augment_prob:
			try:
				mol = Chem.MolFromSmiles(smiles)
				if mol:
					smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
			except Exception:
				pass 

		try:
			token_data = self.processor.process_smiles(smiles)
		except Exception as e:
			return None

		byte_ids = token_data['byte_ids']
		is_atom_patch = token_data['is_atom_patch']
		num_atoms = token_data['num_tokens']
		
		if num_atoms == 0:
			return None

		max_bytes_per_atom = self.processor.max_bytes_per_atom
		
		atom_labels = torch.full((num_atoms, max_bytes_per_atom), -100, dtype=torch.long)
		atom_attention_mask = torch.ones(num_atoms, dtype=torch.long)
		
		# Select which atoms to mask
		prob_mask = torch.rand(num_atoms) < self.mlm_probability
		
		if prob_mask.sum() == 0 and num_atoms > 0:
			mask_idx = torch.randint(0, num_atoms, (1,)).item()
			prob_mask[mask_idx] = True

		atom_mask = prob_mask.bool()
		
		byte_ids_patched = byte_ids.view(num_atoms, max_bytes_per_atom)
		
		# Determine masking strategy for each masked atom
		# 0 = use [MASK], 1 = use random, 2 = keep original
		mask_strategy = torch.zeros(num_atoms, dtype=torch.long)
		
		if num_atoms > 0:
			# Store labels for all masked positions
			atom_labels[atom_mask] = byte_ids_patched[atom_mask]
			
			# Assign strategy to each masked atom
			mask_indices = torch.where(atom_mask)[0]
			for idx in mask_indices:
				prob = torch.rand(1).item()
				if prob < self.mask_token_prob:
					mask_strategy[idx] = 0  # Use [MASK] token
				elif prob < self.mask_token_prob + self.random_token_prob:
					mask_strategy[idx] = 1  # Use random patch
				else:
					mask_strategy[idx] = 2  # Keep original

		return {
			'byte_ids': byte_ids,
			'atom_labels': atom_labels,
			'atom_mask': atom_mask,
			'mask_strategy': mask_strategy,
			'atom_attention_mask': atom_attention_mask,
			'is_atom_patch': is_atom_patch,
		}

def mlm_collate_fn(batch: List[Optional[Dict[str, any]]]) -> Dict[str, torch.Tensor]:
	"""Pads batches to the max length in the batch, handles mask_strategy."""
	
	batch = [item for item in batch if item is not None and item['byte_ids'].shape[0] > 0]
	if not batch:
		return {} 

	max_byte_len = max(item['byte_ids'].shape[0] for item in batch)
	byte_ids_padded = torch.full((len(batch), max_byte_len), 0, dtype=torch.long)
	
	max_atom_len = max(item['atom_labels'].shape[0] for item in batch)
	max_bytes = batch[0]['atom_labels'].shape[1]
	
	atom_labels_padded = torch.full((len(batch), max_atom_len, max_bytes), -100, dtype=torch.long)
	atom_mask_padded = torch.full((len(batch), max_atom_len), False, dtype=torch.bool)
	atom_attention_mask_padded = torch.full((len(batch), max_atom_len), 0, dtype=torch.long)
	mask_strategy_padded = torch.full((len(batch), max_atom_len), 0, dtype=torch.long)
	is_atom_patch_padded = torch.full((len(batch), max_atom_len), False, dtype=torch.bool)
	
	for i, item in enumerate(batch):
		b_len = item['byte_ids'].shape[0]
		byte_ids_padded[i, :b_len] = item['byte_ids']
		
		a_len = item['atom_labels'].shape[0]
		atom_labels_padded[i, :a_len, :] = item['atom_labels']
		atom_mask_padded[i, :a_len] = item['atom_mask']
		atom_attention_mask_padded[i, :a_len] = item['atom_attention_mask']
		mask_strategy_padded[i, :a_len] = item['mask_strategy']
		is_atom_patch_padded[i, :a_len] = item['is_atom_patch']

	bert_attention_mask = atom_attention_mask_padded.view(
		len(batch), 1, 1, max_atom_len
	)
	bert_attention_mask = (1.0 - bert_attention_mask.float()) * -10000.0

	return {
		'byte_ids': byte_ids_padded,
		'atom_labels': atom_labels_padded,
		'atom_mask': atom_mask_padded,
		'atom_attention_mask': atom_attention_mask_padded,
		'mask_strategy': mask_strategy_padded,
		'bert_attention_mask': bert_attention_mask,
		'is_atom_patch': is_atom_patch_padded,
	}

class ChemBERTaFinetuneDataset(Dataset):
	"""Dataset for fine-tuning ChemBERTa."""
	def __init__(
		self,
		list_of_smiles: List[str],
		list_of_targets: List[float],
		tokenizer,
		max_length: int = 128
	):
		self.smiles_list = list_of_smiles
		self.targets = list_of_targets
		self.tokenizer = tokenizer
		self.max_length = max_length
		print(f"ChemBERTaFinetuneDataset initialized with {len(self.smiles_list)} SMILES.")

	def __len__(self) -> int:
		return len(self.smiles_list)

	def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
		smiles = self.smiles_list[idx]
		target = self.targets[idx]
		
		if not smiles or not isinstance(smiles, str) or target is None:
			return None
		
		try:
			tokenized = self.tokenizer(
				smiles,
				truncation=True,
				padding=False, # Collate will pad
				max_length=self.max_length,
				return_tensors="pt"
			)
		except Exception as e:
			print(f"Warning: Skipping SMILES {smiles} due to tokenizer error: {e}")
			return None

		return {
			'input_ids': tokenized['input_ids'].squeeze(0),
			'attention_mask': tokenized['attention_mask'].squeeze(0),
			'target': torch.tensor(target, dtype=torch.float)
		}

def chemberta_collate_fn(batch: List[Optional[Dict[str, any]]]) -> Dict[str, torch.Tensor]:
	"""Pads batches for ChemBERTa fine-tuning."""
	batch = [item for item in batch if item is not None]
	if not batch: return {}
	
	# Pad input_ids and attention_mask
	input_ids_padded = torch.nn.utils.rnn.pad_sequence(
		[item['input_ids'] for item in batch],
		batch_first=True,
		padding_value=0 # Tokenizer pad token ID
	)
	attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
		[item['attention_mask'] for item in batch],
		batch_first=True,
		padding_value=0 # 0 means "don't attend"
	)
	
	targets = torch.stack([item['target'] for item in batch])

	return {
		'input_ids': input_ids_padded,
		'attention_mask': attention_mask_padded,
		'targets': targets
	}

class FinetuneMapDataset(Dataset):
	"""Map-style dataset for fine-tuning on regression tasks."""
	def __init__(
		self,
		list_of_smiles: List[str],
		list_of_targets: List[float],
		processor: BytePatchSMILESProcessor,
	):
		self.processor = processor
		self.smiles_list = list_of_smiles
		self.targets = list_of_targets
		print(f"FinetuneMapDataset initialized with {len(self.smiles_list)} SMILES.")

	def __len__(self) -> int:
		return len(self.smiles_list)

	def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
		smiles = self.smiles_list[idx]
		target = self.targets[idx]
		
		if not smiles or not isinstance(smiles, str) or target is None:
			return None
		
		try:
			token_data = self.processor.process_smiles(smiles)
		except Exception as e:
			return None

		num_atoms = token_data['num_tokens']
		if num_atoms == 0:
			return None

		return {
			'byte_ids': token_data['byte_ids'],
			'is_atom_patch': token_data['is_atom_patch'],
			'atom_attention_mask': torch.ones(num_atoms, dtype=torch.long),
			'target': torch.tensor(target, dtype=torch.float)
		}

def finetune_collate_fn(batch: List[Optional[Dict[str, any]]]) -> Dict[str, torch.Tensor]:
	"""Pads batches for fine-tuning."""
	batch = [item for item in batch if item is not None]
	if not batch: return {}

	# Pad Byte-level sequences (L dimension)
	max_byte_len = max(item['byte_ids'].shape[0] for item in batch)
	byte_ids_padded = torch.full((len(batch), max_byte_len), 0, dtype=torch.long)
	
	# Pad Atom-level sequences (N dimension)
	max_atom_len = max(item['atom_attention_mask'].shape[0] for item in batch)
	
	atom_attention_mask_padded = torch.full((len(batch), max_atom_len), 0, dtype=torch.long)
	is_atom_patch_padded = torch.full((len(batch), max_atom_len), False, dtype=torch.bool)
	
	targets = []

	for i, item in enumerate(batch):
		b_len = item['byte_ids'].shape[0]
		byte_ids_padded[i, :b_len] = item['byte_ids']
		
		a_len = item['atom_attention_mask'].shape[0]
		atom_attention_mask_padded[i, :a_len] = item['atom_attention_mask']
		is_atom_patch_padded[i, :a_len] = item['is_atom_patch']
		
		targets.append(item['target'])

	# Create additive mask (for MambaBERT, ignored by MambaBiMamba)
	bert_attention_mask = atom_attention_mask_padded.view(
		len(batch), 1, 1, max_atom_len
	)
	bert_attention_mask = (1.0 - bert_attention_mask.float()) * -10000.0

	return {
		'byte_ids': byte_ids_padded,
		'atom_attention_mask': atom_attention_mask_padded,
		'bert_attention_mask': bert_attention_mask,
		'is_atom_patch': is_atom_patch_padded,
		'targets': torch.stack(targets)
	}
