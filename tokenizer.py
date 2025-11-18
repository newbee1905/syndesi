import re
from enum import Enum
from dataclasses import dataclass
from typing import List

# ATOM TOKENIZER

SMILES_ATOM_PATTERN = r"""
	\[			# Bracketed atom
		(?:[0-9]{1,3})?	# isotope (optional)
		(?:			# element symbol
			Cl|Br|
			[A-Z][a-z]?
		)
		(?:@{1,2})?		# chirality (optional)
		(?:H[0-9]?)?		# explicit H count (optional)
		(?:[+-](?:[0-9]+)?)?  # charge (optional)
		(?::[0-9]{1,3})?	# atom class/map (optional)
	\] |
	(?:			# Unbracketed atom (organic subset)
		Cl|Br|		# Two-letter atoms
		[BC]|[NO]|[PS]|F|I|At|Ts|  # One-letter atoms
		b|c|n|o|p|s|As|Se  # Aromatic atoms
	)
"""

SMILES_BOND_PATTERN = r"[-=#$:/\\]|(?<!%)[0-9]{1}|%[0-9]{2}"
SMILES_BRANCH_PATTERN = r"[().]"


class TokenType(Enum):
	"""SMILES token types based on grammar"""
	ATOM = "atom"
	BOND = "bond"
	BRANCH_OPEN = "("
	BRANCH_CLOSE = ")"
	RING_CLOSURE = "ring"
	DISCONNECTION = "."
	UNKNOWN = "unknown"


@dataclass
class SMILESToken:
	"""Represents a single SMILES token"""
	value: str
	token_type: TokenType
	is_atom: bool
	idx: int

	def __repr__(self):
		return f"{self.value}"


class AtomLevelSMILESTokenizer:
	"""
	SMILES tokenizer using atom-level (fixed grammar) tokenization.
	"""

	def __init__(self, vocab_size: int = 257, max_atom_bytes: int = 8):
		"""
		Args:
			vocab_size: Byte vocabulary size (0-255 for bytes, +1 for pad)
			max_atom_bytes: Max bytes per atom token (for padding)
		"""
		self.vocab_size = vocab_size
		self.max_atom_bytes = max_atom_bytes

		# Compile regex pattern
		pattern = f"({SMILES_ATOM_PATTERN}|{SMILES_BOND_PATTERN}|{SMILES_BRANCH_PATTERN})"
		self.smiles_regex = re.compile(pattern, re.VERBOSE)

		self.atom_token_set = set()

	def _classify_token(self, token: str) -> TokenType:
		"""Classify a token according to SMILES grammar"""
		if token == "(":
			return TokenType.BRANCH_OPEN
		elif token == ")":
			return TokenType.BRANCH_CLOSE
		elif token == ".":
			return TokenType.DISCONNECTION
		elif re.match(r'^[0-9]$|^%[0-9]{2}$', token):
			return TokenType.RING_CLOSURE
		elif re.match(r'^[-=#$:/\\]$', token):
			return TokenType.BOND
		elif re.match(SMILES_ATOM_PATTERN, token, re.VERBOSE):
			return TokenType.ATOM
		else:
			return TokenType.UNKNOWN

	def tokenize_smiles_atoms(self, smiles: str) -> List[SMILESToken]:
		"""
		Parse SMILES into atom-level tokens using grammar-based tokenization.
		"""
		matches = self.smiles_regex.findall(smiles)
		tokens = []

		for idx, token in enumerate(matches):
			token_type = self._classify_token(token)
			is_atom = token_type == TokenType.ATOM

			if is_atom:
				self.atom_token_set.add(token)

			tokens.append(SMILESToken(
				value=token,
				token_type=token_type,
				is_atom=is_atom,
				idx=idx
			))

		return tokens

	def tokens_to_bytes(self, tokens: List[SMILESToken]) -> List[int]:
		"""
		Convert SMILES tokens to a flattened byte sequence, padding
		each token to `max_atom_bytes`.
		"""
		byte_seq = []
		for token in tokens:
			token_bytes = token.value.encode('utf-8')
			byte_vals = list(token_bytes)
			
			# Pad with zeros (0 is our padding ID)
			padding = [0] * (self.max_atom_bytes - len(byte_vals))
			
			# Truncate if too long (shouldn't happen with good regex)
			token_byte_seq = (byte_vals + padding)[:self.max_atom_bytes]
			
			byte_seq.extend(token_byte_seq)
			
		return byte_seq
