"""
RNA data module for MPRA-based edit effect prediction.

This module provides tools for:
- Downloading and processing MPRA datasets
- Extracting RNA sequence pairs with Δ-expression labels
- RNA-specific sequence utilities
"""

from .mpra_pair_extractor import MPRAPairExtractor
from .sequence_utils import (
    validate_rna_sequence,
    compute_edit_distance,
    extract_edit,
    compute_kozak_score,
    find_uaugs,
    reverse_complement,
    dna_to_rna,
    rna_to_dna,
)
from .dataset import RNAPairDataset

__all__ = [
    'MPRAPairExtractor',
    'RNAPairDataset',
    'validate_rna_sequence',
    'compute_edit_distance',
    'extract_edit',
    'compute_kozak_score',
    'find_uaugs',
    'reverse_complement',
    'dna_to_rna',
    'rna_to_dna',
]
