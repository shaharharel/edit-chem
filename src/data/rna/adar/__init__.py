"""
ADAR RNA editing data module.

This module provides tools for:
- Processing ADAR editing data from RNA-seq experiments
- Extracting positive (edited) and negative (non-edited) examples
- Generating RNA sequence windows with proper strand handling
"""

from .data_loader import (
    ADARDataLoader,
    load_edited_sites,
    load_pileup_data,
    load_genome,
    load_transcript_annotations,
)
from .dataset import ADAREditingDataset

__all__ = [
    'ADARDataLoader',
    'ADAREditingDataset',
    'load_edited_sites',
    'load_pileup_data',
    'load_genome',
    'load_transcript_annotations',
]
