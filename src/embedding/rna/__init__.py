"""
RNA embedding module for regulatory sequence analysis.

This module provides:
- RNAEmbedder: Abstract base class for RNA embedders
- RNAFMEmbedder: RNA-FM pretrained embeddings (640-dim)
- RNABERTEmbedder: RNABERT pretrained embeddings (768-dim)
- NucleotideEmbedder: Simple baseline (one-hot + k-mer)
- RNAEditEmbedder: Edit embeddings for Δ-prediction
- StructuredRNAEditEmbedder: Rich edit embeddings with mutation type, position, context
"""

from .base import RNAEmbedder
from .nucleotide import NucleotideEmbedder
from .edit_embedder import RNAEditEmbedder

# Conditionally import pretrained embedders
try:
    from .rnafm import RNAFMEmbedder
except ImportError:
    RNAFMEmbedder = None

try:
    from .rnabert import RNABERTEmbedder
except ImportError:
    RNABERTEmbedder = None

# Structured edit embedder (requires RNAFMEmbedder)
try:
    from .structured_edit_embedder import StructuredRNAEditEmbedder
except ImportError:
    StructuredRNAEditEmbedder = None

__all__ = [
    'RNAEmbedder',
    'NucleotideEmbedder',
    'RNAEditEmbedder',
    'RNAFMEmbedder',
    'RNABERTEmbedder',
    'StructuredRNAEditEmbedder',
]
