"""
RNA embedding module for regulatory sequence analysis.

This module provides:

Sequence Embedders (molecule → vector):
- NucleotideEmbedder: Simple baseline (one-hot + k-mer + stats)
- RNAFMEmbedder: RNA-FM pretrained embeddings (640-dim, general RNA)
- RNABERTEmbedder: RNABERT pretrained embeddings (768-dim, general RNA)
- UTRLMEmbedder: UTR-LM for 5' UTR sequences (128-dim, 5'UTR specific)

Edit Embedders (edit → vector):
- RNAEditEmbedder: Simple difference (emb_B - emb_A), works with any embedder
- StructuredRNAEditEmbedder: Rich edit embeddings with:
    * Mutation type (12 SNV types)
    * Mutation effect (Δ learned nucleotide embedding)
    * Position encoding (sinusoidal + learned + relative)
    * Local context (mean-pooled window from base LM)
    * Attention context (attention-weighted tokens to edit site)
    * (Optional) Structure features (Δ-pairing, Δ-accessibility, Δ-MFE)

Factory functions:
- create_rnafm_structured_embedder(): StructuredRNAEditEmbedder with RNA-FM
- create_utrlm_structured_embedder(): StructuredRNAEditEmbedder with UTR-LM

Structure predictors:
- RNAplfoldPredictor: ViennaRNA-based local folding
- EternaFoldPredictor: EternaFold structure prediction
- CombinedStructurePredictor: Ensemble of structure predictors
"""

from .base import RNAEmbedder
from .nucleotide import NucleotideEmbedder
from .edit_embedder import RNAEditEmbedder, TrainableRNAEditEmbedder

# Conditionally import pretrained embedders
try:
    from .rnafm import RNAFMEmbedder
except ImportError:
    RNAFMEmbedder = None

try:
    from .rnabert import RNABERTEmbedder
except ImportError:
    RNABERTEmbedder = None

# Structured edit embedder (works with any base embedder)
try:
    from .structured_edit_embedder import (
        StructuredRNAEditEmbedder,
        create_rnafm_structured_embedder,
        create_utrlm_structured_embedder,
        MUTATION_TYPES,
        NUC_TO_IDX
    )
except ImportError:
    StructuredRNAEditEmbedder = None
    create_rnafm_structured_embedder = None
    create_utrlm_structured_embedder = None
    MUTATION_TYPES = None
    NUC_TO_IDX = None

# Structure predictors
try:
    from .rnaplfold import RNAplfoldPredictor, RNAfoldPredictor
except ImportError:
    RNAplfoldPredictor = None
    RNAfoldPredictor = None

try:
    from .eternafold import EternaFoldPredictor, CombinedStructurePredictor
except ImportError:
    EternaFoldPredictor = None
    CombinedStructurePredictor = None

# UTR-LM embedder
try:
    from .utrlm import UTRLMEmbedder, load_utrlm
except ImportError:
    UTRLMEmbedder = None
    load_utrlm = None

# CodonBERT (stub)
try:
    from .codonbert import CodonBERTEmbedder, CodonDeltaEmbedder, load_codonbert
except ImportError:
    CodonBERTEmbedder = None
    CodonDeltaEmbedder = None
    load_codonbert = None

__all__ = [
    # Base
    'RNAEmbedder',
    'NucleotideEmbedder',
    # Edit embedders
    'RNAEditEmbedder',
    'TrainableRNAEditEmbedder',
    'StructuredRNAEditEmbedder',
    # Factory functions
    'create_rnafm_structured_embedder',
    'create_utrlm_structured_embedder',
    # Constants
    'MUTATION_TYPES',
    'NUC_TO_IDX',
    # Pretrained
    'RNAFMEmbedder',
    'RNABERTEmbedder',
    # Structure predictors
    'RNAplfoldPredictor',
    'RNAfoldPredictor',
    'EternaFoldPredictor',
    'CombinedStructurePredictor',
    # UTR-LM
    'UTRLMEmbedder',
    'load_utrlm',
    # CodonBERT
    'CodonBERTEmbedder',
    'CodonDeltaEmbedder',
    'load_codonbert',
]
