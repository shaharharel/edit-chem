"""
Antibody embedding module for mutation effect prediction.

This module provides:

Sequence Embedders (antibody → per-residue + global embeddings):
- IgT5Embedder: T5-based paired antibody LM (Exscientia)
- IgBertEmbedder: BERT-based paired antibody LM (Exscientia)
- AntiBERTa2Embedder: RoFormer-based antibody LM (Alchemab)
- AbLang2Embedder: Antibody LM for NGL prediction (Oxford)
- BALMEmbedder: Bio-inspired antibody LM (BEAM-Labs)
- BALMPairedEmbedder: Paired antibody LM (Briney Lab)

Structural Encoders (3D structure → per-residue embeddings):
- GVPEncoder: Geometric Vector Perceptron GNN
- SE3TransformerEncoder: SE(3)-equivariant transformer
- EquiformerEncoder: Equivariant transformer (SOTA)

Edit Embedders (mutation → edit vector):
- AntibodyEditEmbedder: Simple difference (emb_mut - emb_wt)
- StructuredAntibodyEditEmbedder: Rich edit embeddings with:
    * Identity features (BLOSUM, hydropathy, charge, volume)
    * Location features (chain, IMGT position, CDR/FR region)
    * Sequence LM context (Δ at mutation site, window context)
    * Structure context (optional, from structural encoders)
    * Multi-mutation aggregation (mean or self-attention)

Utilities:
- Amino acid feature lookups (BLOSUM62, Kyte-Doolittle, etc.)
- IMGT numbering and CDR detection (via ANARCI/AbNumber)
"""

from .base import (
    AntibodyEmbedder,
    AntibodyEmbedderOutput,
    BatchedAntibodyEmbedderOutput,
    StructuralEncoder,
)

# Sequence embedders
try:
    from .igt5 import IgT5Embedder
except ImportError:
    IgT5Embedder = None

try:
    from .igbert import IgBertEmbedder
except ImportError:
    IgBertEmbedder = None

try:
    from .antiberta2 import AntiBERTa2Embedder
except ImportError:
    AntiBERTa2Embedder = None

try:
    from .ablang2 import AbLang2Embedder
except ImportError:
    AbLang2Embedder = None

try:
    from .balm import BALMEmbedder
except ImportError:
    BALMEmbedder = None

try:
    from .balm_paired import BALMPairedEmbedder
except ImportError:
    BALMPairedEmbedder = None

# Structural encoders
try:
    from .structural.gvp import GVPEncoder
except ImportError:
    GVPEncoder = None

try:
    from .structural.se3_transformer import SE3TransformerEncoder
except ImportError:
    SE3TransformerEncoder = None

try:
    from .structural.equiformer import EquiformerEncoder
except ImportError:
    EquiformerEncoder = None

# Edit embedders
try:
    from .edit_embedder import AntibodyEditEmbedder
except ImportError:
    AntibodyEditEmbedder = None

try:
    from .structured_edit_embedder import (
        StructuredAntibodyEditEmbedder,
        create_structured_embedder,
    )
except ImportError:
    StructuredAntibodyEditEmbedder = None
    create_structured_embedder = None

# Utilities
try:
    from .utils.amino_acid_features import (
        BLOSUM62,
        KYTE_DOOLITTLE_HYDROPATHY,
        AMINO_ACID_CHARGE,
        AMINO_ACID_VOLUME,
        get_mutation_features,
    )
except ImportError:
    BLOSUM62 = None
    KYTE_DOOLITTLE_HYDROPATHY = None
    AMINO_ACID_CHARGE = None
    AMINO_ACID_VOLUME = None
    get_mutation_features = None

try:
    from .utils.numbering import (
        number_antibody,
        get_cdr_regions,
        get_imgt_position,
        IMGTPosition,
    )
except ImportError:
    number_antibody = None
    get_cdr_regions = None
    get_imgt_position = None
    IMGTPosition = None

__all__ = [
    # Base
    'AntibodyEmbedder',
    'AntibodyEmbedderOutput',
    'BatchedAntibodyEmbedderOutput',
    'StructuralEncoder',
    # Sequence embedders
    'IgT5Embedder',
    'IgBertEmbedder',
    'AntiBERTa2Embedder',
    'AbLang2Embedder',
    'BALMEmbedder',
    'BALMPairedEmbedder',
    # Structural encoders
    'GVPEncoder',
    'SE3TransformerEncoder',
    'EquiformerEncoder',
    # Edit embedders
    'AntibodyEditEmbedder',
    'StructuredAntibodyEditEmbedder',
    'create_structured_embedder',
    # Utilities
    'BLOSUM62',
    'KYTE_DOOLITTLE_HYDROPATHY',
    'AMINO_ACID_CHARGE',
    'AMINO_ACID_VOLUME',
    'get_mutation_features',
    'number_antibody',
    'get_cdr_regions',
    'get_imgt_position',
    'IMGTPosition',
]
