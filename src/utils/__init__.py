"""
Modality-agnostic utility modules.

These utilities work with any data type and don't contain domain-specific logic.
For modality-specific utilities (e.g., chemistry for small molecules), see src/data/<modality>/utils/
"""

from .logging import setup_logger
from .embedding_cache import (
    EmbeddingCache,
    get_or_compute_embeddings_for_pairs,
    get_or_compute_embeddings_for_molecules,
    compute_all_embeddings_once,
    compute_all_embeddings_with_fragments,
    map_embeddings_to_pairs,
    map_embeddings_to_molecules,
    map_fragment_embeddings_to_pairs
)
from .splits import (
    MolecularSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TargetSplitter,
    ButinaSplitter,
    PropertyStratifiedSplitter,
    TemporalSplitter,
    FewShotTargetSplitter,
    CoreSplitter,
    get_splitter
)
from .metrics import (
    RegressionMetrics,
    MultiTaskMetrics,
    RankingMetrics,
    ChemistryMetrics,
    print_metrics_summary
)

__all__ = [
    'setup_logger',
    'EmbeddingCache',
    'get_or_compute_embeddings_for_pairs',
    'get_or_compute_embeddings_for_molecules',
    'compute_all_embeddings_once',
    'compute_all_embeddings_with_fragments',
    'map_embeddings_to_pairs',
    'map_embeddings_to_molecules',
    'map_fragment_embeddings_to_pairs',
    'MolecularSplitter',
    'RandomSplitter',
    'ScaffoldSplitter',
    'TargetSplitter',
    'ButinaSplitter',
    'PropertyStratifiedSplitter',
    'TemporalSplitter',
    'FewShotTargetSplitter',
    'CoreSplitter',
    'get_splitter',
    'RegressionMetrics',
    'MultiTaskMetrics',
    'RankingMetrics',
    'ChemistryMetrics',
    'print_metrics_summary'
]
