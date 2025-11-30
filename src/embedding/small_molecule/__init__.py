"""
Small molecule embedding methods.
"""

from .base import MoleculeEmbedder
from .fingerprints import FingerprintEmbedder
from .edit_embedder import EditEmbedder
from .trainable_edit_embedder import TrainableEditEmbedder, ConcatenationEditEmbedder
from .structured_edit_embedder import StructuredEditEmbedder

__all__ = [
    'MoleculeEmbedder',
    'FingerprintEmbedder',
    'EditEmbedder',
    'TrainableEditEmbedder',
    'ConcatenationEditEmbedder',
    'StructuredEditEmbedder',
]

# Optional imports (require additional dependencies)
try:
    from .chemberta import ChemBERTaEmbedder
    __all__.append('ChemBERTaEmbedder')
except ImportError:
    ChemBERTaEmbedder = None

try:
    from .chemprop import ChemPropEmbedder
    __all__.append('ChemPropEmbedder')
except ImportError:
    ChemPropEmbedder = None
