"""
Small molecule embedding methods.
"""

from .base import MoleculeEmbedder
from .fingerprints import FingerprintEmbedder
from .edit_embedder import EditEmbedder

__all__ = [
    'MoleculeEmbedder',
    'FingerprintEmbedder',
    'EditEmbedder'
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
