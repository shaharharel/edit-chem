"""
Embedding modules for molecules, proteins, and edits.
"""

from .small_molecule import (
    MoleculeEmbedder,
    FingerprintEmbedder,
    EditEmbedder
)

__all__ = [
    'MoleculeEmbedder',
    'FingerprintEmbedder',
    'EditEmbedder'
]

# Optional imports
try:
    from .small_molecule import ChemBERTaEmbedder
    __all__.append('ChemBERTaEmbedder')
except (ImportError, AttributeError):
    pass

try:
    from .small_molecule import ChemPropEmbedder
    __all__.append('ChemPropEmbedder')
except (ImportError, AttributeError):
    pass
