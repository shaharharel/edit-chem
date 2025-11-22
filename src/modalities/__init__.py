"""
Modality-specific implementations for different molecular types.

This package contains implementations for various drug modalities:
- small_molecule: Traditional small molecule drugs
- (future) antibody: Therapeutic antibodies
- (future) protein: Protein therapeutics
"""

from . import small_molecule

__all__ = ['small_molecule']
