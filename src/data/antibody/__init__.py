"""
Antibody data module for mutation effect prediction.

This module provides:
- AbEditPairs: Schema for antibody mutation data
- AbPairDataset: PyTorch dataset for training
- Dataset loaders for AbBiBench, AbAgym, etc.
"""

from .schema import (
    AbMutation,
    AbEditPair,
    AbEditPairsDataset,
    AssayType,
)

from .dataset import (
    AbPairDataset,
    AbPairCollator,
    create_dataloaders,
)

from .loaders import (
    load_abbibench,
    load_abagym,
    load_ab_bind,
    load_skempi2_antibodies,
)

__all__ = [
    # Schema
    'AbMutation',
    'AbEditPair',
    'AbEditPairsDataset',
    'AssayType',
    # Dataset
    'AbPairDataset',
    'AbPairCollator',
    'create_dataloaders',
    # Loaders
    'load_abbibench',
    'load_abagym',
    'load_ab_bind',
    'load_skempi2_antibodies',
]
