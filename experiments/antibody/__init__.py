"""
Antibody mutation effect prediction experiments.

This module provides experiment infrastructure for training and evaluating
antibody mutation effect predictors using various embedders and datasets.
"""

from .experiment_config import AntibodyExperimentConfig
from .model_factory import create_embedder, create_edit_embedder, create_predictor

__all__ = [
    'AntibodyExperimentConfig',
    'create_embedder',
    'create_edit_embedder',
    'create_predictor',
]
