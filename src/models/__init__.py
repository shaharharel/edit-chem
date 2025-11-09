"""
Predictive models for molecular property optimization.
"""

# Property prediction (baseline non-causal)
from .property_predictor import PropertyPredictor, PropertyPredictorMLP

# Edit effect prediction (causal)
from .edit_effect_predictor import EditEffectPredictor, EditEffectMLP

# Causal estimators (IPW, DR)
from .causal_estimators import (
    IPWEstimator,
    DoublyRobustEstimator,
    PropensityScoreModel
)

# Legacy models
from .delta_predictor import DeltaPropertyPredictor, DeltaPropertyMLP

__all__ = [
    # New PyTorch Lightning models
    'PropertyPredictor',
    'PropertyPredictorMLP',
    'EditEffectPredictor',
    'EditEffectMLP',

    # Causal estimators
    'IPWEstimator',
    'DoublyRobustEstimator',
    'PropensityScoreModel',

    # Legacy models
    'DeltaPropertyPredictor',
    'DeltaPropertyMLP',
]
