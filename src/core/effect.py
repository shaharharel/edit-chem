"""Causal effect representation - τₑ in the edit calculus."""

from dataclasses import dataclass
from typing import Optional, Tuple
from .edit import Edit


@dataclass
class CausalEffect:
    """
    Represents the causal effect τₑ of an edit on a property.

    This is the central object in our calculus - it quantifies how much
    a specific edit changes a specific property under a specific context.

    Attributes:
        edit: The edit whose effect we're measuring
        property_name: Name of the property (e.g., 'logp', 'ic50', 'solubility')
        delta_mean: Mean change in property value (μ₁ - μ₀)
        delta_std: Standard deviation of the change
        n_observations: Number of times this edit was observed
        source_dataset: Which dataset this observation came from
        confidence_interval: Optional (lower, upper) bounds
        context: Optional context information (scaffold, assay, etc.)
    """
    edit: Edit
    property_name: str
    delta_mean: float
    delta_std: float
    n_observations: int
    source_dataset: str
    confidence_interval: Optional[Tuple[float, float]] = None
    context: Optional[dict] = None

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'edit_id': self.edit.edit_id,
            'property_name': self.property_name,
            'delta_mean': self.delta_mean,
            'delta_std': self.delta_std,
            'n_observations': self.n_observations,
            'source_dataset': self.source_dataset,
            'confidence_interval': self.confidence_interval,
            'context': self.context
        }

    @classmethod
    def from_dict(cls, data, edit):
        """Create CausalEffect from dictionary and Edit object."""
        return cls(
            edit=edit,
            property_name=data['property_name'],
            delta_mean=data['delta_mean'],
            delta_std=data['delta_std'],
            n_observations=data['n_observations'],
            source_dataset=data['source_dataset'],
            confidence_interval=data.get('confidence_interval'),
            context=data.get('context')
        )

    def __repr__(self):
        return f"CausalEffect({self.edit}, Δ{self.property_name}={self.delta_mean:.2f}±{self.delta_std:.2f}, n={self.n_observations})"
