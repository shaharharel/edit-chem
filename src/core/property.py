"""Property definitions for molecular properties."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class PropertyType(Enum):
    """Types of molecular properties we can measure."""
    CHEMICAL = "chemical"  # Physicochemical (logP, MW, solubility)
    BIOLOGICAL = "biological"  # Bioactivity (IC50, Ki, Kd)
    ADME = "adme"  # Absorption, distribution, metabolism, excretion
    TOXICITY = "toxicity"  # Toxic effects


@dataclass
class Property:
    """
    Defines a molecular property that can be affected by edits.

    Attributes:
        name: Property name (e.g., 'logp', 'ic50')
        property_type: Type of property
        units: Units of measurement (e.g., 'log units', 'nM')
        description: Human-readable description
        lower_is_better: Whether lower values are preferred
    """
    name: str
    property_type: PropertyType
    units: str
    description: str
    lower_is_better: bool = False

    def __repr__(self):
        return f"Property({self.name}, {self.property_type.value})"


# Common properties
LOGP = Property(
    name='logp',
    property_type=PropertyType.CHEMICAL,
    units='log units',
    description='Octanol-water partition coefficient (lipophilicity)',
    lower_is_better=False
)

IC50 = Property(
    name='ic50',
    property_type=PropertyType.BIOLOGICAL,
    units='nM',
    description='Half-maximal inhibitory concentration',
    lower_is_better=True
)

SOLUBILITY = Property(
    name='solubility',
    property_type=PropertyType.CHEMICAL,
    units='mg/mL',
    description='Aqueous solubility',
    lower_is_better=False
)

MOLECULAR_WEIGHT = Property(
    name='mw',
    property_type=PropertyType.CHEMICAL,
    units='Da',
    description='Molecular weight',
    lower_is_better=True
)
