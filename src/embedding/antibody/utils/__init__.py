"""
Utility functions for antibody embedding.

This module provides:
- Amino acid feature lookups (BLOSUM62, hydropathy, charge, volume)
- IMGT numbering and CDR region detection
"""

from .amino_acid_features import (
    BLOSUM62,
    KYTE_DOOLITTLE_HYDROPATHY,
    AMINO_ACID_CHARGE,
    AMINO_ACID_VOLUME,
    AMINO_ACID_MASS,
    get_mutation_features,
    get_blosum_score,
    get_hydropathy_diff,
    get_charge_diff,
    get_volume_diff,
)

from .numbering import (
    number_antibody,
    get_cdr_regions,
    get_imgt_position,
    get_region_label,
    IMGTPosition,
    CDRRegion,
    IMGT_CDR_RANGES,
)

__all__ = [
    # Amino acid features
    'BLOSUM62',
    'KYTE_DOOLITTLE_HYDROPATHY',
    'AMINO_ACID_CHARGE',
    'AMINO_ACID_VOLUME',
    'AMINO_ACID_MASS',
    'get_mutation_features',
    'get_blosum_score',
    'get_hydropathy_diff',
    'get_charge_diff',
    'get_volume_diff',
    # Numbering
    'number_antibody',
    'get_cdr_regions',
    'get_imgt_position',
    'get_region_label',
    'IMGTPosition',
    'CDRRegion',
    'IMGT_CDR_RANGES',
]
