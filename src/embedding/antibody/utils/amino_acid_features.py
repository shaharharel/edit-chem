"""
Amino acid feature lookups for mutation analysis.

Provides:
- BLOSUM62 substitution matrix
- Kyte-Doolittle hydropathy scale
- Amino acid charges
- Amino acid volumes (side chain)
- Amino acid masses

These features are used to characterize mutations in the structured
edit embedding.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import torch


# Standard amino acid codes
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# =============================================================================
# BLOSUM62 Substitution Matrix
# =============================================================================

# BLOSUM62 matrix values (symmetric)
# Rows and columns ordered as ARNDCQEGHILKMFPSTWYV
_BLOSUM62_ORDER = 'ARNDCQEGHILKMFPSTWYV'
_BLOSUM62_VALUES = [
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
]

# Convert to dictionary for easy lookup
BLOSUM62: Dict[Tuple[str, str], int] = {}
for i, aa1 in enumerate(_BLOSUM62_ORDER):
    for j, aa2 in enumerate(_BLOSUM62_ORDER):
        BLOSUM62[(aa1, aa2)] = _BLOSUM62_VALUES[i][j]


def get_blosum_score(aa_from: str, aa_to: str) -> int:
    """
    Get BLOSUM62 substitution score for a mutation.

    Args:
        aa_from: Original amino acid (single letter)
        aa_to: Mutated amino acid (single letter)

    Returns:
        BLOSUM62 score (higher = more conservative substitution)
    """
    aa_from = aa_from.upper()
    aa_to = aa_to.upper()
    return BLOSUM62.get((aa_from, aa_to), 0)


# =============================================================================
# Kyte-Doolittle Hydropathy Scale
# =============================================================================

# Hydropathy values from Kyte & Doolittle (1982)
# Positive = hydrophobic, Negative = hydrophilic
KYTE_DOOLITTLE_HYDROPATHY: Dict[str, float] = {
    'A':  1.8,   # Alanine
    'R': -4.5,   # Arginine
    'N': -3.5,   # Asparagine
    'D': -3.5,   # Aspartic acid
    'C':  2.5,   # Cysteine
    'Q': -3.5,   # Glutamine
    'E': -3.5,   # Glutamic acid
    'G': -0.4,   # Glycine
    'H': -3.2,   # Histidine
    'I':  4.5,   # Isoleucine
    'L':  3.8,   # Leucine
    'K': -3.9,   # Lysine
    'M':  1.9,   # Methionine
    'F':  2.8,   # Phenylalanine
    'P': -1.6,   # Proline
    'S': -0.8,   # Serine
    'T': -0.7,   # Threonine
    'W': -0.9,   # Tryptophan
    'Y': -1.3,   # Tyrosine
    'V':  4.2,   # Valine
}


def get_hydropathy_diff(aa_from: str, aa_to: str) -> float:
    """
    Get hydropathy difference for a mutation.

    Args:
        aa_from: Original amino acid
        aa_to: Mutated amino acid

    Returns:
        Hydropathy difference (positive = became more hydrophobic)
    """
    aa_from = aa_from.upper()
    aa_to = aa_to.upper()
    return KYTE_DOOLITTLE_HYDROPATHY.get(aa_to, 0.0) - KYTE_DOOLITTLE_HYDROPATHY.get(aa_from, 0.0)


# =============================================================================
# Amino Acid Charges
# =============================================================================

# Net charge at physiological pH (~7.4)
AMINO_ACID_CHARGE: Dict[str, int] = {
    'A':  0,  # Alanine
    'R': +1,  # Arginine (positive)
    'N':  0,  # Asparagine
    'D': -1,  # Aspartic acid (negative)
    'C':  0,  # Cysteine
    'Q':  0,  # Glutamine
    'E': -1,  # Glutamic acid (negative)
    'G':  0,  # Glycine
    'H': +1,  # Histidine (partially positive at pH 7.4, ~10% protonated)
    'I':  0,  # Isoleucine
    'L':  0,  # Leucine
    'K': +1,  # Lysine (positive)
    'M':  0,  # Methionine
    'F':  0,  # Phenylalanine
    'P':  0,  # Proline
    'S':  0,  # Serine
    'T':  0,  # Threonine
    'W':  0,  # Tryptophan
    'Y':  0,  # Tyrosine
    'V':  0,  # Valine
}


def get_charge_diff(aa_from: str, aa_to: str) -> int:
    """
    Get charge difference for a mutation.

    Args:
        aa_from: Original amino acid
        aa_to: Mutated amino acid

    Returns:
        Charge difference
    """
    aa_from = aa_from.upper()
    aa_to = aa_to.upper()
    return AMINO_ACID_CHARGE.get(aa_to, 0) - AMINO_ACID_CHARGE.get(aa_from, 0)


# =============================================================================
# Amino Acid Volumes (Side Chain)
# =============================================================================

# Side chain volumes in Å³ (from Zamyatnin, 1972)
AMINO_ACID_VOLUME: Dict[str, float] = {
    'A':  88.6,   # Alanine
    'R': 173.4,   # Arginine
    'N': 114.1,   # Asparagine
    'D': 111.1,   # Aspartic acid
    'C': 108.5,   # Cysteine
    'Q': 143.8,   # Glutamine
    'E': 138.4,   # Glutamic acid
    'G':  60.1,   # Glycine
    'H': 153.2,   # Histidine
    'I': 166.7,   # Isoleucine
    'L': 166.7,   # Leucine
    'K': 168.6,   # Lysine
    'M': 162.9,   # Methionine
    'F': 189.9,   # Phenylalanine
    'P': 112.7,   # Proline
    'S':  89.0,   # Serine
    'T': 116.1,   # Threonine
    'W': 227.8,   # Tryptophan
    'Y': 193.6,   # Tyrosine
    'V': 140.0,   # Valine
}


def get_volume_diff(aa_from: str, aa_to: str) -> float:
    """
    Get volume difference for a mutation.

    Args:
        aa_from: Original amino acid
        aa_to: Mutated amino acid

    Returns:
        Volume difference in Å³
    """
    aa_from = aa_from.upper()
    aa_to = aa_to.upper()
    return AMINO_ACID_VOLUME.get(aa_to, 0.0) - AMINO_ACID_VOLUME.get(aa_from, 0.0)


# =============================================================================
# Amino Acid Masses
# =============================================================================

# Molecular mass in Daltons (monoisotopic)
AMINO_ACID_MASS: Dict[str, float] = {
    'A':  71.04,   # Alanine
    'R': 156.10,   # Arginine
    'N': 114.04,   # Asparagine
    'D': 115.03,   # Aspartic acid
    'C': 103.01,   # Cysteine
    'Q': 128.06,   # Glutamine
    'E': 129.04,   # Glutamic acid
    'G':  57.02,   # Glycine
    'H': 137.06,   # Histidine
    'I': 113.08,   # Isoleucine
    'L': 113.08,   # Leucine
    'K': 128.09,   # Lysine
    'M': 131.04,   # Methionine
    'F': 147.07,   # Phenylalanine
    'P':  97.05,   # Proline
    'S':  87.03,   # Serine
    'T': 101.05,   # Threonine
    'W': 186.08,   # Tryptophan
    'Y': 163.06,   # Tyrosine
    'V':  99.07,   # Valine
}


# =============================================================================
# Combined Mutation Feature Extraction
# =============================================================================

def get_mutation_features(
    aa_from: str,
    aa_to: str,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Get all mutation features as a dictionary.

    Args:
        aa_from: Original amino acid (single letter)
        aa_to: Mutated amino acid (single letter)
        normalize: Whether to normalize features to roughly [-1, 1] range

    Returns:
        Dictionary with mutation features:
        - blosum: BLOSUM62 score
        - hydropathy_diff: Hydropathy difference
        - charge_diff: Charge difference
        - volume_diff: Volume difference
        - mass_diff: Mass difference
    """
    aa_from = aa_from.upper()
    aa_to = aa_to.upper()

    features = {
        'blosum': get_blosum_score(aa_from, aa_to),
        'hydropathy_diff': get_hydropathy_diff(aa_from, aa_to),
        'charge_diff': get_charge_diff(aa_from, aa_to),
        'volume_diff': get_volume_diff(aa_from, aa_to),
        'mass_diff': AMINO_ACID_MASS.get(aa_to, 0) - AMINO_ACID_MASS.get(aa_from, 0),
    }

    if normalize:
        # Normalize to roughly [-1, 1] range
        features['blosum'] = features['blosum'] / 11.0  # Max BLOSUM62 = 11
        features['hydropathy_diff'] = features['hydropathy_diff'] / 9.0  # Max diff ~9
        features['charge_diff'] = features['charge_diff'] / 2.0  # Max diff = 2
        features['volume_diff'] = features['volume_diff'] / 170.0  # Max diff ~170
        features['mass_diff'] = features['mass_diff'] / 130.0  # Max diff ~130

    return features


def get_mutation_feature_vector(
    aa_from: str,
    aa_to: str,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Get mutation features as a tensor.

    Args:
        aa_from: Original amino acid
        aa_to: Mutated amino acid
        normalize: Whether to normalize features
        device: Target device

    Returns:
        Tensor of shape [5] with features
    """
    features = get_mutation_features(aa_from, aa_to, normalize)
    vec = torch.tensor([
        features['blosum'],
        features['hydropathy_diff'],
        features['charge_diff'],
        features['volume_diff'],
        features['mass_diff'],
    ], dtype=torch.float32)

    if device is not None:
        vec = vec.to(device)

    return vec


def get_aa_embedding_matrix(
    normalize: bool = True,
) -> torch.Tensor:
    """
    Get embedding matrix for all amino acids based on physicochemical properties.

    Returns:
        Tensor of shape [20, 4] with features for each amino acid:
        [hydropathy, charge, volume, mass]
    """
    embeddings = []
    for aa in AMINO_ACIDS:
        features = [
            KYTE_DOOLITTLE_HYDROPATHY.get(aa, 0.0),
            AMINO_ACID_CHARGE.get(aa, 0),
            AMINO_ACID_VOLUME.get(aa, 0.0),
            AMINO_ACID_MASS.get(aa, 0.0),
        ]
        embeddings.append(features)

    matrix = torch.tensor(embeddings, dtype=torch.float32)

    if normalize:
        # Normalize each column
        matrix[:, 0] = matrix[:, 0] / 4.5  # Hydropathy max ~4.5
        matrix[:, 1] = matrix[:, 1]  # Charge already -1, 0, 1
        matrix[:, 2] = matrix[:, 2] / 230.0  # Volume max ~230
        matrix[:, 3] = matrix[:, 3] / 190.0  # Mass max ~190

    return matrix
