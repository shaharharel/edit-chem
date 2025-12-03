"""
Antibody numbering and CDR region detection.

Provides IMGT numbering and CDR region identification using ANARCI/AbNumber
or fallback heuristics.

IMGT numbering:
- Standard positions 1-128 for variable regions
- Structurally equivalent positions across antibodies
- CDR regions at defined positions

CDR Regions (IMGT):
- CDR1: 27-38
- CDR2: 56-65
- CDR3: 105-117

Framework Regions (IMGT):
- FR1: 1-26
- FR2: 39-55
- FR3: 66-104
- FR4: 118-128
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import warnings


class CDRRegion(Enum):
    """CDR and framework region labels."""
    FR1 = "FR1"
    CDR1 = "CDR1"
    FR2 = "FR2"
    CDR2 = "CDR2"
    FR3 = "FR3"
    CDR3 = "CDR3"
    FR4 = "FR4"


# IMGT CDR boundaries
IMGT_CDR_RANGES = {
    'CDR1': (27, 38),
    'CDR2': (56, 65),
    'CDR3': (105, 117),
    'FR1': (1, 26),
    'FR2': (39, 55),
    'FR3': (66, 104),
    'FR4': (118, 128),
}


@dataclass
class IMGTPosition:
    """
    IMGT-numbered position for an antibody residue.

    Attributes:
        position: IMGT position number (1-128)
        insertion: Insertion code (e.g., 'A', 'B') for insertions
        chain: Chain type ('H' for heavy, 'L' for light)
        region: CDR/FR region label
        sequence_position: Original 0-indexed position in sequence
    """
    position: int
    insertion: Optional[str] = None
    chain: str = 'H'
    region: Optional[CDRRegion] = None
    sequence_position: Optional[int] = None

    @property
    def full_position(self) -> str:
        """Get full position string (e.g., '111A')."""
        if self.insertion:
            return f"{self.position}{self.insertion}"
        return str(self.position)

    def __str__(self) -> str:
        return f"{self.chain}{self.full_position}"


def _try_import_abnumber():
    """Try to import abnumber."""
    try:
        from abnumber import Chain
        return Chain
    except ImportError:
        return None


def _try_import_anarci():
    """Try to import ANARCI directly."""
    try:
        import anarci
        return anarci
    except ImportError:
        return None


def number_antibody(
    sequence: str,
    chain_type: str = 'H',
    scheme: str = 'imgt',
) -> List[IMGTPosition]:
    """
    Number an antibody sequence using IMGT (or other) scheme.

    Args:
        sequence: Amino acid sequence
        chain_type: 'H' for heavy chain, 'L' for light chain
        scheme: Numbering scheme ('imgt', 'chothia', 'kabat')

    Returns:
        List of IMGTPosition objects for each residue

    Note:
        Falls back to heuristic numbering if ANARCI/AbNumber not available.
    """
    Chain = _try_import_abnumber()

    if Chain is not None:
        return _number_with_abnumber(sequence, chain_type, scheme)

    anarci = _try_import_anarci()
    if anarci is not None:
        return _number_with_anarci(sequence, chain_type, scheme)

    # Fallback to heuristic
    warnings.warn(
        "ANARCI/AbNumber not available. Using heuristic numbering. "
        "Install with: conda install -c bioconda abnumber"
    )
    return _number_heuristic(sequence, chain_type)


def _number_with_abnumber(
    sequence: str,
    chain_type: str,
    scheme: str,
) -> List[IMGTPosition]:
    """Number using AbNumber."""
    from abnumber import Chain

    try:
        chain = Chain(sequence, scheme=scheme)
    except Exception as e:
        warnings.warn(f"AbNumber failed: {e}. Using heuristic.")
        return _number_heuristic(sequence, chain_type)

    positions = []
    for i, (pos, aa) in enumerate(chain):
        # pos is like 'H1' or 'L1' or '1A' for insertions
        pos_str = str(pos)

        # Parse position number and insertion
        num_str = ''
        insertion = None
        for c in pos_str:
            if c.isdigit():
                num_str += c
            elif c.isalpha() and len(num_str) > 0:
                insertion = c

        position = int(num_str) if num_str else i + 1
        region = get_region_label(position)

        positions.append(IMGTPosition(
            position=position,
            insertion=insertion,
            chain=chain_type,
            region=region,
            sequence_position=i,
        ))

    return positions


def _number_with_anarci(
    sequence: str,
    chain_type: str,
    scheme: str,
) -> List[IMGTPosition]:
    """Number using ANARCI directly."""
    import anarci

    # Run ANARCI
    results = anarci.run_anarci(
        [('seq', sequence)],
        scheme=scheme,
        allowed_species=['human', 'mouse'],
    )

    if not results or not results[0]:
        warnings.warn("ANARCI returned no results. Using heuristic.")
        return _number_heuristic(sequence, chain_type)

    numbering = results[0][0][0]  # First sequence, first domain, numbering

    positions = []
    seq_pos = 0
    for (pos_num, insertion), aa in numbering:
        if aa == '-':
            continue

        region = get_region_label(pos_num)

        positions.append(IMGTPosition(
            position=pos_num,
            insertion=insertion if insertion != ' ' else None,
            chain=chain_type,
            region=region,
            sequence_position=seq_pos,
        ))
        seq_pos += 1

    return positions


def _number_heuristic(
    sequence: str,
    chain_type: str,
) -> List[IMGTPosition]:
    """
    Heuristic numbering without external tools.

    This is a simplified approach that assigns positions sequentially
    and uses approximate CDR boundaries based on typical antibody lengths.
    """
    positions = []
    L = len(sequence)

    # Approximate IMGT position mapping for typical antibody
    # Heavy chain variable region: ~120-130 residues
    # Light chain variable region: ~105-115 residues

    for i in range(L):
        # Simple linear mapping to IMGT positions
        # This is approximate and won't handle insertions correctly
        if chain_type == 'H':
            # Heavy chain heuristic
            if L <= 130:
                imgt_pos = i + 1
            else:
                imgt_pos = int((i / L) * 128) + 1
        else:
            # Light chain heuristic
            if L <= 115:
                imgt_pos = i + 1
            else:
                imgt_pos = int((i / L) * 128) + 1

        imgt_pos = min(max(imgt_pos, 1), 128)
        region = get_region_label(imgt_pos)

        positions.append(IMGTPosition(
            position=imgt_pos,
            insertion=None,
            chain=chain_type,
            region=region,
            sequence_position=i,
        ))

    return positions


def get_region_label(imgt_position: int) -> CDRRegion:
    """
    Get the CDR/FR region for an IMGT position.

    Args:
        imgt_position: IMGT position number (1-128)

    Returns:
        CDRRegion enum value
    """
    if 1 <= imgt_position <= 26:
        return CDRRegion.FR1
    elif 27 <= imgt_position <= 38:
        return CDRRegion.CDR1
    elif 39 <= imgt_position <= 55:
        return CDRRegion.FR2
    elif 56 <= imgt_position <= 65:
        return CDRRegion.CDR2
    elif 66 <= imgt_position <= 104:
        return CDRRegion.FR3
    elif 105 <= imgt_position <= 117:
        return CDRRegion.CDR3
    elif 118 <= imgt_position <= 128:
        return CDRRegion.FR4
    else:
        return CDRRegion.FR4  # Default for positions outside range


def get_imgt_position(
    sequence_position: int,
    numbering: List[IMGTPosition],
) -> Optional[IMGTPosition]:
    """
    Get IMGT position for a sequence position.

    Args:
        sequence_position: 0-indexed position in sequence
        numbering: List of IMGTPosition from number_antibody()

    Returns:
        IMGTPosition or None if not found
    """
    for pos in numbering:
        if pos.sequence_position == sequence_position:
            return pos
    return None


def get_cdr_regions(
    sequence: str,
    chain_type: str = 'H',
    scheme: str = 'imgt',
) -> Dict[str, Tuple[int, int]]:
    """
    Get CDR region boundaries for a sequence.

    Args:
        sequence: Amino acid sequence
        chain_type: 'H' or 'L'
        scheme: Numbering scheme

    Returns:
        Dictionary mapping region names to (start, end) positions
        in the original sequence (0-indexed, inclusive)
    """
    numbering = number_antibody(sequence, chain_type, scheme)

    regions = {}
    for region in CDRRegion:
        # Find first and last position in this region
        positions_in_region = [
            p.sequence_position for p in numbering
            if p.region == region and p.sequence_position is not None
        ]

        if positions_in_region:
            regions[region.value] = (min(positions_in_region), max(positions_in_region))

    return regions


def is_in_cdr(
    sequence_position: int,
    numbering: List[IMGTPosition],
) -> bool:
    """
    Check if a sequence position is in a CDR region.

    Args:
        sequence_position: 0-indexed position
        numbering: IMGT numbering from number_antibody()

    Returns:
        True if position is in CDR1, CDR2, or CDR3
    """
    pos = get_imgt_position(sequence_position, numbering)
    if pos is None:
        return False

    return pos.region in [CDRRegion.CDR1, CDRRegion.CDR2, CDRRegion.CDR3]


def get_region_one_hot(
    sequence_position: int,
    numbering: List[IMGTPosition],
) -> List[int]:
    """
    Get one-hot encoding for the region of a position.

    Args:
        sequence_position: 0-indexed position
        numbering: IMGT numbering

    Returns:
        One-hot vector [FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4]
    """
    pos = get_imgt_position(sequence_position, numbering)

    region_order = [
        CDRRegion.FR1, CDRRegion.CDR1, CDRRegion.FR2, CDRRegion.CDR2,
        CDRRegion.FR3, CDRRegion.CDR3, CDRRegion.FR4
    ]

    one_hot = [0] * 7
    if pos is not None and pos.region is not None:
        try:
            idx = region_order.index(pos.region)
            one_hot[idx] = 1
        except ValueError:
            pass

    return one_hot
