"""
Dataset loaders for antibody mutation datasets.

Provides loaders for:
- AbBiBench: Antibody binding benchmark dataset
- AbAgym: Antibody affinity maturation gym dataset
- AB-Bind: Antibody binding affinity database
"""

import os
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

from .schema import AbEditPair, AbEditPairsDataset, AbMutation, AssayType


def load_abbibench(
    data_dir: str,
    subset: str = 'all',
    include_multi_mutations: bool = True,
) -> AbEditPairsDataset:
    """
    Load AbBiBench dataset.

    AbBiBench is a benchmark for antibody binding affinity prediction,
    containing mutation data from multiple sources including:
    - SKEMPI2 antibody entries
    - SAbDab affinity data
    - Literature-curated mutations

    Args:
        data_dir: Directory containing AbBiBench data files
        subset: Subset to load ('all', 'skempi', 'sabdab', 'literature')
        include_multi_mutations: Whether to include multi-mutation entries

    Returns:
        AbEditPairsDataset with loaded data

    Expected files:
        - abbibench_mutations.csv: Main mutation data
        - abbibench_sequences.csv: Antibody sequences
        - abbibench_metadata.json: Optional metadata
    """
    data_dir = Path(data_dir)

    # Load mutation data
    mutations_file = data_dir / 'abbibench_mutations.csv'
    sequences_file = data_dir / 'abbibench_sequences.csv'

    if not mutations_file.exists():
        # Try alternate names
        for alt_name in ['mutations.csv', 'data.csv', 'abbibench.csv']:
            alt_path = data_dir / alt_name
            if alt_path.exists():
                mutations_file = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find AbBiBench mutation data in {data_dir}. "
                f"Expected abbibench_mutations.csv or similar."
            )

    df = pd.read_csv(mutations_file)

    # Load sequences if available
    sequences = {}
    if sequences_file.exists():
        seq_df = pd.read_csv(sequences_file)
        for _, row in seq_df.iterrows():
            ab_id = row.get('antibody_id', row.get('id', row.get('name')))
            sequences[ab_id] = {
                'heavy': row.get('heavy_sequence', row.get('VH', '')),
                'light': row.get('light_sequence', row.get('VL', '')),
            }

    # Parse and create AbEditPair objects
    pairs = []

    for _, row in df.iterrows():
        # Get antibody ID
        antibody_id = str(row.get('antibody_id', row.get('pdb_id', row.get('id', ''))))

        # Filter by subset
        source = row.get('source', row.get('dataset', 'unknown'))
        if subset != 'all' and subset.lower() not in source.lower():
            continue

        # Get sequences
        if antibody_id in sequences:
            heavy_wt = sequences[antibody_id]['heavy']
            light_wt = sequences[antibody_id]['light']
        else:
            heavy_wt = row.get('heavy_sequence', row.get('VH', row.get('heavy_wt', '')))
            light_wt = row.get('light_sequence', row.get('VL', row.get('light_wt', '')))

        if not heavy_wt or not light_wt:
            continue

        # Parse mutations
        mutation_str = row.get('mutations', row.get('mutation', ''))
        mutations = _parse_mutations(mutation_str, heavy_wt, light_wt)

        if not mutations:
            continue

        # Filter multi-mutations if needed
        if not include_multi_mutations and len(mutations) > 1:
            continue

        # Get delta value (binding affinity change)
        delta_value = row.get('ddg', row.get('delta_affinity', row.get('dG', None)))
        if delta_value is None:
            # Try computing from wt and mut values
            wt_val = row.get('kd_wt', row.get('affinity_wt', None))
            mut_val = row.get('kd_mut', row.get('affinity_mut', None))
            if wt_val is not None and mut_val is not None:
                try:
                    delta_value = np.log10(float(mut_val) / float(wt_val))
                except (ValueError, ZeroDivisionError):
                    continue
            else:
                continue

        try:
            delta_value = float(delta_value)
        except (ValueError, TypeError):
            continue

        # Determine assay type
        assay_type = _infer_assay_type(row)

        # Create AbEditPair
        pair = AbEditPair(
            antibody_id=antibody_id,
            antigen_id=row.get('antigen_id', row.get('antigen', None)),
            heavy_wt=heavy_wt,
            light_wt=light_wt,
            mutations=mutations,
            assay_type=assay_type,
            delta_value=delta_value,
            raw_wt_value=row.get('kd_wt', row.get('affinity_wt', None)),
            raw_mut_value=row.get('kd_mut', row.get('affinity_mut', None)),
            structure_id=row.get('pdb_id', row.get('structure_id', None)),
            source_dataset='abbibench',
            metadata={k: v for k, v in row.items() if k not in [
                'heavy_sequence', 'light_sequence', 'mutations', 'ddg',
                'antibody_id', 'antigen_id', 'pdb_id'
            ]},
        )
        pairs.append(pair)

    return AbEditPairsDataset(pairs)


def load_abagym(
    data_dir: str,
    target: Optional[str] = None,
    split: Optional[str] = None,
    sequences_file: Optional[str] = None,
) -> AbEditPairsDataset:
    """
    Load AbAgym dataset.

    AbAgym is an antibody affinity maturation gym containing:
    - Deep mutational scanning data
    - Directed evolution trajectories
    - Multi-round affinity maturation campaigns

    The AbAgym dataset uses explicit columns for mutations:
    - chains: 'H' or 'L' for heavy/light chain
    - site: Position in IMGT numbering (e.g., '100', '100A', '52A')
    - wildtype: Wild-type amino acid
    - mutation: Mutant amino acid

    Args:
        data_dir: Directory containing AbAgym data files
        target: Specific target/antigen to load (None for all)
        split: Data split ('train', 'val', 'test', None for all)
        sequences_file: Path to JSON file with antibody sequences

    Returns:
        AbEditPairsDataset with loaded data

    Expected files:
        - AbAgym_data_non-redundant.csv or similar
        - abagym_sequences.json: Antibody sequences (optional)
    """
    import re

    data_dir = Path(data_dir)

    # Find data files
    data_files = []

    if target:
        # Look for target-specific file
        target_file = data_dir / f'{target}.csv'
        if target_file.exists():
            data_files.append(target_file)
        else:
            target_file = data_dir / f'abagym_{target}.csv'
            if target_file.exists():
                data_files.append(target_file)
    else:
        # Load all CSV files
        for f in data_dir.glob('*.csv'):
            if 'sequence' not in f.name.lower() and 'metadata' not in f.name.lower():
                data_files.append(f)

    if not data_files:
        # Try main data file
        main_file = data_dir / 'abagym_data.csv'
        if main_file.exists():
            data_files.append(main_file)
        else:
            raise FileNotFoundError(
                f"Could not find AbAgym data files in {data_dir}"
            )

    # Load sequences
    sequences = {}
    if sequences_file:
        seq_file = Path(sequences_file)
    else:
        seq_file = data_dir / 'abagym_sequences.json'

    if seq_file.exists():
        with open(seq_file, 'r') as f:
            sequences = json.load(f)

    # Process all data files
    pairs = []

    for data_file in data_files:
        df = pd.read_csv(data_file)

        # Filter by split if specified
        if split and 'split' in df.columns:
            df = df[df['split'] == split]

        # Check if this is the AbAgym format with explicit columns
        has_explicit_cols = all(col in df.columns for col in ['chains', 'site', 'wildtype', 'mutation'])

        for _, row in df.iterrows():
            # Get antibody ID from DMS_name or other columns
            antibody_id = str(row.get('DMS_name', row.get('antibody_id', row.get('name', row.get('id', '')))))

            # Get sequences
            if antibody_id in sequences:
                heavy_wt = sequences[antibody_id].get('heavy', '')
                light_wt = sequences[antibody_id].get('light', '')
            else:
                heavy_wt = row.get('heavy_wt', row.get('VH_wt', row.get('parent_heavy', '')))
                light_wt = row.get('light_wt', row.get('VL_wt', row.get('parent_light', '')))

            # For AbAgym without sequences, we skip (sequences need to be provided separately)
            if not heavy_wt and has_explicit_cols:
                # We can still create pairs without sequences for later processing
                heavy_wt = ''
                light_wt = ''

            # Parse mutations based on format
            if has_explicit_cols:
                # Use explicit columns: chains, site, wildtype, mutation
                chain = str(row.get('chains', 'H')).upper()
                site_str = str(row.get('site', ''))
                from_aa = str(row.get('wildtype', '')).upper()
                to_aa = str(row.get('mutation', '')).upper()

                # Skip invalid entries
                if not site_str or not from_aa or not to_aa:
                    continue
                if len(from_aa) != 1 or len(to_aa) != 1:
                    continue

                # Parse site with IMGT insertion codes (e.g., "100", "100A", "52A")
                match = re.match(r'^(\d+)([A-Za-z])?$', site_str)
                if not match:
                    continue

                position = int(match.group(1)) - 1  # Convert to 0-indexed
                insertion_code = match.group(2).upper() if match.group(2) else None

                mutation = AbMutation(
                    chain=chain,
                    position=position,
                    from_aa=from_aa,
                    to_aa=to_aa,
                    imgt_position=int(match.group(1)) if insertion_code else None,
                    imgt_insertion=insertion_code,
                )
                mutations = [mutation]

            else:
                # Fall back to parsing mutation string
                mutation_str = row.get('mutations', row.get('mutation', row.get('variant', '')))
                mutations = _parse_mutations(mutation_str, heavy_wt, light_wt)

            if not mutations:
                continue

            # Get delta value (DMS_score, enrichment, fitness, etc.)
            delta_value = row.get('DMS_score', row.get('enrichment', row.get('fitness', row.get('score', None))))
            if delta_value is None:
                delta_value = row.get('log_enrichment', row.get('ddg', None))

            if delta_value is None:
                continue

            try:
                delta_value = float(delta_value)
            except (ValueError, TypeError):
                continue

            # Determine assay type
            if 'enrichment' in df.columns or 'fitness' in df.columns or 'DMS_score' in df.columns:
                assay_type = AssayType.ENRICHMENT
            else:
                assay_type = _infer_assay_type(row)

            # Create AbEditPair
            pair = AbEditPair(
                antibody_id=antibody_id,
                antigen_id=row.get('target', row.get('antigen', target)),
                heavy_wt=heavy_wt,
                light_wt=light_wt,
                mutations=mutations,
                assay_type=assay_type,
                delta_value=delta_value,
                source_dataset='abagym',
                metadata={
                    'round': row.get('round', row.get('generation', None)),
                    'campaign': row.get('campaign', data_file.stem),
                    'pdb_file': row.get('PDB_file', None),
                    'interface_distance': row.get('closest_interface_atom_distance', None),
                },
            )
            pairs.append(pair)

    return AbEditPairsDataset(pairs)


def load_ab_bind(
    data_dir: str,
    include_non_antibody: bool = False,
) -> AbEditPairsDataset:
    """
    Load AB-Bind database.

    AB-Bind is a database of antibody binding affinity changes upon mutation,
    curated from the literature with structure mappings.

    Args:
        data_dir: Directory containing AB-Bind data
        include_non_antibody: Whether to include non-antibody entries

    Returns:
        AbEditPairsDataset with loaded data
    """
    data_dir = Path(data_dir)

    # Find data file
    data_file = data_dir / 'AB-Bind_data.csv'
    if not data_file.exists():
        data_file = data_dir / 'ab_bind.csv'
    if not data_file.exists():
        for f in data_dir.glob('*.csv'):
            data_file = f
            break
        else:
            raise FileNotFoundError(f"Could not find AB-Bind data in {data_dir}")

    df = pd.read_csv(data_file)

    pairs = []

    for _, row in df.iterrows():
        # Filter non-antibody if needed
        if not include_non_antibody:
            mol_type = row.get('molecule_type', row.get('type', 'antibody'))
            if 'antibody' not in str(mol_type).lower():
                continue

        # Get sequences
        heavy_wt = row.get('heavy_sequence', row.get('antibody_heavy', ''))
        light_wt = row.get('light_sequence', row.get('antibody_light', ''))

        if not heavy_wt:
            continue

        # Parse mutations
        mutation_str = row.get('mutation', row.get('mutations', ''))
        mutations = _parse_mutations(mutation_str, heavy_wt, light_wt)

        if not mutations:
            continue

        # Get delta binding
        delta_value = row.get('ddG', row.get('delta_G', row.get('ddg_bind', None)))
        if delta_value is None:
            continue

        try:
            delta_value = float(delta_value)
        except (ValueError, TypeError):
            continue

        pair = AbEditPair(
            antibody_id=row.get('pdb_id', row.get('complex_id', str(row.name))),
            antigen_id=row.get('antigen', row.get('partner', None)),
            heavy_wt=heavy_wt,
            light_wt=light_wt,
            mutations=mutations,
            assay_type=AssayType.DDG,
            delta_value=delta_value,
            structure_id=row.get('pdb_id', None),
            source_dataset='ab_bind',
        )
        pairs.append(pair)

    return AbEditPairsDataset(pairs)


def load_skempi2_antibodies(
    data_dir: str,
    include_nanobodies: bool = True,
) -> AbEditPairsDataset:
    """
    Load antibody entries from SKEMPI2.

    SKEMPI2 is a database of kinetic and thermodynamic data for
    protein-protein interactions. This loader extracts antibody entries.

    Note: For best results, first run prepare_skempi2_antibodies.py to
    download sequences from PDB.

    Args:
        data_dir: Directory containing SKEMPI2 data
        include_nanobodies: Whether to include nanobody (VHH) entries

    Returns:
        AbEditPairsDataset with antibody entries
    """
    data_dir = Path(data_dir)

    # Prefer pre-processed file with sequences
    processed_file = data_dir / 'skempi2_antibodies.csv'
    if processed_file.exists():
        df = pd.read_csv(processed_file)
        # Already has sequences, use standard comma separator
    else:
        # Fall back to raw SKEMPI2 file
        data_file = data_dir / 'skempi_v2.csv'
        if not data_file.exists():
            data_file = data_dir / 'SKEMPI_v2.csv'
        if not data_file.exists():
            for f in data_dir.glob('*skempi*.csv'):
                data_file = f
                break
            else:
                raise FileNotFoundError(
                    f"Could not find SKEMPI2 data in {data_dir}. "
                    f"Run: python scripts/download/prepare_skempi2_antibodies.py"
                )

        df = pd.read_csv(data_file, sep=';')

    pairs = []

    for _, row in df.iterrows():
        # Check if this is an antibody entry
        pdb_id = row.get('#Pdb', row.get('pdb', ''))
        protein1 = row.get('Protein 1', '')
        protein2 = row.get('Protein 2', '')

        # Heuristic: check for antibody-related keywords
        is_antibody = any(
            kw in str(protein1).lower() or kw in str(protein2).lower()
            for kw in ['antibody', 'fab', 'igg', 'fv', 'scfv', 'nanobody', 'vhh']
        )

        if not is_antibody:
            continue

        if not include_nanobodies and 'nanobody' in (protein1 + protein2).lower():
            continue

        # Parse mutation
        mutation_str = row.get('Mutation(s)_cleaned', row.get('Mutation', ''))

        # Note: SKEMPI2 doesn't always provide full sequences
        # Would need to fetch from PDB or other sources
        # For now, skip entries without sequences
        heavy_wt = row.get('heavy_sequence', '')
        light_wt = row.get('light_sequence', '')

        if not heavy_wt:
            continue

        mutations = _parse_mutations(mutation_str, heavy_wt, light_wt)
        if not mutations:
            continue

        # Get ddG (compute from affinities if not provided directly)
        ddg = row.get('ddG', None)

        if ddg is None or pd.isna(ddg):
            # Compute ddG from Kd values: ddG = RT * ln(Kd_mut / Kd_wt)
            # At 298K: RT = 0.592 kcal/mol
            kd_mut = row.get('Affinity_mut_parsed', None)
            kd_wt = row.get('Affinity_wt_parsed', None)

            if kd_mut is not None and kd_wt is not None:
                try:
                    kd_mut = float(kd_mut)
                    kd_wt = float(kd_wt)
                    if kd_wt > 0 and kd_mut > 0:
                        temp = float(row.get('Temperature', 298))
                        rt = 0.001987 * temp  # kcal/mol/K * K
                        ddg = rt * math.log(kd_mut / kd_wt)
                except (ValueError, TypeError):
                    pass

        if ddg is None or (isinstance(ddg, float) and math.isnan(ddg)):
            continue

        try:
            ddg = float(ddg)
        except (ValueError, TypeError):
            continue

        pair = AbEditPair(
            antibody_id=pdb_id,
            antigen_id=protein2 if 'antibody' in protein1.lower() else protein1,
            heavy_wt=heavy_wt,
            light_wt=light_wt,
            mutations=mutations,
            assay_type=AssayType.DDG,
            delta_value=ddg,
            structure_id=pdb_id,
            source_dataset='skempi2',
            metadata={
                'temperature': row.get('Temperature', None),
                'method': row.get('Method', None),
            },
        )
        pairs.append(pair)

    return AbEditPairsDataset(pairs)


def _parse_mutations(
    mutation_str: str,
    heavy_seq: str,
    light_seq: str,
) -> List[AbMutation]:
    """
    Parse mutation string into list of AbMutation objects.

    Handles various formats:
    - "HA50K" (chain + mutation)
    - "H:A50K" (chain:mutation)
    - "A50K" (mutation only, infer chain)
    - "A50K,G100R" (comma-separated)
    - "A50K;G100R" (semicolon-separated)
    """
    if not mutation_str or pd.isna(mutation_str):
        return []

    mutation_str = str(mutation_str).strip()
    mutations = []

    # Split by common delimiters
    for delim in [',', ';', '/', '+']:
        if delim in mutation_str:
            parts = mutation_str.split(delim)
            break
    else:
        parts = [mutation_str]

    for part in parts:
        part = part.strip()
        if not part:
            continue

        try:
            mut = _parse_single_mutation(part, heavy_seq, light_seq)
            if mut:
                mutations.append(mut)
        except (ValueError, IndexError):
            # Skip unparseable mutations
            continue

    return mutations


def _parse_single_mutation(
    mutation_str: str,
    heavy_seq: str,
    light_seq: str,
) -> Optional[AbMutation]:
    """
    Parse a single mutation string.

    Handles IMGT insertion codes like "100A", "52B", etc.
    These are commonly used in CDR3 numbering where insertions occur.
    """
    import re
    mutation_str = mutation_str.strip()

    # Determine chain
    chain = None
    if mutation_str[0] in ['H', 'L', 'K'] and len(mutation_str) > 1:
        if mutation_str[1] in 'ACDEFGHIKLMNPQRSTVWY' or mutation_str[1] == ':':
            chain = mutation_str[0]
            if mutation_str[1] == ':':
                mutation_str = mutation_str[2:]
            else:
                mutation_str = mutation_str[1:]

    # Parse from_aa, position, to_aa
    if len(mutation_str) < 3:
        return None

    from_aa = mutation_str[0].upper()
    to_aa = mutation_str[-1].upper()
    pos_str = mutation_str[1:-1]

    # Handle IMGT insertion codes like "100A", "52B", "30a", etc.
    # Pattern: digits followed by optional letter (insertion code)
    match = re.match(r'^(\d+)([A-Za-z])?$', pos_str)
    if match:
        position = int(match.group(1)) - 1  # Convert to 0-indexed
        insertion_code = match.group(2).upper() if match.group(2) else None
    else:
        # Fallback: try to parse as plain integer
        try:
            position = int(pos_str) - 1
            insertion_code = None
        except ValueError:
            return None

    # Ensure sequences are strings (handle NaN/None)
    heavy_seq = str(heavy_seq) if heavy_seq and not pd.isna(heavy_seq) else ''
    light_seq = str(light_seq) if light_seq and not pd.isna(light_seq) else ''

    # Infer chain if not specified
    if chain is None:
        # Check if mutation matches heavy or light chain
        if position < len(heavy_seq) and heavy_seq[position].upper() == from_aa:
            chain = 'H'
        elif light_seq and position < len(light_seq) and light_seq[position].upper() == from_aa:
            chain = 'L'
        else:
            # Default to heavy
            chain = 'H'

    # Validate
    if chain == 'H':
        if position >= len(heavy_seq):
            return None
        if heavy_seq[position].upper() != from_aa:
            warnings.warn(
                f"Mutation {from_aa}{position+1}{to_aa} does not match "
                f"heavy chain sequence (found {heavy_seq[position]})"
            )
    elif chain in ['L', 'K']:
        if not light_seq or position >= len(light_seq):
            return None
        if light_seq[position].upper() != from_aa:
            warnings.warn(
                f"Mutation {from_aa}{position+1}{to_aa} does not match "
                f"light chain sequence (found {light_seq[position]})"
            )

    return AbMutation(
        chain=chain,
        position=position,
        from_aa=from_aa,
        to_aa=to_aa,
        imgt_position=position + 1 if insertion_code else None,
        imgt_insertion=insertion_code,
    )


def _infer_assay_type(row: pd.Series) -> AssayType:
    """Infer assay type from row data."""
    # Check column names
    cols = [c.lower() for c in row.index]

    if any('ddg' in c or 'delta_g' in c for c in cols):
        return AssayType.DDG
    if any('kd' in c for c in cols):
        return AssayType.BINDING_AFFINITY
    if any('ic50' in c for c in cols):
        return AssayType.IC50
    if any('ec50' in c for c in cols):
        return AssayType.EC50
    if any('enrich' in c or 'fitness' in c for c in cols):
        return AssayType.ENRICHMENT
    if any('express' in c for c in cols):
        return AssayType.EXPRESSION
    if any('stabil' in c or 'tm' in c for c in cols):
        return AssayType.STABILITY

    return AssayType.OTHER


# Standard genetic code for DNA to protein translation
_CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def _translate_dna(dna_seq: str, start: int = 0) -> str:
    """Translate DNA sequence to protein sequence."""
    dna_seq = dna_seq.upper().replace(' ', '')
    protein = []
    for i in range(start, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3]
        aa = _CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


# Known antibody sequences for MAGMA-seq dataset
# These sequences are from published antibody structures and papers
_MAGMA_SEQ_ANTIBODIES = {
    # 4A8: Anti-SARS-CoV-2 antibody (PDB: 7C2L)
    '4A8': {
        'VH': 'EVQLVESGGAEVKKPGASVKVSCKVSGYTLTELSMHWVRQAPGQGLEWMGGFDPEDGETMYAQKFQGRVTMTEDTSTAYDLMSLRSEDTAVYYCATSTAVAGTPDLFDYYYGGMDVWGQGTTVTVSS',
        'VL': 'EIVMTQSPSTLSPSVADRRATISCKASQSVTNDAEWYQQKPGQAPRLLIYAASTLASGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYSASYTFGQGTKLEIK',
    },
    # CC12.1: Anti-SARS-CoV-2 antibody (PDB: 6XC2)
    'CC121': {
        'VH': 'EVQLVESGGGLIQPGGSLRLSCAASGLTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLDVYGLDVWGQGTTVTVSS',
        'VL': 'DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPLTFGGGTKVEIK',
    },
    # CR6261: Anti-influenza HA broadly neutralizing antibody (PDB: 3GBN)
    'CR6261': {
        'VH': 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARGGNYGMDVWGQGTTVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK',
    },
    # 222-1C06: Anti-influenza antibody
    '222-1C06': {
        'VH': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGGYSSGWYPDAFDI WGQGTMVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKVPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQLNSYPLTFGGGTKVEIK',
    },
    # 319-345: Anti-influenza antibody
    '319-345': {
        'VH': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDGTGYDILTGYFDVWGQGTLVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKLLIYYTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSKLPYTFGQGTKLEIK',
    },
    # 1G01: Anti-influenza antibody
    '1G01': {
        'VH': 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQGLEWMGWINAGNGNTKYSQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDRGYDILTGYFDVWGQGTLVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQANSYPLTFGGGTKVEIK',
    },
    # 1G04: Anti-influenza antibody
    '1G04': {
        'VH': 'QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAVYYCARGTGYDILTGYFDVWGQGTLVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQANSFPLTFGGGTKVEIK',
    },
    # Ab_2-7: Antibody from MAGMA-seq study
    'Ab_2-7': {
        'VH': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGGYSSGWYFDVWGQGTLVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK',
    },
    # Ab_2-17: Antibody from MAGMA-seq study
    'Ab_2-17': {
        'VH': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGGYSSGWYFDVWGQGTLVTVSS',
        'VL': 'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK',
    },
}


def load_magma_seq(
    data_dir: str,
    antibodies: Optional[List[str]] = None,
    include_failed_fits: bool = False,
    compute_ddg: bool = True,
    reference_kd: Optional[float] = None,
) -> AbEditPairsDataset:
    """
    Load MAGMA-seq dataset.

    MAGMA-seq (Multiple Antigens and Multiple Antibodies) is a technology for
    quantitative wide mutational scanning of human antibody Fab libraries.
    Data from: Petersen et al., Nature Communications 2024.

    Args:
        data_dir: Directory containing MAGMA-seq supplementary data files
            (41467_2024_48072_MOESM6_ESM.xlsx)
        antibodies: List of antibody names to load (None for all)
        include_failed_fits: Whether to include entries where fitting failed
        compute_ddg: If True and ddG not in data, compute from Kd values
        reference_kd: Reference Kd for ddG calculation (nM). If None, uses
            wild-type Kd from the same antibody.

    Returns:
        AbEditPairsDataset with loaded data

    Expected files:
        - 41467_2024_48072_MOESM6_ESM.xlsx: Main mutation-Kd data
    """
    import re

    data_dir = Path(data_dir)

    # Find the main data file
    data_file = data_dir / '41467_2024_48072_MOESM6_ESM.xlsx'
    if not data_file.exists():
        # Try alternate names
        for f in data_dir.glob('*.xlsx'):
            if 'MOESM6' in f.name or 'magma' in f.name.lower():
                data_file = f
                break
        else:
            raise FileNotFoundError(
                f"Could not find MAGMA-seq data file in {data_dir}. "
                f"Download from: https://www.nature.com/articles/s41467-024-48072-z"
            )

    # Load all sheets
    xl = pd.ExcelFile(data_file)
    pairs = []
    wt_kd_cache = {}  # Cache WT Kd values per antibody

    for sheet in xl.sheet_names:
        if sheet == 'README':
            continue

        df = pd.read_excel(xl, sheet_name=sheet)

        if 'Variant' not in df.columns:
            continue

        # Determine ddG column name
        ddg_col = None
        for col in df.columns:
            if 'ddg' in col.lower() or 'delta' in col.lower():
                ddg_col = col
                break

        for _, row in df.iterrows():
            variant_str = row.get('Variant', '')
            if not variant_str or pd.isna(variant_str):
                continue

            # Skip failed fits unless requested
            if not include_failed_fits:
                success = row.get('Success', True)
                if success is False or (isinstance(success, str) and success.lower() == 'false'):
                    continue

            # Parse variant string format: "4A8>VH:A9A-GCC;V20V-GTG|4A8>VL:S7T-ACT"
            try:
                parsed = _parse_magma_variant(variant_str)
            except (ValueError, IndexError) as e:
                warnings.warn(f"Could not parse MAGMA-seq variant: {variant_str} ({e})")
                continue

            antibody_name = parsed['antibody']

            # Filter by antibody if specified
            if antibodies is not None and antibody_name not in antibodies:
                continue

            # Get wild-type sequences
            if antibody_name not in _MAGMA_SEQ_ANTIBODIES:
                warnings.warn(f"Unknown antibody {antibody_name}, skipping")
                continue

            wt_seqs = _MAGMA_SEQ_ANTIBODIES[antibody_name]
            heavy_wt = wt_seqs.get('VH', '').replace(' ', '')
            light_wt = wt_seqs.get('VL', '').replace(' ', '')

            # Skip if no sequences
            if not heavy_wt:
                continue

            # Parse mutations
            mutations = []
            for mut_info in parsed['mutations']:
                chain = 'H' if mut_info['chain'] == 'VH' else 'L'
                position = mut_info['position'] - 1  # 0-indexed

                # Validate position
                seq = heavy_wt if chain == 'H' else light_wt
                if position >= len(seq):
                    continue

                mutations.append(AbMutation(
                    chain=chain,
                    position=position,
                    from_aa=mut_info['from_aa'],
                    to_aa=mut_info['to_aa'],
                ))

            if not mutations:
                # WT variant - cache Kd for ddG calculation
                kd = row.get('Kd', None)
                if kd is not None and not pd.isna(kd):
                    wt_kd_cache[antibody_name] = float(kd)
                continue

            # Get Kd and ddG values
            kd = row.get('Kd', None)
            if kd is not None and not pd.isna(kd):
                kd = float(kd)
            else:
                kd = None

            # Get or compute ddG
            ddg = None
            if ddg_col and ddg_col in row and not pd.isna(row[ddg_col]):
                ddg = float(row[ddg_col])
            elif 'ddg' in row and not pd.isna(row['ddg']):
                ddg = float(row['ddg'])
            elif compute_ddg and kd is not None:
                # Compute ddG from Kd: ddG = RT * ln(Kd_mut / Kd_wt)
                # R = 0.001987 kcal/mol/K, T = 298 K
                ref_kd = reference_kd or wt_kd_cache.get(antibody_name, 100.0)
                RT = 0.001987 * 298  # kcal/mol
                ddg = RT * math.log(kd / ref_kd)

            if ddg is None:
                continue

            # Create AbEditPair
            pair = AbEditPair(
                antibody_id=f"magma_{antibody_name}_{len(pairs)}",
                antigen_id=parsed.get('antigen', None),
                heavy_wt=heavy_wt,
                light_wt=light_wt,
                mutations=mutations,
                assay_type=AssayType.DDG,
                delta_value=ddg,
                raw_wt_value=wt_kd_cache.get(antibody_name),
                raw_mut_value=kd,
                source_dataset='magma_seq',
                metadata={
                    'antibody_name': antibody_name,
                    'sheet': sheet,
                    'kd_nm': kd,
                    'fmax': row.get('Fmax', None),
                    'barcode': row.get('Barcode', None),
                },
            )
            pairs.append(pair)

    return AbEditPairsDataset(pairs)


def _parse_magma_variant(variant_str: str) -> Dict[str, Any]:
    """
    Parse MAGMA-seq variant string format.

    Format: "4A8>VH:A9A-GCC;V20V-GTG|4A8>VL:S7T-ACT"
    - Antibody name before '>'
    - Chain (VH or VL) after '>'
    - Mutations as 'OrigPosNew-Codon' separated by ';'
    - Heavy and light chains separated by '|'

    Returns dict with:
        - antibody: str
        - mutations: List[Dict] with chain, position, from_aa, to_aa, codon
    """
    import re

    result = {
        'antibody': None,
        'mutations': [],
    }

    # Handle WT variants
    if variant_str.endswith(':WT') or '>WT' in variant_str:
        parts = variant_str.split('>')
        result['antibody'] = parts[0]
        return result

    # Split by '|' for heavy/light chains
    chain_parts = variant_str.split('|')

    for chain_part in chain_parts:
        if '>' not in chain_part:
            continue

        # Parse "4A8>VH:A9A-GCC;V20V-GTG"
        ab_chain, mutations_str = chain_part.split('>', 1)
        result['antibody'] = ab_chain

        if ':' not in mutations_str:
            continue

        chain_type, mut_list = mutations_str.split(':', 1)

        if mut_list == 'WT':
            continue

        # Parse individual mutations
        for mut_str in mut_list.split(';'):
            mut_str = mut_str.strip()
            if not mut_str or mut_str == 'WT':
                continue

            # Format: A9A-GCC or A9G (without codon)
            if '-' in mut_str:
                mut_part, codon = mut_str.split('-', 1)
            else:
                mut_part = mut_str
                codon = None

            # Parse mutation: first char = from_aa, last char = to_aa, middle = position
            if len(mut_part) < 3:
                continue

            from_aa = mut_part[0].upper()
            to_aa = mut_part[-1].upper()
            pos_str = mut_part[1:-1]

            # Handle IMGT insertion codes
            match = re.match(r'^(\d+)([A-Za-z])?$', pos_str)
            if match:
                position = int(match.group(1))
            else:
                try:
                    position = int(pos_str)
                except ValueError:
                    continue

            # Skip synonymous mutations (same AA)
            if from_aa == to_aa:
                continue

            result['mutations'].append({
                'chain': chain_type,
                'position': position,
                'from_aa': from_aa,
                'to_aa': to_aa,
                'codon': codon,
            })

    return result


# =============================================================================
# FLAb Dataset Loaders
# =============================================================================
# FLAb (Fitness Landscape for Antibodies) benchmark datasets
# From: https://github.com/Graylab/FLAb
# Reference: Paper datasets used in IgBert/IgT5 (PLOS Comp Bio 2024)

# Wild-type sequences for FLAb datasets
_FLAB_WT_SEQUENCES = {
    # Koenig 2017 - G6 antibody against VEGF
    'koenig2017': {
        'heavy': 'EVQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEWVAGITPAGGYTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARFVFFLPYAMDYWGQGTLVTVSS',
        'light': 'DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYTTPPTFGQGTKVEIKR',
        'antigen': 'VEGF',
        'antibody_name': 'G6',
    },
    # Warszawski 2019 - D44.1 antibody against Hen lysozyme
    'warszawski2019': {
        'heavy': 'QVQLQESGAEIMKPGASVKISCKATGYTFSTYWIEWVKQRPGHGLEWIGEILPGSGSTYYNEKFKGKATFTADTSSNTAYMQLSSLTSEDSAVYYCARGDGNYGYWGQGTTLTV',
        'light': 'DIELTQSPATLSVTPGDSVSLSCRASQSISNNLHWYQQKSHESPRLLIKYVSQSSSGIPSRFSGSGSGTDFTLSINSVETEDFGMYFCQQSNSWPRTFGGGTKLEIKR',
        'antigen': 'Hen_lysozyme',
        'antibody_name': 'D44.1',
    },
    # Shanehsazzadeh 2023 - Trastuzumab against HER2
    'shanehsazzadeh2023': {
        'heavy': 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'light': 'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'antigen': 'HER2',
        'antibody_name': 'Trastuzumab',
    },
}


def _find_mutations_between_sequences(
    wt_seq: str,
    mut_seq: str,
    chain: str,
) -> List[AbMutation]:
    """
    Find mutations between wild-type and mutant sequences.

    Args:
        wt_seq: Wild-type sequence
        mut_seq: Mutant sequence
        chain: Chain identifier ('H' or 'L')

    Returns:
        List of AbMutation objects
    """
    mutations = []

    # Handle length differences - only compare up to shorter sequence
    min_len = min(len(wt_seq), len(mut_seq))

    for i in range(min_len):
        if wt_seq[i] != mut_seq[i]:
            mutations.append(AbMutation(
                chain=chain,
                position=i,
                from_aa=wt_seq[i],
                to_aa=mut_seq[i],
            ))

    return mutations


def load_flab_binding(
    data_dir: str,
    dataset: str = 'koenig2017',
    include_wt: bool = False,
) -> AbEditPairsDataset:
    """
    Load FLAb binding affinity dataset.

    FLAb (Fitness Landscape for Antibodies) contains binding affinity data
    from multiple DMS studies. Data from: https://github.com/Graylab/FLAb

    Args:
        data_dir: Directory containing FLAb CSV files
        dataset: Which dataset to load. Options:
            - 'koenig2017': G6 antibody vs VEGF (4,274 variants)
            - 'warszawski2019': D44.1 antibody vs Hen lysozyme (2,047 variants)
            - 'shanehsazzadeh2023': Trastuzumab vs HER2 (421 variants)
            - 'all': Load all datasets combined
        include_wt: Whether to include wild-type (no mutation) entries

    Returns:
        AbEditPairsDataset with loaded data

    Expected files in data_dir:
        - koenig2017_binding.csv
        - warszawski2019_binding.csv
        - shanehsazzadeh2023_binding.csv
    """
    data_dir = Path(data_dir)
    pairs = []

    # Determine which datasets to load
    if dataset == 'all':
        datasets_to_load = ['koenig2017', 'warszawski2019', 'shanehsazzadeh2023']
    else:
        datasets_to_load = [dataset]

    for ds_name in datasets_to_load:
        if ds_name not in _FLAB_WT_SEQUENCES:
            raise ValueError(f"Unknown FLAb dataset: {ds_name}")

        # Find the data file
        file_path = data_dir / f'{ds_name}_binding.csv'
        if not file_path.exists():
            warnings.warn(f"FLAb binding file not found: {file_path}")
            continue

        # Load data
        df = pd.read_csv(file_path)

        # Get WT sequences
        wt_info = _FLAB_WT_SEQUENCES[ds_name]
        wt_heavy = wt_info['heavy']
        wt_light = wt_info['light']

        # Identify fitness/binding column
        fitness_col = None
        for col in ['fitness', '-log( KD (M) )', '-log(KD (M))', '-log10 (KD (M) )']:
            if col in df.columns:
                fitness_col = col
                break

        if fitness_col is None:
            warnings.warn(f"No fitness column found in {ds_name}")
            continue

        for _, row in df.iterrows():
            heavy_seq = row.get('heavy', '')
            light_seq = row.get('light', '')

            if not heavy_seq or not light_seq:
                continue

            # Find mutations
            heavy_mutations = _find_mutations_between_sequences(wt_heavy, heavy_seq, 'H')
            light_mutations = _find_mutations_between_sequences(wt_light, light_seq, 'L')
            all_mutations = heavy_mutations + light_mutations

            # Skip WT if not requested
            if not include_wt and len(all_mutations) == 0:
                continue

            # Get fitness value
            fitness = row.get(fitness_col)
            if pd.isna(fitness):
                continue

            try:
                fitness = float(fitness)
            except (ValueError, TypeError):
                continue

            # Create AbEditPair
            pair = AbEditPair(
                antibody_id=f"{wt_info['antibody_name']}_{ds_name}",
                antigen_id=wt_info['antigen'],
                heavy_wt=wt_heavy,
                light_wt=wt_light,
                mutations=all_mutations,
                assay_type='binding_affinity',
                delta_value=fitness,
                source_dataset=f'flab_{ds_name}',
                metadata={
                    'dataset': ds_name,
                    'heavy_mut_seq': heavy_seq,
                    'light_mut_seq': light_seq,
                },
            )
            pairs.append(pair)

    return AbEditPairsDataset(pairs)


def load_flab_expression(
    data_dir: str,
    dataset: str = 'koenig2017',
    include_wt: bool = False,
) -> AbEditPairsDataset:
    """
    Load FLAb expression dataset.

    FLAb contains expression/enrichment ratio data from DMS studies.
    Data from: https://github.com/Graylab/FLAb

    Args:
        data_dir: Directory containing FLAb CSV files
        dataset: Which dataset to load. Currently supports:
            - 'koenig2017': G6 antibody expression (4,274 variants)
        include_wt: Whether to include wild-type (no mutation) entries

    Returns:
        AbEditPairsDataset with loaded data

    Expected files in data_dir:
        - koenig2017_expression.csv
    """
    data_dir = Path(data_dir)
    pairs = []

    if dataset not in _FLAB_WT_SEQUENCES:
        raise ValueError(f"Unknown FLAb dataset: {dataset}")

    # Find the data file
    file_path = data_dir / f'{dataset}_expression.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"FLAb expression file not found: {file_path}")

    # Load data
    df = pd.read_csv(file_path)

    # Get WT sequences
    wt_info = _FLAB_WT_SEQUENCES[dataset]
    wt_heavy = wt_info['heavy']
    wt_light = wt_info['light']

    # Identify fitness/expression column
    fitness_col = None
    for col in ['fitness', 'enrichment ratio', 'expression']:
        if col in df.columns:
            fitness_col = col
            break

    if fitness_col is None:
        raise ValueError(f"No fitness/expression column found in {dataset}")

    for _, row in df.iterrows():
        heavy_seq = row.get('heavy', '')
        light_seq = row.get('light', '')

        if not heavy_seq or not light_seq:
            continue

        # Find mutations
        heavy_mutations = _find_mutations_between_sequences(wt_heavy, heavy_seq, 'H')
        light_mutations = _find_mutations_between_sequences(wt_light, light_seq, 'L')
        all_mutations = heavy_mutations + light_mutations

        # Skip WT if not requested
        if not include_wt and len(all_mutations) == 0:
            continue

        # Get fitness value
        fitness = row.get(fitness_col)
        if pd.isna(fitness):
            continue

        try:
            fitness = float(fitness)
        except (ValueError, TypeError):
            continue

        # Create AbEditPair
        pair = AbEditPair(
            antibody_id=f"{wt_info['antibody_name']}_{dataset}",
            antigen_id=wt_info['antigen'],
            heavy_wt=wt_heavy,
            light_wt=wt_light,
            mutations=all_mutations,
            assay_type='expression',
            delta_value=fitness,
            source_dataset=f'flab_{dataset}_expression',
            metadata={
                'dataset': dataset,
                'heavy_mut_seq': heavy_seq,
                'light_mut_seq': light_seq,
            },
        )
        pairs.append(pair)

    return AbEditPairsDataset(pairs)
