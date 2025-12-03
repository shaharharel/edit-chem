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
) -> AbEditPairsDataset:
    """
    Load AbAgym dataset.

    AbAgym is an antibody affinity maturation gym containing:
    - Deep mutational scanning data
    - Directed evolution trajectories
    - Multi-round affinity maturation campaigns

    Args:
        data_dir: Directory containing AbAgym data files
        target: Specific target/antigen to load (None for all)
        split: Data split ('train', 'val', 'test', None for all)

    Returns:
        AbEditPairsDataset with loaded data

    Expected files:
        - abagym_data.csv or target-specific files
        - abagym_sequences.json: Antibody sequences
    """
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
            if 'sequence' not in f.name.lower():
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

        for _, row in df.iterrows():
            # Get antibody ID
            antibody_id = str(row.get('antibody_id', row.get('name', row.get('id', ''))))

            # Get sequences
            if antibody_id in sequences:
                heavy_wt = sequences[antibody_id].get('heavy', '')
                light_wt = sequences[antibody_id].get('light', '')
            else:
                heavy_wt = row.get('heavy_wt', row.get('VH_wt', row.get('parent_heavy', '')))
                light_wt = row.get('light_wt', row.get('VL_wt', row.get('parent_light', '')))

            if not heavy_wt:
                continue

            # For some datasets, light chain may be optional
            if not light_wt:
                light_wt = ''

            # Parse mutations
            mutation_str = row.get('mutations', row.get('mutation', row.get('variant', '')))
            mutations = _parse_mutations(mutation_str, heavy_wt, light_wt)

            if not mutations:
                continue

            # Get delta value (enrichment score or binding)
            delta_value = row.get('enrichment', row.get('fitness', row.get('score', None)))
            if delta_value is None:
                delta_value = row.get('log_enrichment', row.get('ddg', None))

            if delta_value is None:
                continue

            try:
                delta_value = float(delta_value)
            except (ValueError, TypeError):
                continue

            # Determine assay type
            if 'enrichment' in df.columns or 'fitness' in df.columns:
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
    """Parse a single mutation string."""
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

    try:
        position = int(mutation_str[1:-1]) - 1  # Convert to 0-indexed
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
