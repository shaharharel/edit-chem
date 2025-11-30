"""
MPRA pair extractor for RNA Δ-expression prediction.

Extracts pairs of RNA sequences with measured expression changes
from MPRA (Massively Parallel Reporter Assay) datasets.

Parallel to src/data/small_molecule/mmp_long_format.py, this creates
long-format pairs suitable for the edit-chem framework.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

from .sequence_utils import (
    validate_rna_sequence,
    compute_hamming_distance,
    extract_edit,
    compute_kozak_score,
    find_uaugs,
    gc_content
)

logger = logging.getLogger(__name__)


class MPRAPairExtractor:
    """
    Extract paired RNA sequences with Δ-expression labels from MPRA data.

    Supports two main modes:
    1. Natural WT→variant pairs: Direct experimental Δ measurements
    2. Synthetic pairs: Computed from random libraries (Hamming neighbors)

    Output schema (long format):
        seq_a: Reference sequence
        seq_b: Variant sequence
        edit_type: SNV, insertion, deletion, complex
        edit_position: 0-indexed position of edit
        edit_from: Original nucleotide(s)
        edit_to: New nucleotide(s)
        value_a: Expression value for seq_a
        value_b: Expression value for seq_b
        delta: value_b - value_a
        property_name: e.g., "MRL_5UTR", "stability_3UTR"
        cell_type: Cell line used
        experiment_id: Batch/experiment identifier

    Example:
        >>> extractor = MPRAPairExtractor()
        >>> pairs = extractor.extract_snv_pairs(snv_df)
        >>> pairs = extractor.extract_hamming_neighbors(random_df, max_distance=2)
    """

    def __init__(self, property_name: str = "MRL_5UTR"):
        """
        Initialize extractor.

        Args:
            property_name: Name of the expression property being measured
        """
        self.property_name = property_name

    def extract_snv_pairs(
        self,
        snv_df: pd.DataFrame,
        ref_col: str = 'ref_sequence',
        var_col: str = 'var_sequence',
        ref_value_col: str = 'ref_MRL',
        var_value_col: str = 'var_MRL',
        cell_type: str = 'HEK293T',
        experiment_id: str = 'MPRA'
    ) -> pd.DataFrame:
        """
        Extract pairs from SNV variant data.

        This handles data where each row contains a reference sequence,
        a variant sequence, and measured expression values for both.

        Args:
            snv_df: DataFrame with SNV data
            ref_col: Column name for reference sequence
            var_col: Column name for variant sequence
            ref_value_col: Column name for reference expression value
            var_value_col: Column name for variant expression value
            cell_type: Cell type for annotation
            experiment_id: Experiment ID for annotation

        Returns:
            Long-format DataFrame with pairs
        """
        logger.info(f"Extracting SNV pairs from {len(snv_df)} variants...")

        pairs = []

        for idx, row in tqdm(snv_df.iterrows(), total=len(snv_df), desc="Extracting pairs"):
            ref_seq = str(row.get(ref_col, '')).upper().replace('T', 'U')
            var_seq = str(row.get(var_col, '')).upper().replace('T', 'U')

            ref_value = row.get(ref_value_col)
            var_value = row.get(var_value_col)

            # Skip if missing data
            if pd.isna(ref_value) or pd.isna(var_value):
                continue
            if not ref_seq or not var_seq:
                continue

            # Extract edit information
            edit_info = extract_edit(ref_seq, var_seq)

            # Create pair record
            pair = {
                'seq_a': ref_seq,
                'seq_b': var_seq,
                'edit_type': edit_info['edit_type'],
                'edit_position': edit_info['position'],
                'edit_from': edit_info['edit_from'],
                'edit_to': edit_info['edit_to'],
                'value_a': float(ref_value),
                'value_b': float(var_value),
                'delta': float(var_value) - float(ref_value),
                'property_name': self.property_name,
                'cell_type': cell_type,
                'experiment_id': experiment_id
            }

            # Add optional metadata
            if 'log2_delta_MRL' in row:
                pair['log2_delta'] = row['log2_delta_MRL']

            pairs.append(pair)

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"  Extracted {len(pairs_df)} pairs")

        return pairs_df

    def extract_hamming_neighbors(
        self,
        sequences_df: pd.DataFrame,
        seq_col: str = 'sequence',
        value_col: str = 'MRL',
        max_distance: int = 2,
        min_value_diff: float = 0.0,
        max_pairs_per_sequence: int = 10,
        cell_type: str = 'HEK293T',
        experiment_id: str = 'MPRA_synthetic'
    ) -> pd.DataFrame:
        """
        Extract synthetic pairs from random library by finding Hamming neighbors.

        For each sequence, finds other sequences that differ by at most
        max_distance nucleotides. This creates synthetic Δ-pairs from
        the random library where no explicit WT→variant pairing exists.

        Args:
            sequences_df: DataFrame with sequences and expression values
            seq_col: Column name for sequence
            value_col: Column name for expression value
            max_distance: Maximum Hamming distance for pairing
            min_value_diff: Minimum |Δ| to include pair
            max_pairs_per_sequence: Maximum neighbors per sequence
            cell_type: Cell type annotation
            experiment_id: Experiment ID annotation

        Returns:
            Long-format DataFrame with synthetic pairs
        """
        logger.info(f"Extracting Hamming neighbor pairs (max_distance={max_distance})...")
        logger.info(f"  Input sequences: {len(sequences_df)}")

        # Filter valid sequences
        valid_df = sequences_df.dropna(subset=[seq_col, value_col]).copy()
        valid_df['_clean_seq'] = valid_df[seq_col].str.upper().str.replace('T', 'U')

        # Group by sequence length for efficiency
        seq_by_length = defaultdict(list)
        for idx, row in valid_df.iterrows():
            seq = row['_clean_seq']
            seq_by_length[len(seq)].append({
                'idx': idx,
                'seq': seq,
                'value': row[value_col]
            })

        pairs = []
        total_checked = 0

        for length, seq_list in tqdm(seq_by_length.items(), desc="Processing lengths"):
            if len(seq_list) < 2:
                continue

            # Build index for fast lookup
            seq_to_data = {item['seq']: item for item in seq_list}

            for i, item_a in enumerate(seq_list):
                neighbors_found = 0

                for j in range(i + 1, len(seq_list)):
                    if neighbors_found >= max_pairs_per_sequence:
                        break

                    item_b = seq_list[j]
                    total_checked += 1

                    # Compute Hamming distance
                    dist = compute_hamming_distance(item_a['seq'], item_b['seq'])

                    if dist > 0 and dist <= max_distance:
                        delta = item_b['value'] - item_a['value']

                        if abs(delta) >= min_value_diff:
                            # Extract edit info
                            edit_info = extract_edit(item_a['seq'], item_b['seq'])

                            pair = {
                                'seq_a': item_a['seq'],
                                'seq_b': item_b['seq'],
                                'edit_type': edit_info['edit_type'],
                                'edit_position': edit_info['position'],
                                'edit_from': edit_info['edit_from'],
                                'edit_to': edit_info['edit_to'],
                                'value_a': item_a['value'],
                                'value_b': item_b['value'],
                                'delta': delta,
                                'hamming_distance': dist,
                                'property_name': self.property_name,
                                'cell_type': cell_type,
                                'experiment_id': experiment_id
                            }
                            pairs.append(pair)
                            neighbors_found += 1

        pairs_df = pd.DataFrame(pairs)

        logger.info(f"  Checked {total_checked:,} sequence pairs")
        logger.info(f"  Found {len(pairs_df):,} Hamming neighbor pairs")

        if len(pairs_df) > 0:
            logger.info(f"  Distance distribution:")
            for dist in range(1, max_distance + 1):
                count = (pairs_df['hamming_distance'] == dist).sum()
                logger.info(f"    Distance {dist}: {count:,} pairs")

        return pairs_df

    def extract_motif_based_pairs(
        self,
        sequences_df: pd.DataFrame,
        seq_col: str = 'sequence',
        value_col: str = 'MRL',
        motif_type: str = 'kozak',
        cell_type: str = 'HEK293T'
    ) -> pd.DataFrame:
        """
        Extract pairs based on motif presence/absence.

        Groups sequences by presence/absence of regulatory motifs
        (Kozak, uAUG, etc.) and creates pairs within groups.

        Args:
            sequences_df: DataFrame with sequences and values
            seq_col: Column name for sequence
            value_col: Column name for expression value
            motif_type: 'kozak', 'uaug', or 'gc'
            cell_type: Cell type annotation

        Returns:
            Long-format DataFrame with motif-based pairs
        """
        logger.info(f"Extracting motif-based pairs (motif={motif_type})...")

        valid_df = sequences_df.dropna(subset=[seq_col, value_col]).copy()
        valid_df['_clean_seq'] = valid_df[seq_col].str.upper().str.replace('T', 'U')

        # Compute motif scores
        if motif_type == 'kozak':
            def get_motif_score(seq):
                uaugs = find_uaugs(seq)
                if uaugs:
                    return max(compute_kozak_score(seq, pos) for pos in uaugs)
                return 0

        elif motif_type == 'uaug':
            def get_motif_score(seq):
                return len(find_uaugs(seq))

        elif motif_type == 'gc':
            def get_motif_score(seq):
                return gc_content(seq)

        else:
            raise ValueError(f"Unknown motif type: {motif_type}")

        valid_df['_motif_score'] = valid_df['_clean_seq'].apply(get_motif_score)

        # Bin by motif score
        valid_df['_motif_bin'] = pd.qcut(
            valid_df['_motif_score'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )

        # Create pairs between adjacent bins
        pairs = []
        bins = ['very_low', 'low', 'medium', 'high', 'very_high']

        for i in range(len(bins) - 1):
            low_bin = valid_df[valid_df['_motif_bin'] == bins[i]]
            high_bin = valid_df[valid_df['_motif_bin'] == bins[i + 1]]

            # Sample pairs
            n_pairs = min(len(low_bin), len(high_bin), 1000)

            if n_pairs < 10:
                continue

            low_sample = low_bin.sample(n=n_pairs, random_state=42)
            high_sample = high_bin.sample(n=n_pairs, random_state=42)

            for (_, low_row), (_, high_row) in zip(
                low_sample.iterrows(), high_sample.iterrows()
            ):
                pairs.append({
                    'seq_a': low_row['_clean_seq'],
                    'seq_b': high_row['_clean_seq'],
                    'edit_type': 'motif_change',
                    'edit_position': -1,
                    'edit_from': f"{motif_type}_{bins[i]}",
                    'edit_to': f"{motif_type}_{bins[i + 1]}",
                    'value_a': low_row[value_col],
                    'value_b': high_row[value_col],
                    'delta': high_row[value_col] - low_row[value_col],
                    'property_name': self.property_name,
                    'cell_type': cell_type,
                    'experiment_id': f'MPRA_motif_{motif_type}'
                })

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"  Created {len(pairs_df)} motif-based pairs")

        return pairs_df

    def combine_pairs(
        self,
        pair_dfs: List[pd.DataFrame],
        deduplicate: bool = True
    ) -> pd.DataFrame:
        """
        Combine multiple pair DataFrames.

        Args:
            pair_dfs: List of pair DataFrames
            deduplicate: Remove duplicate (seq_a, seq_b) pairs

        Returns:
            Combined DataFrame
        """
        combined = pd.concat(pair_dfs, ignore_index=True)

        if deduplicate:
            # Sort to ensure consistent duplicate handling
            combined = combined.sort_values(['seq_a', 'seq_b', 'experiment_id'])
            combined = combined.drop_duplicates(subset=['seq_a', 'seq_b'], keep='first')

        logger.info(f"Combined {len(combined)} pairs from {len(pair_dfs)} sources")

        return combined

    def add_edit_features(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add computed edit features to pairs DataFrame.

        Features:
        - kozak_score_a, kozak_score_b, kozak_change
        - uaug_count_a, uaug_count_b, uaug_change
        - gc_content_a, gc_content_b, gc_change
        - length_a, length_b

        Args:
            pairs_df: Pairs DataFrame

        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Adding edit features to pairs...")

        df = pairs_df.copy()

        # Length
        df['length_a'] = df['seq_a'].str.len()
        df['length_b'] = df['seq_b'].str.len()

        # GC content
        df['gc_content_a'] = df['seq_a'].apply(gc_content)
        df['gc_content_b'] = df['seq_b'].apply(gc_content)
        df['gc_change'] = df['gc_content_b'] - df['gc_content_a']

        # uAUG count
        df['uaug_count_a'] = df['seq_a'].apply(lambda s: len(find_uaugs(s)))
        df['uaug_count_b'] = df['seq_b'].apply(lambda s: len(find_uaugs(s)))
        df['uaug_change'] = df['uaug_count_b'] - df['uaug_count_a']

        # Kozak score (max across all AUGs)
        def max_kozak(seq):
            uaugs = find_uaugs(seq)
            if uaugs:
                return max(compute_kozak_score(seq, pos) for pos in uaugs)
            return 0

        df['kozak_score_a'] = df['seq_a'].apply(max_kozak)
        df['kozak_score_b'] = df['seq_b'].apply(max_kozak)
        df['kozak_change'] = df['kozak_score_b'] - df['kozak_score_a']

        logger.info(f"  Added features to {len(df)} pairs")

        return df

    def save_pairs(
        self,
        pairs_df: pd.DataFrame,
        output_path: Path,
        format: str = 'csv'
    ):
        """
        Save pairs DataFrame to file.

        Args:
            pairs_df: Pairs DataFrame
            output_path: Output file path
            format: 'csv' or 'parquet'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            pairs_df.to_csv(output_path, index=False)
        elif format == 'parquet':
            pairs_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(pairs_df)} pairs to {output_path}")
