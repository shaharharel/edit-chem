#!/usr/bin/env python
"""
Generate generalized edit pairs dataset from m6A binding data.

This script generates two types of paired datasets:

1. M/A pairs (original): Pairs of identical sequences differing only in
   methylation status at the center position (A vs m6A).

2. Generalized edit pairs (new): All pairs of sequences that differ by
   at most `max_distance` nucleotide changes in the loop region.
   This creates a much richer dataset for training edit-effect models.

The generalized pairs enable:
- Learning general SNV effects, not just methylation
- More training data (O(n²) pairs vs O(n) M/A pairs)
- Understanding positional effects of each nucleotide change

Output format matches the existing m6a processed data structure for
compatibility with experiments/m6a/run_comprehensive_benchmark.py

Usage:
    # Generate M/A pairs only (default behavior)
    python generate_m6a_edit_pairs.py --mode ma_only

    # Generate all edit pairs with Hamming distance <= 1
    python generate_m6a_edit_pairs.py --mode all_pairs --max_distance 1

    # Generate both datasets
    python generate_m6a_edit_pairs.py --mode both --max_distance 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        raise ValueError(f"Strings must have equal length: {len(s1)} vs {len(s2)}")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_edit_info(seq_a: str, seq_b: str) -> Dict:
    """
    Extract edit information between two sequences.

    Returns dict with:
    - edit_positions: list of positions that differ
    - edit_from: nucleotides in seq_a at edit positions
    - edit_to: nucleotides in seq_b at edit positions
    - edit_type: 'SNV' for single change, 'multi_SNV' for multiple
    - n_edits: number of positions changed
    """
    if len(seq_a) != len(seq_b):
        raise ValueError("Sequences must have equal length")

    edit_positions = []
    edit_from = []
    edit_to = []

    for i, (c1, c2) in enumerate(zip(seq_a, seq_b)):
        if c1 != c2:
            edit_positions.append(i)
            edit_from.append(c1)
            edit_to.append(c2)

    n_edits = len(edit_positions)

    return {
        'edit_positions': edit_positions,
        'edit_from': ''.join(edit_from),
        'edit_to': ''.join(edit_to),
        'edit_type': 'SNV' if n_edits == 1 else f'multi_SNV_{n_edits}' if n_edits > 1 else 'identical',
        'n_edits': n_edits,
        'edit_positions_str': ','.join(map(str, edit_positions)),
        'edit_description': '|'.join(f"{p}:{f}>{t}" for p, f, t in zip(edit_positions, edit_from, edit_to))
    }


def load_and_aggregate_measurements(
    design_path: Path,
    measurement_path: Path,
    min_replicates: int = 5
) -> pd.DataFrame:
    """
    Load design file and aggregate measurements from all replicates.

    Returns DataFrame with one row per sequence containing:
    - All design columns (name, sequence_5to3, loop, center, type, etc.)
    - Aggregated intensity measurements (median, mean, std, n_replicates)
    """
    logger.info(f"Loading design file: {design_path}")
    design = pd.read_csv(design_path)
    logger.info(f"Design shape: {design.shape}")

    logger.info(f"Loading measurements: {measurement_path}")
    measurements = pd.read_csv(measurement_path, sep='\t', skiprows=1)
    measurements.columns = measurements.columns.str.strip()

    # Aggregate replicates per sequence
    logger.info("Aggregating technical replicates...")
    agg_df = measurements.groupby('SEQ_ID').agg(
        intensity_median=('PM_635_26112025_nm', 'median'),
        intensity_mean=('PM_635_26112025_nm', 'mean'),
        intensity_std=('PM_635_26112025_nm', 'std'),
        n_replicates=('PM_635_26112025_nm', 'count')
    ).reset_index()
    agg_df.rename(columns={'SEQ_ID': 'name'}, inplace=True)

    # Merge with design
    merged = design.merge(agg_df, on='name', how='inner')
    logger.info(f"Merged shape: {merged.shape}")

    # Filter by minimum replicates
    merged = merged[merged['n_replicates'] >= min_replicates]
    logger.info(f"After filtering (min_replicates={min_replicates}): {merged.shape}")

    return merged


def generate_ma_pairs(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate M/A pairs (methylated vs unmethylated at center).

    This is the original pairing strategy: match sequences that are
    identical except for the center position being A vs m6A (M).
    """
    logger.info("Generating M/A pairs...")

    # Extract pair ID from sequence name (e.g., HP15-7-15_A_000001 -> 000001)
    df = df.copy()
    df['pair_id'] = df['name'].apply(lambda x: x.split('_')[-1])

    # Separate A and M sequences
    a_seqs = df[df['center'] == 'A'].copy()
    m_seqs = df[df['center'] == 'M'].copy()

    logger.info(f"A sequences: {len(a_seqs)}, M sequences: {len(m_seqs)}")

    # Rename columns for A
    a_cols = {col: f"{col}_A" if col not in ['pair_id', 'type', 'type_custom', 'stem_len_each', 'total_len']
              else col for col in a_seqs.columns}
    a_seqs = a_seqs.rename(columns=a_cols)

    # Rename columns for M
    m_rename = {
        'name': 'name_M',
        'sequence_5to3': 'sequence_M',
        'loop': 'loop_M',
        'center': 'center_M',
        'intensity_median': 'intensity_median_M',
        'intensity_mean': 'intensity_mean_M',
        'intensity_std': 'intensity_std_M',
        'n_replicates': 'n_replicates_M'
    }
    m_seqs = m_seqs.rename(columns=m_rename)

    # Merge A and M by pair_id
    paired = a_seqs.merge(
        m_seqs[['pair_id', 'name_M', 'sequence_M', 'loop_M', 'center_M',
                'intensity_median_M', 'intensity_mean_M', 'intensity_std_M', 'n_replicates_M']],
        on='pair_id',
        how='inner'
    )

    logger.info(f"Paired M/A sequences: {len(paired)}")

    # Rename for consistency with expected output
    paired = paired.rename(columns={
        'name_A': 'name_A',
        'sequence_5to3_A': 'sequence_A',
        'loop_A': 'loop_A',
        'intensity_median_A': 'intensity_median_A',
        'intensity_mean_A': 'intensity_mean_A',
        'intensity_std_A': 'intensity_std_A',
        'n_replicates_A': 'n_replicates_A'
    })

    # Calculate differential metrics
    paired['delta_intensity_median'] = paired['intensity_median_M'] - paired['intensity_median_A']
    paired['delta_intensity_mean'] = paired['intensity_mean_M'] - paired['intensity_mean_A']
    paired['fold_change_median'] = paired['intensity_median_M'] / paired['intensity_median_A']
    paired['fold_change_mean'] = paired['intensity_mean_M'] / paired['intensity_mean_A']
    paired['log2_fold_change_median'] = np.log2(paired['fold_change_median'])
    paired['log2_fold_change_mean'] = np.log2(paired['fold_change_mean'])

    # Add context features
    paired['context_3nt_upstream'] = paired['loop_A'].str[:3]
    paired['context_3nt_downstream'] = paired['loop_A'].str[4:]

    # Check DRACH motif
    paired['is_drach'] = paired['loop_A'].apply(_is_drach)

    # Add edit info
    paired['edit_type'] = 'methylation'
    paired['edit_positions_str'] = '3'  # Center position in 7-mer
    paired['edit_description'] = '3:A>M'
    paired['n_edits'] = 1
    paired['pair_type'] = 'M_vs_A'

    return paired


def generate_all_edit_pairs(
    df: pd.DataFrame,
    max_distance: int = 1,
    include_ma_pairs: bool = True,
    sample_fraction: float = 1.0
) -> pd.DataFrame:
    """
    Generate all pairs of sequences within max_distance Hamming distance.

    This creates a much larger dataset by pairing all sequences whose loops
    differ by at most max_distance nucleotides.

    Args:
        df: DataFrame with sequences and measurements
        max_distance: Maximum Hamming distance between loop sequences
        include_ma_pairs: Whether to include M/A pairs (methylation pairs)
        sample_fraction: Fraction of pairs to sample (for very large datasets)

    Returns:
        DataFrame with all valid pairs and their differential metrics
    """
    logger.info(f"Generating all edit pairs with max_distance={max_distance}")

    # For efficiency, separate by center type and work with A sequences primarily
    a_seqs = df[df['center'] == 'A'].copy().reset_index(drop=True)
    m_seqs = df[df['center'] == 'M'].copy().reset_index(drop=True)

    logger.info(f"A sequences: {len(a_seqs)}, M sequences: {len(m_seqs)}")

    # Build index of loops to sequences for fast lookup
    loop_to_a_idx = defaultdict(list)
    for idx, row in a_seqs.iterrows():
        loop_to_a_idx[row['loop']].append(idx)

    loop_to_m_idx = defaultdict(list)
    for idx, row in m_seqs.iterrows():
        loop_to_m_idx[row['loop']].append(idx)

    all_pairs = []
    pair_id = 0

    # 1. Generate A vs A pairs (same methylation status, different sequence)
    logger.info("Generating A vs A pairs...")
    a_loops = list(a_seqs['loop'].unique())

    for i, loop1 in enumerate(a_loops):
        for loop2 in a_loops[i+1:]:  # Avoid duplicates
            dist = hamming_distance(loop1, loop2)
            if dist <= max_distance and dist > 0:
                # Get all sequences with these loops
                for idx1 in loop_to_a_idx[loop1]:
                    for idx2 in loop_to_a_idx[loop2]:
                        row1 = a_seqs.iloc[idx1]
                        row2 = a_seqs.iloc[idx2]

                        pair = _create_pair_record(
                            row1, row2,
                            pair_id=f"AA_{pair_id:06d}",
                            pair_type='A_vs_A'
                        )
                        all_pairs.append(pair)
                        pair_id += 1

    logger.info(f"Generated {pair_id} A vs A pairs")

    # 2. Generate M vs M pairs (same methylation status, different sequence)
    logger.info("Generating M vs M pairs...")
    m_loops = list(m_seqs['loop'].unique())
    m_pair_start = pair_id

    for i, loop1 in enumerate(m_loops):
        for loop2 in m_loops[i+1:]:
            # For M sequences, loop contains 'M' at center, compare flanking
            loop1_base = loop1.replace('M', 'A')  # Normalize for comparison
            loop2_base = loop2.replace('M', 'A')
            dist = hamming_distance(loop1_base, loop2_base)

            if dist <= max_distance and dist > 0:
                for idx1 in loop_to_m_idx[loop1]:
                    for idx2 in loop_to_m_idx[loop2]:
                        row1 = m_seqs.iloc[idx1]
                        row2 = m_seqs.iloc[idx2]

                        pair = _create_pair_record(
                            row1, row2,
                            pair_id=f"MM_{pair_id:06d}",
                            pair_type='M_vs_M'
                        )
                        all_pairs.append(pair)
                        pair_id += 1

    logger.info(f"Generated {pair_id - m_pair_start} M vs M pairs")

    # 3. Generate M vs A pairs (methylation change)
    if include_ma_pairs:
        logger.info("Generating M vs A pairs (methylation)...")
        ma_pair_start = pair_id

        # Match by identical flanking sequence
        for a_idx, a_row in a_seqs.iterrows():
            a_loop = a_row['loop']
            # Find corresponding M sequence (same flanking, center M instead of A)
            m_loop = a_loop[:3] + 'M' + a_loop[4:]

            if m_loop in loop_to_m_idx:
                for m_idx in loop_to_m_idx[m_loop]:
                    m_row = m_seqs.iloc[m_idx]

                    pair = _create_pair_record(
                        a_row, m_row,
                        pair_id=f"MA_{pair_id:06d}",
                        pair_type='M_vs_A'
                    )
                    all_pairs.append(pair)
                    pair_id += 1

        logger.info(f"Generated {pair_id - ma_pair_start} M vs A pairs")

    logger.info(f"Total pairs generated: {len(all_pairs)}")

    # Convert to DataFrame
    pairs_df = pd.DataFrame(all_pairs)

    # Sample if requested
    if sample_fraction < 1.0:
        n_sample = int(len(pairs_df) * sample_fraction)
        pairs_df = pairs_df.sample(n=n_sample, random_state=42)
        logger.info(f"Sampled {n_sample} pairs ({sample_fraction*100:.1f}%)")

    return pairs_df


def _create_pair_record(row_a: pd.Series, row_b: pd.Series, pair_id: str, pair_type: str) -> Dict:
    """Create a pair record from two sequence rows."""

    # Get edit info (compare loops, normalizing M to A for comparison)
    loop_a = row_a['loop'].replace('M', 'A')
    loop_b = row_b['loop'].replace('M', 'A')

    if pair_type == 'M_vs_A':
        # For M vs A, the edit is at center position
        edit_info = {
            'edit_positions': [3],
            'edit_from': 'A',
            'edit_to': 'M',
            'edit_type': 'methylation',
            'n_edits': 1,
            'edit_positions_str': '3',
            'edit_description': '3:A>M'
        }
    else:
        edit_info = get_edit_info(loop_a, loop_b)

    # Calculate differential metrics
    # Convention: B - A (row_b is the "edited" version)
    delta_median = row_b['intensity_median'] - row_a['intensity_median']
    delta_mean = row_b['intensity_mean'] - row_a['intensity_mean']

    # Handle division by zero
    if row_a['intensity_median'] > 0:
        fold_change_median = row_b['intensity_median'] / row_a['intensity_median']
        log2_fc_median = np.log2(fold_change_median)
    else:
        fold_change_median = np.nan
        log2_fc_median = np.nan

    if row_a['intensity_mean'] > 0:
        fold_change_mean = row_b['intensity_mean'] / row_a['intensity_mean']
        log2_fc_mean = np.log2(fold_change_mean)
    else:
        fold_change_mean = np.nan
        log2_fc_mean = np.nan

    return {
        'pair_id': pair_id,
        'name_A': row_a['name'],
        'name_B': row_b['name'],
        'sequence_A': row_a['sequence_5to3'],
        'sequence_B': row_b['sequence_5to3'],
        'loop_A': row_a['loop'],
        'loop_B': row_b['loop'],
        'center_A': row_a['center'],
        'center_B': row_b['center'],
        'type': row_a.get('type', 'unknown'),
        'type_custom': row_a.get('type_custom', 'unknown'),
        'stem_len_each': row_a.get('stem_len_each', 15),
        'total_len': row_a.get('total_len', 37),

        # Intensity A
        'intensity_median_A': row_a['intensity_median'],
        'intensity_mean_A': row_a['intensity_mean'],
        'intensity_std_A': row_a['intensity_std'],
        'n_replicates_A': row_a['n_replicates'],

        # Intensity B
        'intensity_median_B': row_b['intensity_median'],
        'intensity_mean_B': row_b['intensity_mean'],
        'intensity_std_B': row_b['intensity_std'],
        'n_replicates_B': row_b['n_replicates'],

        # Differential metrics
        'delta_intensity_median': delta_median,
        'delta_intensity_mean': delta_mean,
        'fold_change_median': fold_change_median,
        'fold_change_mean': fold_change_mean,
        'log2_fold_change_median': log2_fc_median,
        'log2_fold_change_mean': log2_fc_mean,

        # Context features
        'context_3nt_upstream': loop_a[:3],
        'context_3nt_downstream': loop_a[4:],
        'is_drach': _is_drach(loop_a),

        # Edit info
        'edit_type': edit_info['edit_type'],
        'edit_positions_str': edit_info['edit_positions_str'],
        'edit_description': edit_info['edit_description'],
        'n_edits': edit_info['n_edits'],
        'pair_type': pair_type
    }


def _is_drach(loop: str) -> bool:
    """Check if a 7-mer loop matches the DRACH motif."""
    if len(loop) != 7:
        return False

    # Normalize M to A for checking
    loop = loop.replace('M', 'A')

    d_pos = loop[1]  # -2 position
    r_pos = loop[2]  # -1 position
    center = loop[3]  # center (A)
    c_pos = loop[4]  # +1 position
    h_pos = loop[5]  # +2 position

    d_valid = d_pos in ['A', 'G', 'U']
    r_valid = r_pos in ['A', 'G']
    a_valid = center == 'A'
    c_valid = c_pos == 'C'
    h_valid = h_pos in ['A', 'C', 'U']

    return d_valid and r_valid and a_valid and c_valid and h_valid


def convert_to_ma_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert generalized pairs format to M/A pairs format for compatibility.

    Renames columns B -> M to match the original M/A pairs output format.
    """
    rename_map = {
        'name_B': 'name_M',
        'sequence_B': 'sequence_M',
        'loop_B': 'loop_M',
        'center_B': 'center_M',
        'intensity_median_B': 'intensity_median_M',
        'intensity_mean_B': 'intensity_mean_M',
        'intensity_std_B': 'intensity_std_M',
        'n_replicates_B': 'n_replicates_M'
    }

    return df.rename(columns=rename_map)


def main():
    parser = argparse.ArgumentParser(
        description='Generate m6A edit pairs dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--data_dir', type=str,
                        default='data/rna/m6a',
                        help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str,
                        default='data/rna/m6a/processed',
                        help='Directory for processed output files')
    parser.add_argument('--measurement_file', type=str,
                        default='635_500_50_2.5res_YTHDF2100nM_26112025.tsv',
                        help='Measurement file to use')
    parser.add_argument('--mode', type=str,
                        default='both',
                        choices=['ma_only', 'all_pairs', 'both'],
                        help='Generation mode: ma_only, all_pairs, or both')
    parser.add_argument('--max_distance', type=int,
                        default=1,
                        help='Maximum Hamming distance for edit pairs')
    parser.add_argument('--min_replicates', type=int,
                        default=5,
                        help='Minimum replicates per sequence')
    parser.add_argument('--sample_fraction', type=float,
                        default=1.0,
                        help='Fraction of pairs to sample (for large datasets)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine sensitivity suffix
    if '500' in args.measurement_file:
        suffix = 'sensitivity_500'
    elif '600' in args.measurement_file:
        suffix = 'sensitivity_600'
    else:
        suffix = 'default'

    # Load data
    design_path = data_dir / 'hairpin_15_7_15_all7mers_centerA_and_M_plus_label.csv'
    measurement_path = data_dir / args.measurement_file

    df = load_and_aggregate_measurements(
        design_path, measurement_path,
        min_replicates=args.min_replicates
    )

    # Generate datasets based on mode
    if args.mode in ['ma_only', 'both']:
        logger.info("\n" + "="*60)
        logger.info("GENERATING M/A PAIRS (METHYLATION)")
        logger.info("="*60)

        ma_pairs = generate_ma_pairs(df)

        # Save M/A pairs
        ma_output = output_dir / f'm6a_paired_binding_{suffix}_simple.csv'
        # Select columns in expected order
        ma_cols = [
            'pair_id', 'name_A', 'name_M', 'sequence_A', 'sequence_M',
            'loop_A', 'loop_M', 'type', 'type_custom', 'stem_len_each', 'total_len',
            'intensity_median_A', 'intensity_mean_A', 'intensity_std_A', 'n_replicates_A',
            'intensity_median_M', 'intensity_mean_M', 'intensity_std_M', 'n_replicates_M',
            'delta_intensity_median', 'delta_intensity_mean',
            'fold_change_median', 'fold_change_mean',
            'log2_fold_change_median', 'log2_fold_change_mean',
            'context_3nt_upstream', 'context_3nt_downstream', 'is_drach'
        ]
        ma_cols_available = [c for c in ma_cols if c in ma_pairs.columns]
        ma_pairs[ma_cols_available].to_csv(ma_output, index=False)
        logger.info(f"Saved M/A pairs to: {ma_output}")
        logger.info(f"Total M/A pairs: {len(ma_pairs)}")

        # Print summary
        logger.info("\n=== M/A Pairs Summary ===")
        logger.info(f"Total pairs: {len(ma_pairs)}")
        logger.info(f"DRACH pairs: {ma_pairs['is_drach'].sum()}")
        logger.info(f"Log2 FC: {ma_pairs['log2_fold_change_median'].mean():.3f} ± {ma_pairs['log2_fold_change_median'].std():.3f}")

    if args.mode in ['all_pairs', 'both']:
        logger.info("\n" + "="*60)
        logger.info(f"GENERATING ALL EDIT PAIRS (max_distance={args.max_distance})")
        logger.info("="*60)

        all_pairs = generate_all_edit_pairs(
            df,
            max_distance=args.max_distance,
            include_ma_pairs=True,
            sample_fraction=args.sample_fraction
        )

        # Save all pairs (keeping B naming for generalized format)
        all_output = output_dir / f'm6a_edit_pairs_dist{args.max_distance}_{suffix}.csv'
        all_pairs.to_csv(all_output, index=False)
        logger.info(f"Saved all edit pairs to: {all_output}")

        # Also save in M/A compatible format (B -> M)
        all_pairs_compat = convert_to_ma_format(all_pairs)
        all_output_compat = output_dir / f'm6a_edit_pairs_dist{args.max_distance}_{suffix}_compat.csv'
        all_pairs_compat.to_csv(all_output_compat, index=False)
        logger.info(f"Saved compatible format to: {all_output_compat}")

        # Print summary
        logger.info("\n=== All Edit Pairs Summary ===")
        logger.info(f"Total pairs: {len(all_pairs)}")
        for pair_type in all_pairs['pair_type'].unique():
            subset = all_pairs[all_pairs['pair_type'] == pair_type]
            logger.info(f"  {pair_type}: {len(subset)} pairs, log2FC = {subset['log2_fold_change_median'].mean():.3f} ± {subset['log2_fold_change_median'].std():.3f}")

        # Edit type distribution
        logger.info("\nEdit type distribution:")
        logger.info(all_pairs['edit_type'].value_counts().to_string())

        # Position distribution for SNV edits
        snv_pairs = all_pairs[all_pairs['edit_type'] == 'SNV']
        if len(snv_pairs) > 0:
            logger.info("\nSNV position distribution:")
            logger.info(snv_pairs['edit_positions_str'].value_counts().to_string())

    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
