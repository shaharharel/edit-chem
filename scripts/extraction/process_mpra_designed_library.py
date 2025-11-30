#!/usr/bin/env python3
"""
Process the MPRA designed library to extract SNV pairs with Δ-MRL labels.

The designed library (GSM3130443) contains:
- Human 5' UTRs
- SNV variants with WT reference
- Designed variants

This script extracts the SNV pairs for edit effect prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def process_designed_library(input_path: Path, output_dir: Path):
    """
    Process the designed library to extract SNV pairs.

    Args:
        input_path: Path to GSM3130443_designed_library.csv.gz
        output_dir: Output directory for processed data
    """
    print(f"Processing: {input_path}")

    # Read the data
    df = pd.read_csv(input_path, compression='gzip', low_memory=False)
    print(f"  Total sequences: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")

    # Show library distribution
    print(f"\n  Library types:")
    for lib_type, count in df['library'].value_counts().items():
        print(f"    {lib_type}: {count:,}")

    # Filter to SNV library
    snv_df = df[df['library'] == 'snv'].copy()
    print(f"\n  SNV library total: {len(snv_df):,}")

    # Build a lookup: sequence -> rl value (from ALL sequences in dataset)
    # This allows us to find mother sequence rl values
    all_seq_to_rl = {}
    for _, row in df.iterrows():
        if pd.notna(row['utr']) and pd.notna(row['rl']):
            seq = str(row['utr'])
            if seq not in all_seq_to_rl:
                all_seq_to_rl[seq] = float(row['rl'])
    print(f"  Built sequence->rl lookup: {len(all_seq_to_rl):,} sequences")

    # info4 breakdown
    print(f"\n  info4 breakdown:")
    print(f"    normal: {(snv_df['info4'] == 'normal').sum():,}")
    print(f"    variant: {(snv_df['info4'] == 'variant').sum():,}")
    print(f"    NaN (untagged): {snv_df['info4'].isna().sum():,}")

    # Extract pairs from ALL rows that have a mother sequence
    # This includes both tagged variants (info4='variant') AND untagged variants (info4=NaN)
    pairs = []

    # Get all rows with mother sequences (these are variants)
    variant_rows = snv_df[snv_df['mother'].notna()].copy()
    print(f"\n  Rows with mother sequence: {len(variant_rows):,}")

    tagged_count = 0
    untagged_count = 0

    for _, var_row in variant_rows.iterrows():
        mother_seq = str(var_row['mother'])
        var_seq = str(var_row['utr'])

        # Skip if mother not in lookup (no rl value available)
        if mother_seq not in all_seq_to_rl:
            continue

        # Skip if variant has no rl value
        if pd.isna(var_row['rl']):
            continue

        mother_rl = all_seq_to_rl[mother_seq]
        var_rl = float(var_row['rl'])

        # Find the edit (difference between mother and variant)
        edit_pos = -1
        edit_from = ''
        edit_to = ''
        edit_count = 0

        if len(mother_seq) == len(var_seq):
            for i, (m, v) in enumerate(zip(mother_seq, var_seq)):
                if m != v:
                    if edit_count == 0:  # Record first edit
                        edit_pos = i
                        edit_from = m
                        edit_to = v
                    edit_count += 1

        # Only include SNVs (single nucleotide changes)
        if edit_count != 1:
            continue

        # Track tagged vs untagged
        if var_row['info4'] == 'variant':
            tagged_count += 1
        else:
            untagged_count += 1

        pairs.append({
            'seq_a': mother_seq.replace('T', 'U'),
            'seq_b': var_seq.replace('T', 'U'),
            'edit_type': 'SNV',
            'edit_position': edit_pos,
            'edit_from': edit_from.replace('T', 'U'),
            'edit_to': edit_to.replace('T', 'U'),
            'value_a': mother_rl,
            'value_b': var_rl,
            'delta': var_rl - mother_rl,
            'log2_delta': np.log2(var_rl) - np.log2(mother_rl),
            'property_name': 'MRL_5UTR',
            'genomic_pos': var_row.get('info1', ''),
            'rsid': var_row.get('info2', ''),
            'cell_type': 'HEK293T',
            'experiment_id': 'GSE114002',
            'source': 'tagged' if var_row['info4'] == 'variant' else 'untagged'
        })

    pairs_df = pd.DataFrame(pairs)

    # Remove duplicates (same seq_a, seq_b pair)
    before_dedup = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=['seq_a', 'seq_b'], keep='first')

    print(f"\n  Pairs extracted:")
    print(f"    From tagged variants (info4='variant'): {tagged_count:,}")
    print(f"    From untagged variants (info4=NaN): {untagged_count:,}")
    print(f"    Total before dedup: {before_dedup:,}")
    print(f"    Total after dedup: {len(pairs_df):,}")

    if len(pairs_df) > 0:
        print(f"  rl (MRL) range: [{pairs_df['value_a'].min():.2f}, {pairs_df['value_a'].max():.2f}]")
        print(f"  Δ-MRL range: [{pairs_df['delta'].min():.2f}, {pairs_df['delta'].max():.2f}]")
        print(f"  Mean |Δ-MRL|: {pairs_df['delta'].abs().mean():.3f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'mpra_5utr_pairs_long.csv'
    pairs_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

    return pairs_df


def process_random_library(egfp_paths: list, output_dir: Path):
    """
    Process the random 280k library to create a reference dataset.

    This can be used for:
    - Creating synthetic pairs by Hamming distance
    - Pre-training embedders
    - Baseline comparisons
    """
    print("\nProcessing random library...")

    all_data = []
    for path in egfp_paths:
        print(f"  Reading: {path.name}")
        df = pd.read_csv(path, compression='gzip')
        all_data.append(df[['utr', 'rl']])

    combined = pd.concat(all_data, ignore_index=True)

    # Average rl across replicates for same sequence
    averaged = combined.groupby('utr').agg({'rl': 'mean'}).reset_index()

    # Convert to RNA
    averaged['sequence'] = averaged['utr'].str.upper().str.replace('T', 'U')
    averaged['MRL'] = averaged['rl']
    averaged['log2_MRL'] = np.log2(averaged['MRL'])

    print(f"  Unique sequences: {len(averaged):,}")
    print(f"  MRL range: [{averaged['MRL'].min():.2f}, {averaged['MRL'].max():.2f}]")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'random_library_280k.csv'
    averaged[['sequence', 'MRL', 'log2_MRL']].to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return averaged


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process MPRA designed library")
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/rna/raw',
        help='Directory with raw GEO files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/rna/pairs',
        help='Output directory for pairs'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/rna/processed',
        help='Output directory for processed data'
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir)

    print("=" * 70)
    print("MPRA Designed Library Processing")
    print("=" * 70)

    # Process designed library for SNV pairs
    designed_path = raw_dir / 'GSM3130443_designed_library.csv.gz'
    if designed_path.exists():
        pairs_df = process_designed_library(designed_path, output_dir)
    else:
        print(f"ERROR: {designed_path} not found")
        return

    # Process random library (optional)
    egfp_files = list(raw_dir.glob('GSM3130435_egfp*.csv.gz')) + \
                 list(raw_dir.glob('GSM3130436_egfp*.csv.gz'))

    if egfp_files:
        process_random_library(egfp_files, processed_dir)

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
