#!/usr/bin/env python3
"""
Extract SNV pairs from the random library.

Strategy:
- Group sequences by length
- For each position, create a "signature" (sequence with that position masked)
- Sequences with the same signature differ only at that position = SNV pair

This is O(n * L) instead of O(n²) where L is sequence length.
"""

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

def find_snv_pairs(df, max_pairs_per_seq=10, sample_size=None):
    """
    Find all SNV pairs in the dataset.

    Args:
        df: DataFrame with 'sequence' and 'MRL' columns
        max_pairs_per_seq: Max pairs to keep per sequence (to avoid explosion)
        sample_size: If set, sample this many sequences first

    Returns:
        DataFrame with pairs
    """
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled to {len(df):,} sequences")

    # Build sequence -> MRL lookup
    seq_to_mrl = dict(zip(df['sequence'], df['MRL']))
    sequences = df['sequence'].tolist()

    print(f"Processing {len(sequences):,} sequences...")

    # Group by length
    by_length = defaultdict(list)
    for seq in sequences:
        by_length[len(seq)].append(seq)

    print(f"Sequence lengths: {sorted(by_length.keys())}")

    pairs = []

    for seq_len, seqs in tqdm(by_length.items(), desc="Processing lengths"):
        if len(seqs) < 2:
            continue

        print(f"\n  Length {seq_len}: {len(seqs):,} sequences")

        # For each position, group by signature (seq with that pos masked)
        for pos in range(seq_len):
            # Create signature: seq[:pos] + '_' + seq[pos+1:]
            sig_to_seqs = defaultdict(list)

            for seq in seqs:
                sig = seq[:pos] + '_' + seq[pos+1:]
                sig_to_seqs[sig].append(seq)

            # Find groups with multiple sequences (= SNV pairs)
            for sig, group in sig_to_seqs.items():
                if len(group) < 2:
                    continue

                # Create pairs within group
                for i, seq_a in enumerate(group):
                    for seq_b in group[i+1:]:
                        # Verify they differ at exactly position pos
                        if seq_a[pos] != seq_b[pos]:
                            mrl_a = seq_to_mrl[seq_a]
                            mrl_b = seq_to_mrl[seq_b]

                            # Add pair in both directions for balance
                            pairs.append({
                                'seq_a': seq_a,
                                'seq_b': seq_b,
                                'edit_position': pos,
                                'edit_from': seq_a[pos],
                                'edit_to': seq_b[pos],
                                'value_a': mrl_a,
                                'value_b': mrl_b,
                                'delta': mrl_b - mrl_a,
                                'seq_len': seq_len
                            })
                            # Reverse pair
                            pairs.append({
                                'seq_a': seq_b,
                                'seq_b': seq_a,
                                'edit_position': pos,
                                'edit_from': seq_b[pos],
                                'edit_to': seq_a[pos],
                                'value_a': mrl_b,
                                'value_b': mrl_a,
                                'delta': mrl_a - mrl_b,
                                'seq_len': seq_len
                            })

        if len(pairs) % 10000 == 0 and len(pairs) > 0:
            print(f"    Found {len(pairs):,} pairs so far...")

    pairs_df = pd.DataFrame(pairs)
    return pairs_df


def main():
    print("=" * 70)
    print("EXTRACTING SNV PAIRS FROM RANDOM LIBRARY")
    print("=" * 70)

    # Load data
    input_path = Path("data/rna/processed/random_library_280k.csv")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} sequences")

    # Check for duplicates
    n_unique = df['sequence'].nunique()
    print(f"Unique sequences: {n_unique:,}")

    if n_unique < len(df):
        print("Removing duplicates...")
        df = df.drop_duplicates(subset='sequence', keep='first')

    # Find pairs from ALL sequences (no sampling)
    pairs_df = find_snv_pairs(df, sample_size=None)

    print(f"\n" + "=" * 70)
    print(f"RESULTS")
    print("=" * 70)

    if len(pairs_df) > 0:
        print(f"Total SNV pairs found: {len(pairs_df):,}")
        print(f"Delta range: [{pairs_df['delta'].min():.3f}, {pairs_df['delta'].max():.3f}]")
        print(f"Mean |delta|: {pairs_df['delta'].abs().mean():.3f}")

        # Mutation type distribution
        pairs_df['mutation_type'] = pairs_df['edit_from'] + '>' + pairs_df['edit_to']
        print(f"\nMutation type distribution:")
        print(pairs_df['mutation_type'].value_counts())

        # Save
        output_dir = Path("data/rna/pairs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "random_library_snv_pairs.csv"
        pairs_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    else:
        print("No SNV pairs found!")


if __name__ == '__main__':
    main()
