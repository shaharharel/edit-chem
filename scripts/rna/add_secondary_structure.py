#!/usr/bin/env python3
"""
Add secondary structure predictions to ADAR editing dataset.

Uses ViennaRNA (RNAfold) to predict:
- Dot-bracket structure notation
- Minimum free energy (MFE)
- Whether the center adenosine is base-paired (important for ADAR which prefers dsRNA)

Uses multiprocessing for speed since we have ~3.8M sequences.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fold_sequence(seq: str) -> tuple:
    """
    Fold a single RNA sequence using ViennaRNA.

    Returns:
        tuple: (structure, mfe, center_paired)
    """
    import RNA

    if not seq or len(seq) == 0:
        return ('', 0.0, False)

    try:
        structure, mfe = RNA.fold(seq)

        # Check if center position is paired
        center_idx = len(seq) // 2
        center_char = structure[center_idx] if center_idx < len(structure) else '.'
        center_paired = center_char != '.'

        return (structure, mfe, center_paired)
    except Exception as e:
        logger.warning(f"Failed to fold sequence: {e}")
        return ('', 0.0, False)


def fold_batch(sequences: list) -> list:
    """Fold a batch of sequences."""
    return [fold_sequence(seq) for seq in sequences]


def process_chunk(args):
    """Process a chunk of sequences. Used by multiprocessing."""
    chunk_idx, sequences = args
    results = []
    for seq in sequences:
        results.append(fold_sequence(seq))
    return chunk_idx, results


def add_structure_sequential(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """Add secondary structure using sequential processing."""
    logger.info(f"Processing {len(df)} sequences sequentially...")

    structures = []
    mfes = []
    center_paired = []

    for i, seq in enumerate(tqdm(df['sequence'], desc="Folding")):
        struct, mfe, paired = fold_sequence(seq)
        structures.append(struct)
        mfes.append(mfe)
        center_paired.append(paired)

        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1}/{len(df)} sequences")

    df = df.copy()
    df['structure'] = structures
    df['mfe'] = mfes
    df['center_paired'] = center_paired

    return df


def add_structure_parallel(df: pd.DataFrame, n_workers: int = None, chunk_size: int = 1000) -> pd.DataFrame:
    """Add secondary structure using parallel processing."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    logger.info(f"Processing {len(df)} sequences with {n_workers} workers...")

    sequences = df['sequence'].tolist()
    n_seqs = len(sequences)

    # Create chunks
    chunks = []
    for i in range(0, n_seqs, chunk_size):
        chunks.append((i // chunk_size, sequences[i:i + chunk_size]))

    logger.info(f"Created {len(chunks)} chunks of size {chunk_size}")

    # Process in parallel
    results_dict = {}
    with Pool(n_workers) as pool:
        for chunk_idx, chunk_results in tqdm(
            pool.imap_unordered(process_chunk, chunks),
            total=len(chunks),
            desc="Folding chunks"
        ):
            results_dict[chunk_idx] = chunk_results

    # Reassemble results in order
    all_results = []
    for i in range(len(chunks)):
        all_results.extend(results_dict[i])

    # Add to dataframe
    df = df.copy()
    df['structure'] = [r[0] for r in all_results]
    df['mfe'] = [r[1] for r in all_results]
    df['center_paired'] = [r[2] for r in all_results]

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add secondary structure to ADAR dataset')
    parser.add_argument('--input', type=str,
                        default=str(project_root / 'data' / 'rna' / 'adar' / 'adar_editing_dataset.csv'),
                        help='Input CSV file')
    parser.add_argument('--output', type=str,
                        default=str(project_root / 'data' / 'rna' / 'adar' / 'adar_editing_dataset_with_structure.csv'),
                        help='Output CSV file')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Chunk size for parallel processing')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential processing instead of parallel')
    parser.add_argument('--sample', type=int, default=None,
                        help='Only process first N rows (for testing)')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows")

    if args.sample:
        logger.info(f"Sampling first {args.sample} rows for testing")
        df = df.head(args.sample)

    # Add structure
    if args.sequential:
        df = add_structure_sequential(df)
    else:
        df = add_structure_parallel(df, n_workers=args.workers, chunk_size=args.chunk_size)

    # Summary stats
    logger.info("\nStructure statistics:")
    logger.info(f"  Total sequences: {len(df)}")
    logger.info(f"  Center paired (dsRNA): {df['center_paired'].sum()} ({df['center_paired'].mean()*100:.1f}%)")
    logger.info(f"  MFE range: {df['mfe'].min():.1f} to {df['mfe'].max():.1f}")
    logger.info(f"  Mean MFE: {df['mfe'].mean():.1f}")

    # Compare edited vs non-edited
    edited = df[df['is_edited'] == 1]
    non_edited = df[df['is_edited'] == 0]

    logger.info("\nComparison edited vs non-edited:")
    logger.info(f"  Edited center paired: {edited['center_paired'].mean()*100:.1f}%")
    logger.info(f"  Non-edited center paired: {non_edited['center_paired'].mean()*100:.1f}%")
    logger.info(f"  Edited mean MFE: {edited['mfe'].mean():.1f}")
    logger.info(f"  Non-edited mean MFE: {non_edited['mfe'].mean():.1f}")

    # Save
    logger.info(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    logger.info("Done!")

    return df


if __name__ == '__main__':
    main()
